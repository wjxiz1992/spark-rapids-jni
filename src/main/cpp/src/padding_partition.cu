/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "padding_partition.hpp"

//
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>

//
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

//
#include <cub/device/device_histogram.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>


namespace spark_rapids_jni {

struct dispatch_map_type {

  using column_view = cudf::column_view;
  using size_type = cudf::size_type;
  using table = cudf::table;
  using table_view = cudf::table_view;

  template <typename MapType>
  std::enable_if_t<cudf::is_index_type<MapType>(),
                   std::tuple<std::unique_ptr<table>, std::vector<size_type>, std::vector<size_type>>>
  operator()(table_view const& t,
             column_view const& partition_map,
             size_type num_partitions,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const
  {
    // Build a histogram of the number of rows in each partition
    rmm::device_uvector<size_type> histogram(num_partitions + 1, stream);
    std::size_t temp_storage_bytes{};
    std::size_t const num_levels = num_partitions + 1;
    size_type const lower_level  = 0;
    size_type const upper_level  = num_partitions;
    cub::DeviceHistogram::HistogramEven(nullptr,
                                        temp_storage_bytes,
                                        partition_map.begin<MapType>(),
                                        histogram.data(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        partition_map.size(),
                                        stream.value());

    rmm::device_buffer temp_storage(temp_storage_bytes, stream);

    cub::DeviceHistogram::HistogramEven(temp_storage.data(),
                                        temp_storage_bytes,
                                        partition_map.begin<MapType>(),
                                        histogram.data(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        partition_map.size(),
                                        stream.value());

    // do padding only if there exists nullable columns
    bool padding_or_not = std::any_of(t.begin(), t.end(), [](const cudf::column_view &cv)
                                      { return cv.nullable(); });
    size_type padding_rows = 0;

    if (padding_or_not) {
      // do padding only if there exists partition offset which % 8 != 0
      auto padding_delta = [] __host__ __device__(size_type v) {
        return ((v + 7) >> 3 << 3) - v;
      };
      padding_rows = thrust::reduce(
        rmm::exec_policy(stream), 
        thrust::make_transform_iterator(histogram.begin(), padding_delta),
        thrust::make_transform_iterator(histogram.end(), padding_delta)
      );
      padding_or_not = padding_rows > 0;
    }

    // building the offsets of each partition through padding and accumulation
    rmm::device_uvector<size_type> offsets(num_partitions + 1, stream);

    if (padding_or_not) {
      // padding the partitions' intervals with 8 before making offsets from them  
      thrust::transform(rmm::exec_policy(stream),
        histogram.begin(), histogram.end(), offsets.begin(),
        [] __device__(auto v) { return (v + 7) >> 3 << 3; });
    } else {
      thrust::transform(rmm::exec_policy(stream),
        histogram.begin(), histogram.end(), offsets.begin(),
        thrust::identity<size_type>());
    }

    // `histogram` was created with an extra entry at the end such that an
    // exclusive scan will put the total number of rows at the end
    thrust::exclusive_scan(
      rmm::exec_policy(stream), offsets.begin(), offsets.end(), offsets.begin());

    // Copy offsets to host before the transform below modifies it
    auto const partition_offsets = cudf::detail::make_std_vector_sync(offsets, stream);
    // Copy lengths to host before the transform below modifies it
    auto const partition_lengths = cudf::detail::make_std_vector_sync(
        cudf::device_span<size_type const>{histogram.data(), histogram.size() - 1},
        stream);

    // Unfortunately need to materialize the scatter map because
    // `detail::scatter` requires multiple passes through the iterator
    rmm::device_uvector<size_type> scatter_map(partition_map.size(), stream);

    // For each `partition_map[i]`, atomically increment the corresponding
    // partition offset to determine `i`s location in the output
    thrust::transform(rmm::exec_policy(stream),
                      partition_map.begin<MapType>(),
                      partition_map.end<MapType>(),
                      scatter_map.begin(),
                      [offset_data = offsets.data()] __device__(auto partition_number) {
                        return atomicAdd(&offset_data[partition_number], 1);
                      });

    // Apply scattering if padding is not necessary
    if (not padding_or_not) {
      auto scattered = cudf::detail::scatter(t, scatter_map, t, stream, mr);

      return {std::move(scattered), std::move(partition_offsets), std::move(partition_lengths)};
    }

    // Convert scattering to gathering for padding
    auto original_size = scatter_map.size();
    auto padded_size = scatter_map.size() + padding_rows;

    // initialize gather_map with index 0
    rmm::device_uvector<size_type> gather_map(padded_size, stream);
    thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                               gather_map.begin(),
                               gather_map.end(),
                               0);

    // Convert scatter map to a gather map
    thrust::scatter(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(original_size),
                    scatter_map.begin(),
                    gather_map.begin());

    auto gathered = cudf::detail::gather(t,
                                         cudf::device_span<size_type const>(gather_map),
                                         cudf::out_of_bounds_policy::DONT_CHECK,
                                         cudf::detail::negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);

    return {std::move(gathered), std::move(partition_offsets), std::move(partition_lengths)};
  }

  template <typename MapType, typename... Args>
  std::enable_if_t<not cudf::is_index_type<MapType>(),
                   std::tuple<std::unique_ptr<table>, std::vector<size_type>, std::vector<size_type>>>
  operator()(Args&&...) const
  {
    CUDF_FAIL("Unexpected, non-integral partition map.");
  }
};

std::tuple<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>, std::vector<cudf::size_type>> padding_partition(
  cudf::table_view const& t,
  cudf::column_view const& partition_map,
  cudf::size_type num_partitions,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(t.num_rows() == partition_map.size(),
               "Size mismatch between table and partition map.");
  CUDF_EXPECTS(not partition_map.has_nulls(), "Unexpected null values in partition_map.");

  if (num_partitions == 0 or t.num_rows() == 0) {
    // The output offsets vector must have size `num_partitions + 1` as per documentation.
    return {empty_like(t), std::vector<cudf::size_type>(num_partitions + 1, 0), std::vector<cudf::size_type>(num_partitions, 0)};
  }

  return cudf::type_dispatcher(
    partition_map.type(), dispatch_map_type{}, t, partition_map, num_partitions,
    cudf::get_default_stream(), mr);
}

}
