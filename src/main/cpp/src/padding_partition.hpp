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

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief Partitions rows of `t` according to the mapping specified by
 * `partition_map`.
 *
 * For each row at `i` in `t`, `partition_map[i]` indicates which partition row
 * `i` belongs to. `partition` creates a new table by rearranging the rows of
 * `t` such that rows in the same partition are contiguous. The returned table
 * is in ascending partition order from `[0, num_partitions)`. The order within
 * each partition is undefined.
 *
 * Returns a `vector<size_type>` of `num_partitions + 1` values that indicate
 * the starting position of each partition within the returned table, i.e.,
 * partition `i` starts at `offsets[i]` (inclusive) and ends at `offset[i+1]`
 * (exclusive). As a result, if value `j` in `[0, num_partitions)` does not
 * appear in `partition_map`, partition `j` will be empty, i.e.,
 * `offsets[j+1] - offsets[j] == 0`.
 *
 * Values in `partition_map` must be in the range `[0, num_partitions)`,
 * otherwise behavior is undefined.
 *
 * @throw cudf::logic_error when `partition_map` is a non-integer type
 * @throw cudf::logic_error when `partition_map.has_nulls() == true`
 * @throw cudf::logic_error when `partition_map.size() != t.num_rows()`
 *
 * @param t The table to partition
 * @param partition_map Non-nullable column of integer values that map each row
 * in `t` to it's partition.
 * @param num_partitions The total number of partitions
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Pair containing the reordered table and vector of `num_partitions +
 * 1` offsets to each partition such that the size of partition `i` is
 * determined by `offset[i+1] - offset[i]`.
 */
std::tuple<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>, std::vector<cudf::size_type>> padding_partition(
  cudf::table_view const& t,
  cudf::column_view const& partition_map,
  cudf::size_type num_partitions,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}
