/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/generate_input.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cast_string.hpp>
#include <nvbench/nvbench.cuh>

namespace {

void run_float_to_string(nvbench::state& state, bool const json_string)
{
  auto const num_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const type_id         = static_cast<cudf::type_id>(state.get_int64("type_id"));
  data_profile const profile = data_profile_builder().no_validity();
  auto const table           = create_random_table({type_id}, row_count{num_rows}, profile);
  auto const stream          = cudf::get_default_stream();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto const result =
      spark_rapids_jni::float_to_string(table->view().column(0), json_string, stream);
  });
}

void float_to_string_ryu(nvbench::state& state) { run_float_to_string(state, false); }

void float_to_json_string_legacy_java(nvbench::state& state) { run_float_to_string(state, true); }

}  // namespace

NVBENCH_BENCH(float_to_string_ryu)
  .set_name("Float to String (Ryu)")
  .add_int64_axis("type_id",
                  {static_cast<int64_t>(cudf::type_id::FLOAT32),
                   static_cast<int64_t>(cudf::type_id::FLOAT64)})
  .add_int64_axis("num_rows", {1'000'000, 10'000'000});

NVBENCH_BENCH(float_to_json_string_legacy_java)
  .set_name("Float to JSON String (legacy Java)")
  .add_int64_axis("type_id",
                  {static_cast<int64_t>(cudf::type_id::FLOAT32),
                   static_cast<int64_t>(cudf::type_id::FLOAT64)})
  .add_int64_axis("num_rows", {1'000'000, 10'000'000});
