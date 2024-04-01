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

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "jni_utils.hpp"
#include "padding_partition.hpp"

extern "C" {

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_TableOperation_paddingPartition(JNIEnv *env, jclass,
                                                                                              jlong input_table,
                                                                                              jlong partition_column,
                                                                                              jint number_of_partitions,
                                                                                              jintArray output_slices) {

  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, partition_column, "partition_column is null", NULL);
  JNI_NULL_CHECK(env, output_slices, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, number_of_partitions > 0, "number_of_partitions is zero", NULL);

  try {
    cudf::jni::auto_set_device(env);
    auto const n_input_table = reinterpret_cast<cudf::table_view const *>(input_table);
    auto const n_part_column = reinterpret_cast<cudf::column_view const *>(partition_column);

    std::unique_ptr<cudf::table> table;
    std::vector<cudf::size_type> part_offsets;
    std::vector<cudf::size_type> part_lengths;
    std::tie(table, part_offsets, part_lengths) =
      spark_rapids_jni::padding_partition(*n_input_table, *n_part_column, number_of_partitions);

    // construct partition slices (starts and ends) from offsets and lengths
    // For example:
    //  offsets [0, 8, 16, 24, 40]
    //  lengths [7, 8, 4, 10]
    //  slices  [0, 7, 8, 16, 16, 20, 24, 34]
    auto part_slices = std::vector<cudf::size_type>(2 * number_of_partitions);
    for (size_t i = 0; i < part_offsets.size() - 1; ++i) {
      part_slices[i * 2] = part_offsets[i];
      part_slices[i * 2 + 1] = part_offsets[i] + part_lengths[i];
    }

    cudf::jni::native_jintArray n_output_slices(env, output_slices);
    std::copy(part_slices.begin(), part_slices.end(), n_output_slices.begin());

    return cudf::jni::convert_table_for_return(env, table);
  }
  CATCH_STD(env, NULL);
}

}
