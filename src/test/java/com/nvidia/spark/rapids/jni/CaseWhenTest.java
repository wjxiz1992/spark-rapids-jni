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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class CaseWhenTest {

  @Test
  void selectIndexTest() {
    try (
        ColumnVector b0 = ColumnVector.fromBooleans(
            true, false, false, false);
        ColumnVector b1 = ColumnVector.fromBooleans(
            true, true, false, false);
        ColumnVector b2 = ColumnVector.fromBooleans(
            false, false, true, false);
        ColumnVector b3 = ColumnVector.fromBooleans(
            true, true, true, false);
        ColumnVector expected = ColumnVector.fromInts(0, 1, 2, 4)) {
      ColumnVector[] boolColumns = new ColumnVector[] { b0, b1, b2, b3 };
      try (ColumnVector actual = CaseWhen.selectFirstTrueIndex(boolColumns)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void selectTest() {
    try (ColumnVector values = ColumnVector.fromStrings(
        "s0", "s1", "s2", "s3");
        ColumnVector selects = ColumnVector.fromInts(0, 1, 2, 3, 3, 2, 1, 0, 4, 5, 6);
        ColumnVector expected = ColumnVector.fromStrings("s0", "s1", "s2", "s3", "s3", "s2", "s1", "s0", null, null,
            null);
        ColumnVector actual = CaseWhen.selectFromIndex(values, selects)) {
      assertColumnsAreEqual(expected, actual);
    }
  }
}
