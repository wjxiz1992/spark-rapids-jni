/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.OrderByArg;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

public class TableOperationTest {

	@Test
	void testPartitionNoPadding() {
		try (Table t = new Table.TestBuilder()
				.column(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
				.build();
				 ColumnVector parts = ColumnVector
						 .fromInts(1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
				 PartitionedTable pt = TableOperation.paddingPartition(t, parts, 3)) {
			int[] partitions = pt.getPartitions();
			assertIntArrayEqual(new int[]{0, 0, 0, 5, 5, 10}, partitions);
			ColumnVector[] slicedColumns = pt.getTable().getColumn(0).slice(partitions);
			try (Table part1 = new Table(slicedColumns[1]);
					 Table part1Expected = new Table.TestBuilder().column(1, 3, 5, 7, 9).build();
					 Table part2 = new Table(slicedColumns[2]);
					 Table part2Expected = new Table.TestBuilder().column(2, 4, 6, 8, 10).build()) {
				AssertUtils.assertTablesAreEqual(part1Expected, part1);
				AssertUtils.assertTablesAreEqual(part2Expected, part2);
			} finally {
				for (ColumnVector c : slicedColumns) {
					c.close();
				}
			}
		}
	}


	@Test
	void testPaddingPartition() {
		try (Table t = new Table.TestBuilder()
				.column(null, 2, 3, 4, 5, 6, 7, 8, 9, null, 11, 12, 13, 14, 15, 16, 17, null)
				.build();
				 ColumnVector parts = ColumnVector
						 .fromInts(0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3);
				 PartitionedTable pt = TableOperation.paddingPartition(t, parts, 4)) {
			int[] partitions = pt.getPartitions();
			assertIntArrayEqual(new int[]{0, 7, 8, 16, 16, 17, 24, 26}, partitions);
			ColumnVector[] slicedColumns = pt.getTable().getColumn(0).slice(partitions);
			try (Table part1 = new Table(slicedColumns[0]);
					 Table part1Expected = new Table.TestBuilder().column(null, 2, 3, 4, 5, 6, 7).build();
					 Table part2 = new Table(slicedColumns[1]);
					 Table part2Expected = new Table.TestBuilder().column(8, 9, null, 11, 12, 13, 14, 15).build();
					 Table part3 = new Table(slicedColumns[2]);
					 Table part3Expected = new Table.TestBuilder().column(16).build();
					 Table part4 = new Table(slicedColumns[3]);
					 Table part4Expected = new Table.TestBuilder().column(17, null).build()) {
				AssertUtils.assertTablesAreEqual(part1Expected, part1);
				AssertUtils.assertTablesAreEqual(part2Expected, part2);
				AssertUtils.assertTablesAreEqual(part3Expected, part3);
				AssertUtils.assertTablesAreEqual(part4Expected, part4);
			} finally {
				for (ColumnVector c : slicedColumns) {
					c.close();
				}
			}
		}

		try (Table t = new Table.TestBuilder()
				.column(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
				.column("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", null, "R", "S", "T")
				.build();
				 ColumnVector parts = ColumnVector
						 .fromInts(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2);
				 PartitionedTable pt = TableOperation.paddingPartition(t, parts, 3)) {
			int[] partitions = pt.getPartitions();
			assertIntArrayEqual(new int[]{0, 17, 24, 27, 32, 33}, partitions);
			ColumnVector[] slicedCol0 = pt.getTable().getColumn(0).slice(partitions);
			ColumnVector[] slicedCol1 = pt.getTable().getColumn(1).slice(partitions);
			try (Table part1 = new Table(slicedCol0[0], slicedCol1[0]);
					 Table part1Expected = new Table.TestBuilder()
							 .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
							 .column("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q")
							 .build();
					 Table part2 = new Table(slicedCol0[1], slicedCol1[1]);
					 Table part2Expected = new Table.TestBuilder()
							 .column(18, 19, 20).column( null, "R", "S").build();
					 Table part3 = new Table(slicedCol0[2], slicedCol1[2]);
					 Table part3Expected = new Table.TestBuilder().column(21).column( "T").build()) {
				AssertUtils.assertTablesAreEqual(part1Expected, part1);
				AssertUtils.assertTablesAreEqual(part2Expected, part2);
				AssertUtils.assertTablesAreEqual(part3Expected, part3);
			} finally {
				for (ColumnVector c : slicedCol0) {
					c.close();
				}
				for (ColumnVector c : slicedCol1) {
					c.close();
				}
			}
		}
	}

	private void assertIntArrayEqual(int[] expected, int[] actual) {
		if (expected.length != actual.length) {
			throw new AssertionError(
					"expected length " + expected.length + "; actual length " + actual.length);
		}
		for (int i = 0; i < expected.length; i++) {
			if (expected[i] != actual[i]) {
				StringBuilder expectedDump = new StringBuilder();
				StringBuilder actualDump = new StringBuilder();
				for (int j = 0; j < expected.length; j++) {
					expectedDump.append(expected[j]).append(' ');
					actualDump.append(actual[j]).append(' ');
				}
				throw new AssertionError(
						"expected array: " + expectedDump + "; actual array: " + actualDump);
			}
		}
	}

}
