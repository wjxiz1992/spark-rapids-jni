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


import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Table;

public final class PartitionedTable implements AutoCloseable {
	private final Table table;
	private final int[] partitionSlices;

	/**
	 * The package-private constructor is only called by the partition method in Table
	 * .TableOperation.partition
	 * @param table            - {@link Table} which contains the partitioned data
	 * @param partitionSlices - This param is used to populate the offsets into the returned table
	 *                         where partitionSlices[i] indicates the starting position of
	 *                         partition 'i'
	 */
	PartitionedTable(Table table, int[] partitionSlices) {
		this.table = table;
		this.partitionSlices = partitionSlices;
	}

	public Table getTable() {
		return table;
	}

	public ColumnVector getColumn(int index) {
		return table.getColumn(index);
	}

	public long getNumberOfColumns() {
		return table.getNumberOfColumns();
	}

	public long getRowCount() {
		return table.getRowCount();
	}

	@Override
	public void close() {
		table.close();
	}

	/**
	 * This method returns the partitions on this table. partitionSlices[i] indicates the
	 * starting position of partition 'i' in the partitioned table. Size of the partitions can
	 * be calculated by the next offset
	 * Ex:
	 * partitionSlices[0, 12, 12, 49] indicates 4 partitions with the following sizes
	 * partition[0] - 12
	 * partition[1] - 0 (is empty)
	 * partition[2] - 37
	 * partition[3] has the remaining values of the table (N-49)
	 */
	public int[] getPartitions() {
		return partitionSlices;
	}
}
