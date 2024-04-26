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

import ai.rapids.cudf.NativeDepsLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

public class Profiler {
  private static final Logger LOG = LoggerFactory.getLogger(Profiler.class);
  private static final long DEFAULT_WRITE_BUFFER_SIZE = 1024 * 1024;
  private static final int DEFAULT_FLUSH_PERIOD_MILLIS = 0;
  private static DataWriter writer = null;

  public static void init(DataWriter w) {
    init(w, DEFAULT_WRITE_BUFFER_SIZE, DEFAULT_FLUSH_PERIOD_MILLIS);
  }

  public static void init(DataWriter w, long writeBufferSize, int flushPeriodMillis) {
    if (writer == null) {
      File libPath;
      try {
        libPath = NativeDepsLoader.loadNativeDep("profilerjni", true);
      } catch (IOException e) {
        throw new RuntimeException("Error loading profiler library", e);
      }
      nativeInit(libPath.getAbsolutePath(), w, writeBufferSize, flushPeriodMillis);
      writer = w;
    } else {
      throw new IllegalStateException("Already initialized");
    }
  }

  public static void shutdown() {
    if (writer != null) {
      nativeShutdown();
      try {
        writer.close();
      } catch (Exception e) {
        throw new RuntimeException("Error closing writer", e);
      } finally {
        writer = null;
      }
    }
  }

  private static native void nativeInit(String libPath, DataWriter writer,
                                        long writeBufferSize, int flushPeriodMillis);

  private static native void nativeShutdown();

  public interface DataWriter extends AutoCloseable {
    void write(ByteBuffer data);
  }
}
