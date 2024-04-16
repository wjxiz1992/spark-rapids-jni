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

public class Profiler {
  private static final Logger LOG = LoggerFactory.getLogger(Profiler.class);
  private static long state_handle = 0;

  static {
    try {
      NativeDepsLoader.loadNativeDeps(new String[]{"cupti", "profilerjni"});
    } catch (Exception e) {
      LOG.error("Unable to load profiling libraries", e);
    }
  }

  public static void init() {
    if (state_handle == 0) {
      state_handle = nativeInit();
    } else {
      throw new IllegalStateException("Already initialized");
    }
  }

  public static void shutdown() {
    if (state_handle != 0) {
      nativeShutdown(state_handle);
      state_handle = 0;
    } else {
      throw new IllegalStateException("Not initialized");
    }
  }

  private static native long nativeInit();

  private static native void nativeShutdown(long handle);
}
