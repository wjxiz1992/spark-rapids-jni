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

public class Profiler {
  private static final Logger LOG = LoggerFactory.getLogger(Profiler.class);
  private static boolean isInitialized = false;

  public static void init() {
    if (!isInitialized) {
      File libPath;
      try {
        libPath = NativeDepsLoader.loadNativeDep("profilerjni", true);
      } catch (IOException e) {
        throw new RuntimeException("Error loading profiler library", e);
      }
      nativeInit(libPath.getAbsolutePath());
      isInitialized = true;
    } else {
      throw new IllegalStateException("Already initialized");
    }
  }

  public static void shutdown() {
    if (isInitialized) {
      nativeShutdown();
      isInitialized = false;
    } else {
      throw new IllegalStateException("Not initialized");
    }
  }

  private static native void nativeInit(String libPath);

  private static native void nativeShutdown();
}
