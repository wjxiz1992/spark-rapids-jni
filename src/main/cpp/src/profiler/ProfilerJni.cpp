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

#include "profiler_generated.h"

#include <cupti.h>
#include <jni.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <stack>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

// HACK - FIXME
#define JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val)                                                 \
  {                                                                                                \
    if (env->ExceptionOccurred()) {                                                                \
      return ret_val;                                                                              \
    }                                                                                              \
  }

#define JNI_THROW_NEW(env, class_name, message, ret_val)                                           \
  {                                                                                                \
    jclass ex_class = env->FindClass(class_name);                                                  \
    if (ex_class == NULL) {                                                                        \
      return ret_val;                                                                              \
    }                                                                                              \
    env->ThrowNew(ex_class, message);                                                              \
    return ret_val;                                                                                \
  }

#define CATCH_STD_CLASS(env, class_name, ret_val)                                                  \
  catch (const std::exception &e) {                                                                \
    JNI_THROW_NEW(env, class_name, e.what(), ret_val)                                              \
  }

#define CATCH_STD(env, ret_val) CATCH_STD_CLASS(env, "java/lang/RuntimeException", ret_val)

// END HACK - FIXME

namespace {

constexpr char const* DATA_WRITER_CLASS = "com/nvidia/spark/rapids/jni/Profiler$DataWriter";
constexpr uint32_t PROFILE_VERSION = 1;

// TODO: Timer-based flushing

JavaVM* get_jvm(JNIEnv* env)
{
  JavaVM* vm;
  if (env->GetJavaVM(&vm) != 0) {
    throw std::runtime_error("Unable to get JavaVM");
  }
  return vm;
}

JNIEnv* attach_to_jvm(JavaVM* vm)
{
  JavaVMAttachArgs args;
  args.version = JNI_VERSION_1_6;
  args.name = const_cast<char*>("profiler writer");
  args.group = nullptr;
  JNIEnv* env;
  if (vm->AttachCurrentThread(reinterpret_cast<void**>(&env), &args) != JNI_OK) {
    char const* msg = "PROFILER: UNABLE TO ATTACH TO JVM";
    std::cerr << msg << std::endl;
    throw std::runtime_error(msg);
  }
  return env;
}

struct profile_buffer {
  explicit profile_buffer(size_t size) : size_(size), valid_size_(0) {
    auto err = posix_memalign(reinterpret_cast<void**>(&data_), ALIGN_BYTES, size_);
    if (err != 0) {
      std::cerr << "PROFILER: FAILED TO ALLOCATE CUPTI BUFFER: " << strerror(err) << std::endl;
      data_ = nullptr;
      size_ = 0;
    }
    std::cerr << "PROFILER: ALLOCATED BUFFER at " << static_cast<void*>(data_) << " SIZE=" << size_ << std::endl;
  }

  profile_buffer(uint8_t* data, size_t size, size_t valid_size)
   : data_(data), size_(size), valid_size_(valid_size) {
    std::cerr << "PROFILER: ACQUIRING BUFFER at " << static_cast<void*>(data_)
      << " SIZE=" << size_
      << " VALID=" << valid_size_ << std::endl;
  }

  void release(uint8_t** data_ptr_ptr, size_t* size_ptr) {
    std::cerr << "PROFILER: RELEASING BUFFER at " << static_cast<void*>(data_) << " SIZE=" << size_ << std::endl;
    *data_ptr_ptr = data_;
    *size_ptr = size_;
    data_ = nullptr;
    size_ = 0;
  }

  ~profile_buffer() {
    std::cerr << "PROFILER: FREEING BUFFER at " << static_cast<void*>(data_) << " SIZE=" << size_ << std::endl;
    free(data_);
    data_ = nullptr;
    size_ = 0;
  }

  uint8_t const* data() const { return data_; }
  uint8_t* data() { return data_; }
  size_t size() const { return size_; }
  size_t valid_size() const { return valid_size_; }
  void set_valid_size(size_t size) { valid_size_ = size; }

private:
  static constexpr size_t ALIGN_BYTES = 8;
  uint8_t* data_;
  size_t size_;
  size_t valid_size_;
};

struct completed_buffer_queue {
  std::unique_ptr<profile_buffer> get() {
    std::unique_lock lock(lock_);
    cv_.wait(lock, [this]{ return shutdown_ || buffers_.size() > 0; });
    if (buffers_.size() > 0) {
      auto result = std::move(buffers_.front());
      buffers_.pop();
      return result;
    }
    return std::unique_ptr<profile_buffer>(nullptr);
  }

  void put(std::unique_ptr<profile_buffer>&& buffer) {
    std::unique_lock lock(lock_);
    buffers_.push(std::move(buffer));
    lock.unlock();
    cv_.notify_one();
  }

  void shutdown() {
    std::unique_lock lock(lock_);
    shutdown_ = true;
    lock.unlock();
    cv_.notify_one();
  }

private:
  std::mutex lock_;
  std::condition_variable cv_;
  std::queue<std::unique_ptr<profile_buffer>> buffers_;
  bool shutdown_ = false;
};

struct free_buffer_tracker {
  explicit free_buffer_tracker(size_t size) : buffer_size_(size) {}

  std::unique_ptr<profile_buffer> get() {
    {
      std::lock_guard lock(lock_);
      if (buffers_.size() > 0) {
        auto result = std::move(buffers_.top());
        buffers_.pop();
        return result;
      }
    }
    return std::make_unique<profile_buffer>(buffer_size_);
  }

  void put(std::unique_ptr<profile_buffer>&& buffer) {
    buffer->set_valid_size(0);
    std::lock_guard lock(lock_);
    if (buffers_.size() < NUM_CACHED_BUFFERS) {
      buffers_.push(std::move(buffer));
    } else {
      std::cerr << "PROFILER: FREEING A BUFFER" << std::endl;
      buffer.reset(nullptr);
    }
  }

private:
  static constexpr size_t NUM_CACHED_BUFFERS = 2;
  std::mutex lock_;
  std::stack<std::unique_ptr<profile_buffer>> buffers_;
  size_t buffer_size_;
};

void writer_thread_process(JavaVM* vm);

struct subscriber_state {
  JNIEnv* writer_env;
  CUpti_SubscriberHandle subscriber_handle;
  bool has_cupti_callback_errored;
  // TODO: Move serializer to its own class and file
  jobject j_writer;
  flatbuffers::FlatBufferBuilder fb_builder;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::ApiActivity>> api_offsets;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::DeviceActivity>> device_offsets;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::DroppedRecords>> dropped_offsets;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::KernelActivity>> kernel_offsets;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::MarkerActivity>> marker_offsets;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::MarkerData>> marker_data_offsets;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::MemcpyActivity>> memcpy_offsets;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::MemsetActivity>> memset_offsets;
  std::vector<flatbuffers::Offset<spark_rapids_jni::profiler::OverheadActivity>> overhead_offsets;
  std::thread writer_thread;
  free_buffer_tracker free_buffers;
  completed_buffer_queue completed_buffers;
  size_t flush_threshold;

  subscriber_state(JNIEnv* env, jobject writer, size_t buffer_size)
    : writer_env(nullptr), j_writer(nullptr), has_cupti_callback_errored(false),
      fb_builder(buffer_size), free_buffers(buffer_size), flush_threshold(buffer_size) {}
};


subscriber_state* State = nullptr;
jclass Data_writer_jclass;
jmethodID Data_writer_write_method;


char const* get_cupti_error(CUptiResult rc)
{
  char const* err;
  if (cuptiGetResultString(rc, &err) != CUPTI_SUCCESS) {
    err = "UNKNOWN";
  }
  return err;
}

void check_cupti(CUptiResult rc, std::string msg)
{
  if (rc != CUPTI_SUCCESS) {
    throw std::runtime_error(msg + ": " + get_cupti_error(rc));
  }
}

//void domain_state_callback(CUpti_CallbackId callback_id, CUpti_StateData const* data_ptr)
//{
//  switch (callback_id) {
//    case CUPTI_CBID_STATE_FATAL_ERROR:
//    {
//      auto error = get_cupti_error(data_ptr->notification.result);
//      std::cerr << "CUPTI reported a fatal error: " << error << std::endl;
//      if (data_ptr->notification.message != nullptr) {
//        std::cerr << "CUPTI: " << data_ptr->notification.message << std::endl;
//      }
//    }
//    case CUPTI_CBID_STATE_ERROR:
//    {
//      auto error = get_cupti_error(data_ptr->notification.result);
//      std::cerr << "CUPTI reported an error: " << error << std::endl;
//      if (data_ptr->notification.message != nullptr) {
//        std::cerr << "CUPTI: " << data_ptr->notification.message << std::endl;
//      }
//    }
//    case CUPTI_CBID_STATE_WARNING:
//    {
//      auto error = get_cupti_error(data_ptr->notification.result);
//      std::cerr << "CUPTI reported a warning: " << error << std::endl;
//      if (data_ptr->notification.message != nullptr) {
//        std::cerr << "CUPTI: " << data_ptr->notification.message << std::endl;
//      }
//    }
//    default:
//      std::cer << "Ignoring CUPTI domain state callback for " << callback_id << std::endl;
//      break;
//  }
//}

void domain_runtime_callback(CUpti_CallbackId callback_id, CUpti_CallbackData const* data_ptr)
{
  switch (callback_id) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020:
      if (data_ptr->callbackSite == CUPTI_API_ENTER) {
        auto rc = cuptiActivityFlushAll(0);
        if (rc != CUPTI_SUCCESS) {
          std::cerr << "PROFILER: Error flushing CUPTI activity on device reset: " << get_cupti_error(rc) << std::endl;
        }
      }
      break;
    default:
      break;
  }
}

void CUPTIAPI callback_handler(void*, CUpti_CallbackDomain domain,
    CUpti_CallbackId callback_id, const void* callback_data_ptr)
{
  auto rc = cuptiGetLastError();
  if (rc != CUPTI_SUCCESS && !State->has_cupti_callback_errored) {
    //State->has_cupti_callback_errored = true;
    std::cerr << "PROFILER: ERROR HANDLING CALLBACK: " << get_cupti_error(rc) << std::endl;
    return;
  }

  switch (domain) {
//    case CUPTI_CB_DOMAIN_STATE:
//    {
//      auto domain_data = static_cast<CUpti_StateData const *>(callback_data_ptr);
//      domain_state_callback(callback_id, domain_data);
//      break;
//    }
    case CUPTI_CB_DOMAIN_RUNTIME_API:
    {
      auto runtime_data = static_cast<CUpti_CallbackData const *>(callback_data_ptr);
      domain_runtime_callback(callback_id, runtime_data);
      break;
    }
    default:
      break;
  }
}

std::string activity_kind_to_string(CUpti_ActivityKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: return "CUPTI_ACTIVITY_KIND_MEMCPY";
    case CUPTI_ACTIVITY_KIND_MEMSET: return "CUPTI_ACTIVITY_KIND_MEMSET";
    case CUPTI_ACTIVITY_KIND_KERNEL: return "CUPTI_ACTIVITY_KIND_KERNEL";
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: return "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL";
    case CUPTI_ACTIVITY_KIND_DRIVER: return "CPUTI_ACTIVITY_KIND_DRIVER";
    case CUPTI_ACTIVITY_KIND_RUNTIME: return "CUPTI_ACTIVITY_KIND_RUNTIME";
    case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API: return "CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API";
    case CUPTI_ACTIVITY_KIND_EVENT: return "CUPTI_ACTIVITY_KIND_EVENT";
    case CUPTI_ACTIVITY_KIND_METRIC: return "CUPTI_ACTIVITY_KIND_METRIC";
    case CUPTI_ACTIVITY_KIND_DEVICE: return "CUPTI_ACTIVITY_KIND_DEVICE";
    case CUPTI_ACTIVITY_KIND_CONTEXT: return "CUPTI_ACTIVITY_KIND_CONTEXT";
    case CUPTI_ACTIVITY_KIND_NAME: return "CUPTI_ACTIVITY_KIND_NAME";
    case CUPTI_ACTIVITY_KIND_MARKER: return "CUPTI_ACTIVITY_KIND_MARKER";
    case CUPTI_ACTIVITY_KIND_MARKER_DATA: return "CUPTI_ACTIVITY_KIND_MARKER_DATA";
    case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR: return "CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR";
    case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS: return "CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS";
    case CUPTI_ACTIVITY_KIND_BRANCH: return "CUPTI_ACTIVITY_KIND_BRANCH";
    case CUPTI_ACTIVITY_KIND_OVERHEAD: return "CUPTI_ACTIVITY_KIND_OVERHEAD";
    case CUPTI_ACTIVITY_KIND_CDP_KERNEL: return "CUPTI_ACTIVITY_KIND_CDP_KERNEL";
    case CUPTI_ACTIVITY_KIND_PREEMPTION: return "CUPTI_ACTIVITY_KIND_PREEMPTION";
    case CUPTI_ACTIVITY_KIND_ENVIRONMENT: return "CUPTI_ACTIVITY_KIND_ENVIRONMENT";
    case CUPTI_ACTIVITY_KIND_EVENT_INSTANCE: return "CUPTI_ACTIVITY_KIND_EVENT_INSTANCE";
    case CUPTI_ACTIVITY_KIND_MEMCPY2: return "CUPTI_ACTIVITY_KIND_MEMCPY2";
    case CUPTI_ACTIVITY_KIND_METRIC_INSTANCE: return "CUPTI_ACTIVITY_KIND_METRIC_INSTANCE";
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION: return "CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION";
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: return "CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER";
    case CUPTI_ACTIVITY_KIND_FUNCTION: return "CUPTI_ACTIVITY_KIND_FUNCTION";
    case CUPTI_ACTIVITY_KIND_MODULE: return "CUPTI_ACTIVITY_KIND_MODULE";
    case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: return "CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE";
    case CUPTI_ACTIVITY_KIND_SHARED_ACCESS: return "CUPTI_ACTIVITY_KIND_SHARED_ACCESS";
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING: return "CUPTI_ACTIVITY_KIND_PC_SAMPLING";
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO: return "CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO";
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION: return "CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION";
    case CUPTI_ACTIVITY_KIND_OPENACC_DATA: return "CUPTI_ACTIVITY_KIND_OPENACC_DATA";
    case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH: return "CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH";
    case CUPTI_ACTIVITY_KIND_OPENACC_OTHER: return "CUPTI_ACTIVITY_KIND_OPENACC_OTHER";
    case CUPTI_ACTIVITY_KIND_CUDA_EVENT: return "CUPTI_ACTIVITY_KIND_CUDA_EVENT";
    case CUPTI_ACTIVITY_KIND_STREAM: return "CUPTI_ACTIVITY_KIND_STREAM";
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: return "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION";
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: return "CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION";
    case CUPTI_ACTIVITY_KIND_NVLINK: return "CUPTI_ACTIVITY_KIND_NVLINK";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE";
    case CUPTI_ACTIVITY_KIND_MEMORY: return "CUPTI_ACTIVITY_KIND_MEMORY";
    case CUPTI_ACTIVITY_KIND_PCIE: return "CUPTI_ACTIVITY_KIND_PCIE";
    case CUPTI_ACTIVITY_KIND_OPENMP: return "CUPTI_ACTIVITY_KIND_OPENMP";
    case CUPTI_ACTIVITY_KIND_MEMORY2: return "CUPTI_ACTIVITY_KIND_MEMORY2";
    case CUPTI_ACTIVITY_KIND_MEMORY_POOL: return "CUPTI_ACTIVITY_KIND_MEMORY_POOL";
    case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: return "CUPTI_ACTIVITY_KIND_GRAPH_TRACE";
    case CUPTI_ACTIVITY_KIND_JIT: return "CUPTI_ACTIVITY_KIND_JIT";
    default: return "UNKNOWN";
  }
}

std::string marker_flags_to_string(CUpti_ActivityFlag flags)
{
  std::string s("");
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS) {
    s += "INSTANTANEOUS ";
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_START) {
    s += "START ";
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_END) {
    s += "END ";
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE) {
    s += "SYNCACQUIRE ";
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS) {
    s += "SYNCACQUIRESUCCESS ";
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED) {
    s += "SYNCACQUIREFAILED ";
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE) {
    s += "SYNCRELEASE ";
  }
  return s;
}

std::string activity_object_kind_to_string(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS: return "PROCESS";
    case CUPTI_ACTIVITY_OBJECT_THREAD: return "THREAD";
    case CUPTI_ACTIVITY_OBJECT_DEVICE: return "DEVICE";
    case CUPTI_ACTIVITY_OBJECT_CONTEXT: return "CONTEXT";
    case CUPTI_ACTIVITY_OBJECT_STREAM: return "STREAM";
    case CUPTI_ACTIVITY_OBJECT_UNKNOWN:
    default:
      return "UNKNOWN";
  }
}

void print_buffer(uint8_t* buffer, size_t valid_size)
{
  if (valid_size > 0) {
    std::cerr << "PROFILER: GOT A BUFFER OF DATA FROM CUPTI, SIZE: " << valid_size << std::endl;
    CUpti_Activity* record_ptr = nullptr;
    auto rc = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    while (rc == CUPTI_SUCCESS) {
      std::cerr << "RECORD: " << activity_kind_to_string(record_ptr->kind) << std::endl;
      switch (record_ptr->kind) {
        case CUPTI_ACTIVITY_KIND_DRIVER:
        {
          auto api_record = reinterpret_cast<CUpti_ActivityAPI const*>(record_ptr);
          char const* name = nullptr;
          cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, api_record->cbid, &name);
          name = name ? name : "NULL";
          std::cerr << "  NAME: " << name << " THREAD: " << api_record->threadId << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_DEVICE:
        {
          auto device_record = reinterpret_cast<CUpti_ActivityDevice4 const*>(record_ptr);
          char const* name = device_record->name != nullptr ? device_record->name : "NULL";
          std::cerr << "  " << activity_kind_to_string(device_record->kind) << " " << name << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
          auto api_record = reinterpret_cast<CUpti_ActivityAPI const*>(record_ptr);
          char const* name = nullptr;
          cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api_record->cbid, &name);
          name = name ? name : "NULL";
          std::cerr << "  NAME: " << name << " THREAD: " << api_record->threadId << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER:
        {
          auto marker_record = reinterpret_cast<CUpti_ActivityMarker2 const*>(record_ptr);
          std::cerr << "  FLAGS: " << marker_flags_to_string(marker_record->flags)
            << " ID: " << marker_record->id
            << " OBJECTKIND: " << activity_object_kind_to_string(marker_record->objectKind)
            << " NAME: " << std::string(marker_record->name ? marker_record->name : "NULL")
            << " DOMAIN: " << std::string(marker_record->domain ? marker_record->domain : "NULL")
            << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER_DATA:
        {
          auto marker_record = reinterpret_cast<CUpti_ActivityMarkerData const*>(record_ptr);
          std::cerr << "  FLAGS: " << marker_flags_to_string(marker_record->flags)
            << " ID: " << marker_record->id
            << " COLOR: " << marker_record->color
            << " COLOR FLAG: " << marker_record->flags
            << " CATEGORY: " << marker_record->category
            << " DATA KIND: " << marker_record->payloadKind
            << " DATA: " << marker_record->payload.metricValueUint64 << "/" << marker_record->payload.metricValueDouble
            << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
          auto kernel_record = reinterpret_cast<CUpti_ActivityKernel8 const*>(record_ptr);
          std::cerr << "  NAME: " << kernel_record->name << std::endl;
        }
        default:
          break;
      }
      rc = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    }
  }
}

void CUPTIAPI buffer_requested_callback(uint8_t** buffer_ptr_ptr, size_t* size_ptr,
    size_t* max_num_records_ptr)
{
  std::cerr << "PROFILER: BUFFER REQUEST CALLBACK" << std::endl;
  *max_num_records_ptr = 0;
  auto buffer = State->free_buffers.get();
  buffer->release(buffer_ptr_ptr, size_ptr);
}

void CUPTIAPI buffer_completed_callback(CUcontext, uint32_t,
    uint8_t* buffer, size_t buffer_size, size_t valid_size)
{
  std::cerr << "PROFILER: BUFFER COMPLETED CALLBACK" << std::endl;
  State->completed_buffers.put(std::make_unique<profile_buffer>(buffer, buffer_size, valid_size));
}

void cache_writer_callback_method(JNIEnv* env)
{
  auto cls = env->FindClass(DATA_WRITER_CLASS);
  if (!cls) {
    throw std::runtime_error(std::string("Failed to locate class: ") + DATA_WRITER_CLASS);
  }
  Data_writer_write_method = env->GetMethodID(cls, "write", "(Ljava/nio/ByteBuffer;)V");
  if (!Data_writer_write_method) {
    throw std::runtime_error("Failed to locate data writer write method");
  }
  // Convert local reference to global so it cannot be garbage collected.
  Data_writer_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (!Data_writer_jclass) {
    throw std::runtime_error(std::string("Failed to create a global reference to ") + DATA_WRITER_CLASS);
  }
}

void free_writer_callback_cache(JNIEnv* env)
{
  if (Data_writer_jclass) {
    env->DeleteGlobalRef(Data_writer_jclass);
    Data_writer_jclass = nullptr;
  }
}

void setup_nvtx_env(JNIEnv* env, jstring j_lib_path)
{
  auto lib_path = env->GetStringUTFChars(j_lib_path, 0);
  if (lib_path == NULL) {
    throw std::runtime_error("Error getting library path");
  }
  setenv("NVTX_INJECTION64_PATH", lib_path, 1);
  env->ReleaseStringUTFChars(j_lib_path, lib_path);
}

void clear_fbb()
{
  State->fb_builder.Clear();
  State->api_offsets.clear();
  State->device_offsets.clear();
  State->dropped_offsets.clear();
  State->kernel_offsets.clear();
  State->marker_offsets.clear();
  State->marker_data_offsets.clear();
  State->memcpy_offsets.clear();
  State->memset_offsets.clear();
  State->overhead_offsets.clear();
}

// TODO: This should be a state method
void write_current_fb()
{
  auto fb_size = State->fb_builder.GetSize();
  if (fb_size > 0) {
    std::cerr << "PROFILER: sending " << fb_size << " bytes to writer" << std::endl;
    auto fb = State->fb_builder.GetBufferPointer();
    auto env = State->writer_env;
    auto bytebuf_obj = env->NewDirectByteBuffer(fb, fb_size);
    if (bytebuf_obj != nullptr) {
      env->CallVoidMethod(State->j_writer, Data_writer_write_method, bytebuf_obj);
    } else {
      std::cerr << "PROFILER: ERROR: Unable to create ByteBuffer for writer" << std::endl;
    }
  }
  clear_fbb();
}

// TODO: This should be a state method
void flush_activity_records()
{
  auto& fbb = State->fb_builder;
  if (fbb.GetSize() > 0) {
    using flatbuffers::Offset;
    using flatbuffers::Vector;
    Offset<Vector<Offset<spark_rapids_jni::profiler::ApiActivity>>> api_vec;
    Offset<Vector<Offset<spark_rapids_jni::profiler::DeviceActivity>>> device_vec;
    Offset<Vector<Offset<spark_rapids_jni::profiler::DroppedRecords>>> dropped_vec;
    Offset<Vector<Offset<spark_rapids_jni::profiler::KernelActivity>>> kernel_vec;
    Offset<Vector<Offset<spark_rapids_jni::profiler::MarkerActivity>>> marker_vec;
    Offset<Vector<Offset<spark_rapids_jni::profiler::MarkerData>>> marker_data_vec;
    Offset<Vector<Offset<spark_rapids_jni::profiler::MemcpyActivity>>> memcpy_vec;
    Offset<Vector<Offset<spark_rapids_jni::profiler::MemsetActivity>>> memset_vec;
    Offset<Vector<Offset<spark_rapids_jni::profiler::OverheadActivity>>> overhead_vec;
    if (State->api_offsets.size() > 0) {
      api_vec = fbb.CreateVector(State->api_offsets);
    }
    if (State->device_offsets.size() > 0) {
      device_vec = fbb.CreateVector(State->device_offsets);
    }
    if (State->dropped_offsets.size() > 0) {
      dropped_vec = fbb.CreateVector(State->dropped_offsets);
    }
    if (State->kernel_offsets.size() > 0) {
      kernel_vec = fbb.CreateVector(State->kernel_offsets);
    }
    if (State->marker_offsets.size() > 0) {
      marker_vec = fbb.CreateVector(State->marker_offsets);
    }
    if (State->marker_data_offsets.size() > 0) {
      marker_data_vec = fbb.CreateVector(State->marker_data_offsets);
    }
    if (State->memcpy_offsets.size() > 0) {
      memcpy_vec = fbb.CreateVector(State->memcpy_offsets);
    }
    if (State->memset_offsets.size() > 0) {
      memset_vec = fbb.CreateVector(State->memset_offsets);
    }
    if (State->overhead_offsets.size() > 0) {
      overhead_vec = fbb.CreateVector(State->overhead_offsets);
    }
    spark_rapids_jni::profiler::ActivityRecordsBuilder arb(fbb);
    arb.add_api(api_vec);
    arb.add_device(device_vec);
    arb.add_dropped(dropped_vec);
    arb.add_kernel(kernel_vec);
    arb.add_marker(marker_vec);
    arb.add_marker_data(marker_data_vec);
    arb.add_memcpy(memcpy_vec);
    arb.add_memset(memset_vec);
    arb.add_overhead(overhead_vec);
    auto r = arb.Finish();
    fbb.FinishSizePrefixed(r);
    write_current_fb();
  }
}

void maybe_flush_activity_records()
{
  if (State->fb_builder.GetSize() >= State->flush_threshold) {
    flush_activity_records();
  }
}

spark_rapids_jni::profiler::MarkerFlags marker_flags_to_fb(CUpti_ActivityFlag flags)
{
  uint8_t result = 0;
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS) {
    result |= spark_rapids_jni::profiler::MarkerFlags_Instantaneous;
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_START) {
    result |= spark_rapids_jni::profiler::MarkerFlags_Start;
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_END) {
    result |= spark_rapids_jni::profiler::MarkerFlags_End;
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE) {
    result |= spark_rapids_jni::profiler::MarkerFlags_SyncAcquire;
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS) {
    result |= spark_rapids_jni::profiler::MarkerFlags_SyncAcquireSuccess;
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED) {
    result |= spark_rapids_jni::profiler::MarkerFlags_SyncAcquireFailed;
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE) {
    result |= spark_rapids_jni::profiler::MarkerFlags_SyncRelease;
  }
  return static_cast<spark_rapids_jni::profiler::MarkerFlags>(result);
}

spark_rapids_jni::profiler::ChannelType
to_channel_type(CUpti_ChannelType t)
{
  switch (t) {
    case CUPTI_CHANNEL_TYPE_INVALID:
      return spark_rapids_jni::profiler::ChannelType_Invalid;
    case CUPTI_CHANNEL_TYPE_COMPUTE:
      return spark_rapids_jni::profiler::ChannelType_Compute;
    case CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY:
      return spark_rapids_jni::profiler::ChannelType_AsyncMemcpy;
    default:
      std::cerr << "PROFILER: Unrecognized channel type: " << t << std::endl;
      return spark_rapids_jni::profiler::ChannelType_Invalid;
  }
}

spark_rapids_jni::profiler::LaunchType
to_launch_type(uint8_t t)
{
  switch (t) {
    case CUPTI_ACTIVITY_LAUNCH_TYPE_REGULAR:
      return spark_rapids_jni::profiler::LaunchType_Regular;
    case CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE:
      return spark_rapids_jni::profiler::LaunchType_CooperativeSingleDevice;
    case CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE:
      return spark_rapids_jni::profiler::LaunchType_CooperativeMultiDevice;
    default:
      std::cerr << "PROFILER: Unrecognized launch type: " << t << std::endl;
      return spark_rapids_jni::profiler::LaunchType_Regular;
  }
}

spark_rapids_jni::profiler::MemcpyFlags
to_memcpy_flags(uint32_t flags)
{
  uint8_t result = 0;
  if (flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) {
    result |= spark_rapids_jni::profiler::MemcpyFlags_Async;
  }
  return static_cast<spark_rapids_jni::profiler::MemcpyFlags>(result);
}

spark_rapids_jni::profiler::MemcpyKind
to_memcpy_kind(uint8_t k)
{
  switch (k) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN:
      return spark_rapids_jni::profiler::MemcpyKind_Unknown;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return spark_rapids_jni::profiler::MemcpyKind_HtoD;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return spark_rapids_jni::profiler::MemcpyKind_DtoH;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return spark_rapids_jni::profiler::MemcpyKind_HtoA;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return spark_rapids_jni::profiler::MemcpyKind_AtoH;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return spark_rapids_jni::profiler::MemcpyKind_AtoA;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return spark_rapids_jni::profiler::MemcpyKind_AtoD;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return spark_rapids_jni::profiler::MemcpyKind_DtoA;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return spark_rapids_jni::profiler::MemcpyKind_DtoD;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return spark_rapids_jni::profiler::MemcpyKind_HtoH;
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return spark_rapids_jni::profiler::MemcpyKind_PtoP;
    default:
      std::cerr << "PROFILER: Unrecognized memcpy kind: " << k << std::endl;
      return spark_rapids_jni::profiler::MemcpyKind_Unknown;
  }
}

spark_rapids_jni::profiler::MemoryKind
to_memory_kind(uint8_t k)
{
  switch (k) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return spark_rapids_jni::profiler::MemoryKind_Unknown;
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return spark_rapids_jni::profiler::MemoryKind_Pageable;
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return spark_rapids_jni::profiler::MemoryKind_Pinned;
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return spark_rapids_jni::profiler::MemoryKind_Device;
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return spark_rapids_jni::profiler::MemoryKind_Array;
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
      return spark_rapids_jni::profiler::MemoryKind_Managed;
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
      return spark_rapids_jni::profiler::MemoryKind_DeviceStatic;
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
      return spark_rapids_jni::profiler::MemoryKind_ManagedStatic;
    default:
      std::cerr << "PROFILER: Unrecognized memory kind: " << k << std::endl;
      return spark_rapids_jni::profiler::MemoryKind_Unknown;
  }
}

spark_rapids_jni::profiler::MemsetFlags
to_memset_flags(uint32_t flags)
{
  uint8_t result = 0;
  if (flags & CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC) {
    result |= spark_rapids_jni::profiler::MemsetFlags_Async;
  }
  return static_cast<spark_rapids_jni::profiler::MemsetFlags>(result);
}

spark_rapids_jni::profiler::OverheadKind
to_overhead_kind(CUpti_ActivityOverheadKind k)
{
  switch (k) {
    case CUPTI_ACTIVITY_OVERHEAD_UNKNOWN:
      return spark_rapids_jni::profiler::OverheadKind_Unknown;
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return spark_rapids_jni::profiler::OverheadKind_DriverCompiler;
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return spark_rapids_jni::profiler::OverheadKind_CUptiBufferFlush;
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return spark_rapids_jni::profiler::OverheadKind_CUptiInstrumentation;
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return spark_rapids_jni::profiler::OverheadKind_CUptiResource;
    default:
      std::cerr << "PROFILER: Unrecognized overhead kind: " << k << std::endl;
      return spark_rapids_jni::profiler::OverheadKind_Unknown;
  }
}

spark_rapids_jni::profiler::PartitionedGlobalCacheConfig
to_partitioned_global_cache_config(CUpti_ActivityPartitionedGlobalCacheConfig c)
{
  switch (c) {
    case CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN:
      return spark_rapids_jni::profiler::PartitionedGlobalCacheConfig_Unknown;
    case CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED:
      return spark_rapids_jni::profiler::PartitionedGlobalCacheConfig_NotSupported;
    case CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF:
      return spark_rapids_jni::profiler::PartitionedGlobalCacheConfig_Off;
    case CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON:
      return spark_rapids_jni::profiler::PartitionedGlobalCacheConfig_On;
    default:
      std::cerr << "PROFILER: Unrecognized partitioned global cache config: " << c << std::endl;
      return spark_rapids_jni::profiler::PartitionedGlobalCacheConfig_Unknown;
  }
}

spark_rapids_jni::profiler::ShmemLimitConfig
to_shmem_limit_config(CUpti_FuncShmemLimitConfig c)
{
  switch (c) {
    case CUPTI_FUNC_SHMEM_LIMIT_DEFAULT:
      return spark_rapids_jni::profiler::ShmemLimitConfig_Default;
    case CUPTI_FUNC_SHMEM_LIMIT_OPTIN:
      return spark_rapids_jni::profiler::ShmemLimitConfig_Optin;
    default:
      std::cerr << "PROFILER: Unrecognized shmem limit config: " << c << std::endl;
      return spark_rapids_jni::profiler::ShmemLimitConfig_Default;
  }
}

flatbuffers::Offset<spark_rapids_jni::profiler::ActivityObjectId>
add_object_id(flatbuffers::FlatBufferBuilder& fbb, CUpti_ActivityObjectKind kind,
              CUpti_ActivityObjectKindId const& object_id)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
    case CUPTI_ACTIVITY_OBJECT_THREAD:
    {
      spark_rapids_jni::profiler::ActivityObjectIdBuilder aoib(fbb);
      aoib.add_process_id(object_id.pt.processId);
      if (kind == CUPTI_ACTIVITY_OBJECT_THREAD) {
        aoib.add_thread_id(object_id.pt.threadId);
      }
      return aoib.Finish();
    }
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    case CUPTI_ACTIVITY_OBJECT_STREAM:
    {
      spark_rapids_jni::profiler::ActivityObjectIdBuilder aoib(fbb);
      aoib.add_device_id(object_id.dcs.deviceId);
      if (kind == CUPTI_ACTIVITY_OBJECT_CONTEXT || kind == CUPTI_ACTIVITY_OBJECT_STREAM) {
        aoib.add_context_id(object_id.dcs.contextId);
        if (kind == CUPTI_ACTIVITY_OBJECT_STREAM) {
          aoib.add_stream_id(object_id.dcs.streamId);
        }
      }
      return aoib.Finish();
    }
    default:
      std::cerr << "PROFILER: Unrecognized object kind: " << kind << std::endl;
      return flatbuffers::Offset<spark_rapids_jni::profiler::ActivityObjectId>();
  }
}

void process_api_activity(CUpti_ActivityAPI const* r)
{
  auto api_kind = spark_rapids_jni::profiler::ApiKind_Runtime;
  if (r->kind == CUPTI_ACTIVITY_KIND_DRIVER) {
    api_kind = spark_rapids_jni::profiler::ApiKind_Driver;
  } else if (r->kind != CUPTI_ACTIVITY_KIND_RUNTIME) {
    std::cerr << "PROFILER: IGNORING API ACTIVITY RECORD KIND=" << activity_kind_to_string(r->kind) << std::endl;
    return;
  }
  spark_rapids_jni::profiler::ApiActivityBuilder aab(State->fb_builder);
  aab.add_kind(api_kind);
  aab.add_cbid(r->cbid);
  aab.add_start(r->start);
  aab.add_end(r->end);
  aab.add_process_id(r->processId);
  aab.add_thread_id(r->threadId);
  aab.add_correlation_id(r->correlationId);
  aab.add_return_value(r->returnValue);
  State->api_offsets.push_back(aab.Finish());
}

void process_device_activity(CUpti_ActivityDevice4 const* r)
{
  auto& fbb = State->fb_builder;
  auto name = fbb.CreateSharedString(r->name);
  spark_rapids_jni::profiler::DeviceActivityBuilder dab(fbb);
  dab.add_global_memory_bandwidth(r->globalMemoryBandwidth);
  dab.add_global_memory_size(r->globalMemorySize);
  dab.add_constant_memory_size(r->constantMemorySize);
  dab.add_l2_cache_size(r->l2CacheSize);
  dab.add_num_threads_per_warp(r->numThreadsPerWarp);
  dab.add_core_clock_rate(r->coreClockRate);
  dab.add_num_memcpy_engines(r->numMemcpyEngines);
  dab.add_num_multiprocessors(r->numMultiprocessors);
  dab.add_max_ipc(r->maxIPC);
  dab.add_max_warps_per_multiprocessor(r->maxWarpsPerMultiprocessor);
  dab.add_max_blocks_per_multiprocessor(r->maxBlocksPerMultiprocessor);
  dab.add_max_shared_memory_per_multiprocessor(r->maxSharedMemoryPerMultiprocessor);
  dab.add_max_registers_per_multiprocessor(r->maxRegistersPerMultiprocessor);
  dab.add_max_registers_per_block(r->maxRegistersPerBlock);
  dab.add_max_shared_memory_per_block(r->maxSharedMemoryPerBlock);
  dab.add_max_threads_per_block(r->maxThreadsPerBlock);
  dab.add_max_block_dim_x(r->maxBlockDimX);
  dab.add_max_block_dim_y(r->maxBlockDimY);
  dab.add_max_block_dim_z(r->maxBlockDimZ);
  dab.add_max_grid_dim_x(r->maxGridDimX);
  dab.add_max_grid_dim_y(r->maxGridDimY);
  dab.add_max_grid_dim_z(r->maxGridDimZ);
  dab.add_compute_capability_major(r->computeCapabilityMajor);
  dab.add_compute_capability_minor(r->computeCapabilityMinor);
  dab.add_id(r->id);
  dab.add_ecc_enabled(r->eccEnabled);
  dab.add_name(name);
  State->device_offsets.push_back(dab.Finish());
}

void process_dropped_records(size_t num_dropped)
{
  auto dropped = spark_rapids_jni::profiler::CreateDroppedRecords(State->fb_builder, num_dropped);
  State->dropped_offsets.push_back(dropped);
}

void process_kernel(CUpti_ActivityKernel8 const* r)
{
  auto& fbb = State->fb_builder;
  auto name = fbb.CreateSharedString(r->name);
  spark_rapids_jni::profiler::KernelActivityBuilder kab(fbb);
  kab.add_requested(r->cacheConfig.config.requested);
  kab.add_executed(r->cacheConfig.config.executed);
  kab.add_shared_memory_config(r->sharedMemoryConfig);
  kab.add_registers_per_thread(r->registersPerThread);
  kab.add_partitioned_global_cache_requested(
    to_partitioned_global_cache_config(r->partitionedGlobalCacheRequested));
  kab.add_partitioned_global_cache_executed(
    to_partitioned_global_cache_config(r->partitionedGlobalCacheExecuted));
  kab.add_start(r->start);
  kab.add_end(r->end);
  kab.add_completed(r->completed);
  kab.add_device_id(r->deviceId);
  kab.add_context_id(r->contextId);
  kab.add_stream_id(r->streamId);
  kab.add_grid_x(r->gridX);
  kab.add_grid_y(r->gridY);
  kab.add_grid_z(r->gridZ);
  kab.add_block_x(r->blockX);
  kab.add_block_y(r->blockY);
  kab.add_block_z(r->blockZ);
  kab.add_static_shared_memory(r->staticSharedMemory);
  kab.add_dynamic_shared_memory(r->dynamicSharedMemory);
  kab.add_local_memory_per_thread(r->localMemoryPerThread);
  kab.add_local_memory_total(r->localMemoryTotal);
  kab.add_correlation_id(r->correlationId);
  kab.add_grid_id(r->gridId);
  kab.add_name(name);
  kab.add_queued(r->queued);
  kab.add_submitted(r->submitted);
  kab.add_launch_type(to_launch_type(r->launchType));
  kab.add_is_shared_memory_carveout_requested(r->isSharedMemoryCarveoutRequested);
  kab.add_shared_memory_carveout_requested(r->sharedMemoryCarveoutRequested);
  kab.add_shared_memory_executed(r->sharedMemoryExecuted);
  kab.add_graph_node_id(r->graphNodeId);
  kab.add_shmem_limit_config(to_shmem_limit_config(r->shmemLimitConfig));
  kab.add_graph_id(r->graphId);
  kab.add_channel_id(r->channelID);
  kab.add_channel_type(to_channel_type(r->channelType));
  kab.add_cluster_x(r->clusterX);
  kab.add_cluster_y(r->clusterY);
  kab.add_cluster_z(r->clusterZ);
  kab.add_cluster_scheduling_policy(r->clusterSchedulingPolicy);
  kab.add_local_memory_total_v2(r->localMemoryTotal_v2);
  State->kernel_offsets.push_back(kab.Finish());
}

void process_marker_activity(CUpti_ActivityMarker2 const* r)
{
  auto& fbb = State->fb_builder;
  auto object_id = add_object_id(fbb, r->objectKind, r->objectId);
  auto has_name = r->name != nullptr;
  auto has_domain = r->name != nullptr;
  flatbuffers::Offset<flatbuffers::String> name;
  flatbuffers::Offset<flatbuffers::String> domain;
  if (has_name) {
    name = fbb.CreateSharedString(r->name);
  }
  if (has_domain) {
    domain = fbb.CreateSharedString(r->domain);
  }
  spark_rapids_jni::profiler::MarkerActivityBuilder mab(fbb);
  mab.add_flags(marker_flags_to_fb(r->flags));
  mab.add_timestamp(r->timestamp);
  mab.add_id(r->id);
  mab.add_object_id(object_id);
  mab.add_name(name);
  mab.add_domain(domain);
  State->marker_offsets.push_back(mab.Finish());
}

void process_marker_data(CUpti_ActivityMarkerData const* r)
{
  spark_rapids_jni::profiler::MarkerDataBuilder mdb(State->fb_builder);
  mdb.add_flags(marker_flags_to_fb(r->flags));
  mdb.add_id(r->id);
  mdb.add_color(r->color);
  mdb.add_category(r->category);
  State->marker_data_offsets.push_back(mdb.Finish());
}

void process_memcpy(CUpti_ActivityMemcpy5 const* r)
{
  spark_rapids_jni::profiler::MemcpyActivityBuilder mab(State->fb_builder);
  mab.add_copy_kind(to_memcpy_kind(r->copyKind));
  mab.add_src_kind(to_memory_kind(r->srcKind));
  mab.add_dst_kind(to_memory_kind(r->dstKind));
  mab.add_flags(to_memcpy_flags(r->flags));
  mab.add_bytes(r->bytes);
  mab.add_start(r->start);
  mab.add_end(r->end);
  mab.add_device_id(r->deviceId);
  mab.add_context_id(r->contextId);
  mab.add_stream_id(r->streamId);
  mab.add_correlation_id(r->correlationId);
  mab.add_runtime_correlation_id(r->runtimeCorrelationId);
  mab.add_graph_node_id(r->graphNodeId);
  mab.add_graph_id(r->graphId);
  mab.add_channel_id(r->channelID);
  mab.add_channel_type(to_channel_type(r->channelType));
  State->memcpy_offsets.push_back(mab.Finish());
}

void process_memset(CUpti_ActivityMemset4 const* r)
{
  spark_rapids_jni::profiler::MemsetActivityBuilder mab(State->fb_builder);
  mab.add_value(r->value);
  mab.add_bytes(r->bytes);
  mab.add_start(r->start);
  mab.add_end(r->end);
  mab.add_device_id(r->deviceId);
  mab.add_context_id(r->contextId);
  mab.add_stream_id(r->streamId);
  mab.add_correlation_id(r->correlationId);
  mab.add_flags(to_memset_flags(r->flags));
  mab.add_memory_kind(to_memory_kind(r->memoryKind));
  mab.add_graph_node_id(r->graphNodeId);
  mab.add_graph_id(r->graphId);
  mab.add_channel_id(r->channelID);
  mab.add_channel_type(to_channel_type(r->channelType));
  State->memset_offsets.push_back(mab.Finish());
}

void process_overhead(CUpti_ActivityOverhead const* r)
{
  auto& fbb = State->fb_builder;
  auto object_id = add_object_id(fbb, r->objectKind, r->objectId);
  spark_rapids_jni::profiler::OverheadActivityBuilder oab(fbb);
  oab.add_overhead_kind(to_overhead_kind(r->overheadKind));
  oab.add_object_id(object_id);
  oab.add_start(r->start);
  oab.add_end(r->end);
  State->overhead_offsets.push_back(oab.Finish());
}

void write_profile_header()
{
  auto& fbb = State->fb_builder;
  // TODO: This needs to be passed in by Java during init
  auto writer_version = fbb.CreateString("24.06.0");
  auto magic = fbb.CreateString("spark-rapids profile");
  auto header = spark_rapids_jni::profiler::CreateProfileHeader(fbb, magic, PROFILE_VERSION, writer_version);
  fbb.FinishSizePrefixed(header);
  write_current_fb();
}

void report_num_dropped_records()
{
  size_t num_dropped = 0;
  auto rc = cuptiActivityGetNumDroppedRecords(NULL, 0, &num_dropped);
  if (rc == CUPTI_SUCCESS && num_dropped > 0) {
    process_dropped_records(num_dropped);
  }
}

void process_cupti_buffer(uint8_t* buffer, size_t valid_size)
{
#if 1
  print_buffer(buffer, valid_size);
#endif

  report_num_dropped_records();
  if (valid_size > 0) {
    CUpti_Activity* record_ptr = nullptr;
    auto rc = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    while (rc == CUPTI_SUCCESS) {
      switch (record_ptr->kind) {
        case CUPTI_ACTIVITY_KIND_DEVICE:
        {
          auto device_record = reinterpret_cast<CUpti_ActivityDevice4 const*>(record_ptr);
          process_device_activity(device_record);
          break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
          auto api_record = reinterpret_cast<CUpti_ActivityAPI const*>(record_ptr);
          process_api_activity(api_record);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER:
        {
          auto marker = reinterpret_cast<CUpti_ActivityMarker2 const*>(record_ptr);
          process_marker_activity(marker);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER_DATA:
        {
          auto marker = reinterpret_cast<CUpti_ActivityMarkerData const*>(record_ptr);
          process_marker_data(marker);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
          auto r = reinterpret_cast<CUpti_ActivityMemcpy5 const*>(record_ptr);
          process_memcpy(r);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET:
        {
          auto r = reinterpret_cast<CUpti_ActivityMemset4 const*>(record_ptr);
          process_memset(r);
          break;
        }
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
          auto r = reinterpret_cast<CUpti_ActivityKernel8 const*>(record_ptr);
          process_kernel(r);
          break;
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
        {
          auto r = reinterpret_cast<CUpti_ActivityOverhead const*>(record_ptr);
          process_overhead(r);
          break;
        }
        default:
          std::cerr << "IGNORING ACTIVITY RECORD " << activity_kind_to_string(record_ptr->kind) << std::endl;
          break;
      }
      maybe_flush_activity_records();
      rc = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    }
  }
}

void writer_thread_process(JavaVM* vm)
{
  std::cerr << "PROFILER: WRITER THREAD START" << std::endl;
  State->writer_env = attach_to_jvm(vm);
  std::cerr << "WRITER THREAD JVM ATTACHED" << std::endl;
  std::cerr << "EMITTING HEADER" << std::endl;
  write_profile_header();
  auto buffer = State->completed_buffers.get();
  while (buffer) {
    process_cupti_buffer(buffer->data(), buffer->valid_size());
    State->free_buffers.put(std::move(buffer));
    buffer = State->completed_buffers.get();
  }
  flush_activity_records();
  std::cerr << "WRITER THREAD DETACHING" << std::endl;
  vm->DetachCurrentThread();
  std::cerr << "PROFILER: WRITER THREAD COMPLETED" << std::endl;
}

}

extern "C" {

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_Profiler_nativeInit(JNIEnv* env, jclass,
    jstring j_lib_path, jobject j_writer, jlong write_buffer_size, jint flush_period_millis)
{
  try {
    cache_writer_callback_method(env);
    setup_nvtx_env(env, j_lib_path);
    State = new subscriber_state(env, j_writer, write_buffer_size);
    // grab a global reference to the writer instance so it isn't garbage collected
    State->j_writer = static_cast<jobject>(env->NewGlobalRef(j_writer));
    if (State->j_writer == nullptr) {
      throw std::runtime_error("Unable to create a global reference to writer");
    }
    State->writer_thread = std::thread(writer_thread_process, get_jvm(env));
    auto rc = cuptiSubscribe(&State->subscriber_handle, callback_handler, nullptr);
    check_cupti(rc, "Error initializing CUPTI");
    rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020);
    check_cupti(rc, "Error enabling device reset callback");
//    rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_STATE,
//        CUPTI_CBID_STATE_FATAL_ERROR);
//    check_cupti(rc, "Error enabling fatal error callback");
//    rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_STATE,
//        CUPTI_CBID_STATE_ERROR);
//    check_cupti(rc, "Error enabling error callback");
//    rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_STATE,
//        CUPTI_CBID_STATE_WARNING);
//    check_cupti(rc, "Error enabling warning callback");
    rc = cuptiActivityRegisterCallbacks(buffer_requested_callback, buffer_completed_callback);
    check_cupti(rc, "Error registering activity buffer callbacks");

    //check_cupti(cuptiEnableDomain(1, State->subscriber_handle, CUPTI_CB_DOMAIN_NVTX), "Error enabling NVTX domain");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE), "Error enabling device activity");
    //check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT), "Error enabling context activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER), "Error enabling driver activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME), "Error enabling runtime activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY), "Error enabling memcpy activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET), "Error enabling memset activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME), "Error enabling name activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER), "Error enabling marker activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "Error enabling concurrent kernel activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD), "Error enabling overhead activity");

    if (flush_period_millis > 0) {
      std::cerr << "PROFILER: Flushing activity records every " << flush_period_millis << " milliseconds" << std::endl;
      // TODO: Does not seem to be working? May need manual, periodic flushing
      check_cupti(cuptiActivityFlushPeriod(flush_period_millis), "Error requesting periodic activity flush");
    }
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_Profiler_nativeShutdown(JNIEnv* env, jclass)
{
  try {
    if (State != nullptr) {
      auto unsub_rc = cuptiUnsubscribe(State->subscriber_handle);
      auto flush_rc = cuptiActivityFlushAll(1);
      State->completed_buffers.shutdown();
      State->writer_thread.join();
      env->DeleteGlobalRef(State->j_writer);
      delete State;
      State = nullptr;
      free_writer_callback_cache(env);
      check_cupti(unsub_rc, "Error unsubscribing from CUPTI");
      check_cupti(flush_rc, "Error flushing CUPTI records");
    }
  }
  CATCH_STD(env, );
}

}

/* Extern the CUPTI NVTX initialization APIs. The APIs are thread-safe */
extern "C" CUptiResult CUPTIAPI cuptiNvtxInitialize(void* pfnGetExportTable);
extern "C" CUptiResult CUPTIAPI cuptiNvtxInitialize2(void* pfnGetExportTable);

extern "C" int InitializeInjectionNvtx(void* p)
{
  std::cerr << "PROFILER: INIT NVTX V1 CALLED" << std::endl;
  CUptiResult res = cuptiNvtxInitialize(p);
  return (res == CUPTI_SUCCESS) ? 1 : 0;
}

extern "C" int InitializeInjectionNvtx2(void* p)
{
  std::cerr << "PROFILER: INIT NVTX V2 CALLED" << std::endl;
  CUptiResult res = cuptiNvtxInitialize2(p);
  return (res == CUPTI_SUCCESS) ? 1 : 0;
}
