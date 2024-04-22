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

constexpr size_t ALIGN_BYTES = 8;
constexpr size_t BUFF_SIZE = 8 * 1024 * 1024;
constexpr char const* DATA_WRITER_CLASS = "com/nvidia/spark/rapids/jni/Profiler$DataWriter";

struct profile_buffer {
  profile_buffer() : _size(BUFF_SIZE), _valid_size(0) {
    auto err = posix_memalign(reinterpret_cast<void**>(_data), ALIGN_BYTES, _size);
    if (err != 0) {
      std::cerr << "PROFILER: FAILED TO ALLOCATE CUPTI BUFFER: " << strerror(err) << std::endl;
      _data = nullptr;
      _size = 0;
    }
  }

  profile_buffer(uint8_t* data, size_t valid_size)
   : _data(data), _size(valid_size ? BUFF_SIZE : 0), _valid_size(_valid_size) {}

  void release(uint8_t** data_ptr_ptr, size_t* size_ptr) {
    *data_ptr_ptr = _data;
    *size_ptr = _size;
    _data = nullptr;
    _size = 0;
  }

  ~profile_buffer() {
    free(_data);
    _data = nullptr;
    _size = 0;
  }

  uint8_t const* data() const { return _data; }
  uint8_t* data() { return _data; }
  size_t size() const { return _size; }
  size_t valid_size() const { return _valid_size; }
  void set_valid_size(size_t size) { _valid_size = size; }

private:
  uint8_t* _data;
  size_t _size;
  size_t _valid_size;
};

struct completed_buffer_queue {
  std::unique_ptr<profile_buffer> get() {
    std::unique_lock lock(_lock);
    _cv.wait(lock, [&]{ return _shutdown || _buffers.size() > 0; });
    if (_buffers.size() > 0) {
      auto result = std::move(_buffers.front());
      _buffers.pop();
      return result;
    }
    return std::make_unique<profile_buffer>();
  }

  void put(std::unique_ptr<profile_buffer>&& buffer) {
    std::unique_lock lock(_lock);
    _buffers.push(std::move(buffer));
    lock.unlock();
    _cv.notify_one();
  }

  void shutdown() {
    std::unique_lock lock(_lock);
    _shutdown = true;
    lock.unlock();
    _cv.notify_one();
  }

private:
  std::mutex _lock;
  std::condition_variable _cv;
  std::queue<std::unique_ptr<profile_buffer>> _buffers;
  bool _shutdown = false;
};

struct free_buffer_tracker {
  std::unique_ptr<profile_buffer> get() {
    {
      std::lock_guard lock(_lock);
      if (_buffers.size() > 0) {
        auto result = std::move(_buffers.top());
        _buffers.pop();
        return result;
      }
    }
    return std::make_unique<profile_buffer>();
  }

  void put(std::unique_ptr<profile_buffer>&& buffer) {
    buffer->set_valid_size(0);
    std::lock_guard lock(_lock);
    if (_buffers.size() < NUM_CACHED_BUFFERS) {
      _buffers.push(std::move(buffer));
    } else {
      buffer.reset(nullptr);
    }
  }

private:
  static constexpr size_t NUM_CACHED_BUFFERS = 2;
  std::mutex _lock;
  std::stack<std::unique_ptr<profile_buffer>> _buffers;
};

struct subscriber_state {
  JNIEnv* jni;
  CUpti_SubscriberHandle subscriber_handle;
  bool has_cupti_callback_errored;
  jobject j_writer;
  std::thread writer_thread;
  free_buffer_tracker free_buffers;
  completed_buffer_queue completed_buffers;

  subscriber_state(JNIEnv* env)
    : jni(env), has_cupti_callback_errored(false) {}
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
          std::cerr << "Error flushing CUPTI activity on device reset: " << get_cupti_error(rc) << std::endl;
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
    std::cerr << "ERROR HANDLING CALLBACK: " << get_cupti_error(rc) << std::endl;
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
    std::cerr << "GOT A BUFFER OF DATA FROM CUPTI, SIZE: " << valid_size << std::endl;
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
  auto buffer = State->free_buffers.get();
  buffer->release(buffer_ptr_ptr, size_ptr);
}

void CUPTIAPI buffer_completed_callback(CUcontext, uint32_t,
    uint8_t* buffer, size_t buffer_size, size_t valid_size)
{
  State->completed_buffers.put(std::make_unique<profile_buffer>(buffer, valid_size));
}

void cache_writer_callback_method(JNIEnv* env)
{
  auto cls = env->FindClass(DATA_WRITER_CLASS);
  if (!cls) {
    throw std::runtime_error(std::string("Failed to locate class: ") + DATA_WRITER_CLASS);
  }
  Data_writer_write_method = env->GetMethodID(cls, "write", "(Ljava.nio.ByteBuffer;)V");
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

void process_buffer(profile_buffer& buffer)
{
  print_buffer(buffer.data(), buffer.valid_size());
}

void writer_thread(JavaVM* vm, jobject j_writer)
{
  JNIEnv* env = attach_to_jvm(vm);
  auto buffer = State->completed_buffers.get();
  while (buffer) {
    process_buffer(*buffer);
    State->free_buffers.put(std::move(buffer));
    buffer = State->completed_buffers.get();
  }
  vm->DetachCurrentThread();
}

}

extern "C" {

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_Profiler_nativeInit(JNIEnv* env, jclass,
    jstring j_lib_path, jobject j_writer)
{
  try {
    cache_writer_callback_method(env);
    setup_nvtx_env(env, j_lib_path);
    State = new subscriber_state(env);
    // grab a global reference to the writer instance so it isn't garbage collected while
    // we're trying to use it
    State->j_writer = static_cast<jobject>(env->NewGlobalRef(j_writer));
    State->writer_thread = std::thread(writer_thread, get_jvm(env), j_writer);
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
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT), "Error enabling context activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER), "Error enabling driver activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME), "Error enabling runtime activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY), "Error enabling memcpy activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET), "Error enabling memset activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME), "Error enabling name activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER), "Error enabling marker activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "Error enabling concurrent kernel activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD), "Error enabling overhead activity");
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
  std::cerr << "INIT NVTX V1 CALLED" << std::endl;
  CUptiResult res = cuptiNvtxInitialize(p);
  return (res == CUPTI_SUCCESS) ? 1 : 0;
}

extern "C" int InitializeInjectionNvtx2(void* p)
{
  std::cerr << "INIT NVTX V2 CALLED" << std::endl;
  CUptiResult res = cuptiNvtxInitialize2(p);
  return (res == CUPTI_SUCCESS) ? 1 : 0;
}