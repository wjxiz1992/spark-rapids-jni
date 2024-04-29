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

/* A tool that converts a spark-rapids profile binary into other forms. */

#include "profiler_generated.h"
#include "spark_rapids_jni_version.h"

#include <flatbuffers/idl.h>

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace spark_rapids_jni::profiler {
extern char const* Profiler_Schema;
}

struct program_options {
  std::optional<std::filesystem::path> output_path;
  bool help = false;
  bool json = false;
  bool version = false;
};

void print_usage()
{
  std::cout << "spark_rapids_profile_converter [OPTION]... profilebin" << std::endl;
  std::cout << R"(
Converts the spark-rapids profile in profile.bin into other forms.

  -h, --help           show this usage message
  -j, --json           convert to JSON, default output is stdout
  -o, --output=PATH    use PATH as the output filename
  --version            print the version number
  )"
    << std::endl;
}

void print_version()
{
  std::cout << "spark_rapids_profile_converter " << SPARK_RAPIDS_JNI_VERSION << std::endl;
}

std::pair<program_options, std::vector<std::string_view>>
parse_options(std::vector<std::string_view> args)
{
  program_options opts{};
  std::string_view long_output("--output=");
  bool seen_output = false;
  auto argp = args.begin();
  while (argp != args.end()) {
    if (*argp == "-o" || *argp == "--output") {
      if (seen_output) {
        throw std::runtime_error("output path cannot be specified twice");
      }
      seen_output = true;
      if (++argp != args.end()) {
        opts.output_path = std::make_optional(*argp++);
      } else {
        throw std::runtime_error("missing argument for output path");
      }
    } else if (argp->substr(0, long_output.size()) == long_output) {
      if (seen_output) {
        throw std::runtime_error("output path cannot be specified twice");
      }
      seen_output = true;
      argp->remove_prefix(long_output.size());
      if (argp->empty()) {
        throw std::runtime_error("missing argument for output path");
      } else {
        opts.output_path = std::make_optional(*argp++);
      }
    } else if (*argp == "-h" || *argp == "--help") {
      opts.help = true;
      ++argp;
    } else if (*argp == "-j" || *argp == "--json") {
      opts.json = true;
      ++argp;
    } else if (*argp == "--version") {
      opts.version = true;
      ++argp;
    } else if (argp->empty()) {
      throw std::runtime_error("empty argument");
    } else if (argp->at(0) == '-') {
      throw std::runtime_error(std::string("unrecognized option: ") + std::string(*argp));
    } else {
      break;
    }
  }
  return std::make_pair(opts, std::vector<std::string_view>(argp, args.end()));
}

void checked_read(std::ifstream& in, char* buffer, size_t size)
{
  in.read(buffer, size);
  if (in.fail()) {
    if (in.eof()) {
      throw std::runtime_error("Unexpected EOF");
    } else {
      throw std::runtime_error(std::strerror(errno));
    }
  }
}

size_t read_flatbuffer_size(std::ifstream& in)
{
  flatbuffers::uoffset_t fb_size;
  checked_read(in, reinterpret_cast<char*>(&fb_size), sizeof(fb_size));
  return flatbuffers::EndianScalar(fb_size);
}

std::unique_ptr<std::vector<char>> read_flatbuffer(std::ifstream& in)
{
  auto size = read_flatbuffer_size(in);
  auto buffer = std::make_unique<std::vector<char>>(size);
  checked_read(in, buffer->data(), size);
  return buffer;
}

std::ofstream open_output(std::filesystem::path const& path,
                          std::ios::openmode mode = std::ios::out)
{
  if (std::filesystem::exists(path)) {
    throw std::runtime_error(path.string() + " already exists");
  }
  std::ofstream out(path, mode);
  out.exceptions(std::ios::badbit);
  return out;
}

template<typename T>
T const* validate_fb(std::vector<char> const& fb, std::string_view const& name)
{
  flatbuffers::Verifier::Options verifier_opts;
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t const*>(fb.data()), fb.size(), verifier_opts);
  if (!verifier.VerifyBuffer<T>()) {
    throw std::runtime_error(std::string("malformed ") + std::string(name) + " record");
  }
  return flatbuffers::GetRoot<T>(fb.data());
}

void verify_profile_header(std::ifstream& in)
{
  auto fb_ptr = read_flatbuffer(in);
  auto header = validate_fb<spark_rapids_jni::profiler::ProfileHeader>(*fb_ptr, "profile header");
  auto magic = header->magic();
  if (magic == nullptr) {
    throw std::runtime_error("does not appear to be a spark-rapids profile");
  }
  if (magic->str() != "spark-rapids profile") {
    std::ostringstream oss;
    oss << "bad profile magic, expected 'spark-rapids profile' found '" << magic->str() << "'";
    throw std::runtime_error(oss.str());
  }
  auto version = header->version();
  if (version != 1) {
    std::ostringstream oss;
    oss << "unsupported profile version: " << version;
    throw std::runtime_error(oss.str());
  }
}

void convert_to_nsys_rep(std::ifstream& in, std::string_view const& in_filename,
                         program_options const& opts)
{
  verify_profile_header(in);

#if 0
  // TODO: get basename-only
  std::filesystem::path output_path(opts.output_path ? std::string(opts.output_path.value())
    : std::string(in_filename) + ".nsys-rep");
#endif

  while (!in.eof()) {
    auto fb_ptr = read_flatbuffer(in);
    auto records = validate_fb<spark_rapids_jni::profiler::ActivityRecords>(*fb_ptr, "ActivityRecords");
    std::cerr << "ACTIVITY RECORDS:" << std::endl;
    auto api = records->api();
    if (api != nullptr) {
      std::cerr << "NUM APIS=" << api->size() << std::endl;
    }
    auto device = records->device();
    if (device != nullptr) {
      std::cerr << "NUM DEVICES=" << device->size() << std::endl;
    }
    auto dropped = records->dropped();
    if (dropped != nullptr) {
      std::cerr << "NUM DROPPED=" << dropped->size() << std::endl;
    }
    auto kernel = records->kernel();
    if (kernel != nullptr) {
      std::cerr << "NUM KERNEL=" << kernel->size() << std::endl;
    }
    auto marker = records->marker();
    if (marker != nullptr) {
      std::cerr << "NUM MARKERS=" << marker->size() << std::endl;
    }
    auto marker_data = records->marker_data();
    if (marker_data != nullptr) {
      std::cerr << "NUM MARKER DATA=" << marker_data->size() << std::endl;
      for (int i = 0; i < marker_data->size(); ++i) {
        std::cerr << "MARKER DATA " << i << std::endl;
        auto md = marker_data->Get(i);
        std::cerr << " FLAGS: " << md->flags();
        std::cerr << " ID: " << md->id();
        std::cerr << " COLOR: " << md->color();
        std::cerr << " CATEGORY: " << md->category() << std::endl;
      }
    }
    auto memcpy = records->memcpy();
    if (memcpy != nullptr) {
      std::cerr << "NUM MEMCPY=" << memcpy->size() << std::endl;
    }
    auto memset = records->memset();
    if (device != nullptr) {
      std::cerr << "NUM MEMSET=" << memset->size() << std::endl;
    }
    auto overhead = records->overhead();
    if (overhead != nullptr) {
      std::cerr << "NUM OVERHEADS=" << overhead->size() << std::endl;
    }

    in.peek();
  }
  if (!in.eof()) {
    throw std::runtime_error(std::strerror(errno));
  }
}

void convert_to_json(std::ifstream& in, std::ostream& out, program_options const& opts)
{
  flatbuffers::Parser parser;
  if (parser.Parse(spark_rapids_jni::profiler::Profiler_Schema) != 0) {
    std::runtime_error("Internal error: Unable to parse profiler schema");
  }
  parser.opts.strict_json = true;
  while (!in.eof()) {
    auto fb_ptr = read_flatbuffer(in);
    auto records = validate_fb<spark_rapids_jni::profiler::ActivityRecords>(*fb_ptr, "ActivityRecords");
    std::string json;
    char const* err = flatbuffers::GenText(parser, fb_ptr->data(), &json);
    if (err != nullptr) {
      throw std::runtime_error(std::string("Error generating JSON: ") + err);
    }
    out << json;

    in.peek();
  }
  if (!in.eof()) {
    throw std::runtime_error(std::strerror(errno));
  }
}

int main(int argc, char* argv[])
{
  constexpr int RESULT_SUCCESS = 0;
  constexpr int RESULT_FAILURE = 1;
  constexpr int RESULT_USAGE = 2;
  program_options opts;
  std::vector<std::string_view> files;
  if (argc < 2) {
    print_usage();
    return RESULT_USAGE;
  }
  std::vector<std::string_view> args(argv + 1, argv + argc);
  try {
    auto [ options, inputs ] = parse_options(args);
    opts = options;
    files = inputs;
  } catch (std::exception const& e) {
    std::cerr << "spark_rapids_profile_converter: " << e.what() << std::endl;
    print_usage();
    return RESULT_USAGE;
  }
  if (opts.help) {
    print_usage();
    return RESULT_USAGE;
  }
  if (opts.version) {
    print_version();
    return RESULT_SUCCESS;
  }
  if (files.size() != 1) {
    std::cerr << "Missing input file." << std::endl;
    print_usage();
    return RESULT_USAGE;
  }
  auto input_file = files.front();
  try {
    std::ifstream in(std::string(input_file), std::ios::binary | std::ios::in);
    in.exceptions(std::istream::badbit);
    if (opts.json) {
      if (opts.output_path) {
        std::ofstream out = open_output(opts.output_path.value());
        convert_to_json(in, out, opts);
      } else {
        convert_to_json(in, std::cout, opts);
      }
    } else {
      convert_to_nsys_rep(in, input_file, opts);
    }
  } catch (std::system_error const& e) {
    std::cerr << "Error converting " << input_file << ": " << e.code().message() << std::endl;
    return RESULT_FAILURE;
  } catch (std::exception const& e) {
    std::cerr << "Error converting " << input_file << ": " << e.what() << std::endl;
    return RESULT_FAILURE;
  }
  return RESULT_SUCCESS;
}
