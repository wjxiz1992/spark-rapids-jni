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

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

struct program_options {
  std::optional<std::string_view> output_path;
  bool help = false;
};

void print_usage()
{
  std::cout << "spark_rapids_profile_converter [OPTION]... profilebin" << std::endl;
  std::cout << R"(
Converts the spark-rapids profile in profile.bin into other forms.

  -h, --help           show this usage message
  -o, --output=PATH    use PATH as the output filename)"
    << std::endl;
}

std::pair<program_options, std::vector<std::string_view>>
parse_options(std::vector<std::string_view> args)
{
  program_options opts{};
  std::string_view long_output("--output=");
  bool seen_output = false;
  auto argp = args.begin();
  while (argp != args.end()) {
    if (*argp == "-o") {
      if (seen_output) {
        throw std::runtime_error("output path cannot be specified twice");
      }
      seen_output = true;
      if (++argp != args.end()) {
        opts.output_path = std::make_optional(*argp++);
      } else {
        throw std::runtime_error("missing argument for output path");
      }
    } else if (argp->find_first_of(long_output) == 0) {
      argp->remove_prefix(long_output.size());
      if (argp->empty()) {
        throw std::runtime_error("missing argument for output path");
      } else {
        opts.output_path = std::make_optional(*argp++);
      }
    } else if (*argp == "-h" || *argp == "--help") {
      opts.help = true;
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

size_t read_flatbuffer_size(std::ifstream& in)
{
  flatbuffers::uoffset_t fb_size;
  in.read(reinterpret_cast<char*>(&fb_size), sizeof(fb_size));
  return flatbuffers::EndianScalar(fb_size);
}

std::unique_ptr<char[]> read_flatbuffer(std::ifstream& in)
{
  auto size = read_flatbuffer_size(in);
  std::unique_ptr<char[]> buffer(new char[size]);
  in.read(buffer.get(), size);
  return buffer;
}

void verify_profile_header(std::ifstream& in)
{
  auto fb_ptr = read_flatbuffer(in);
  auto header = flatbuffers::GetRoot<spark_rapids_jni::profiler::ProfileHeader>(fb_ptr.get());
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

void convert_to_nsys_rep(std::string_view const& in_file, std::string const& out_file)
{
  std::ifstream in(std::string(in_file), std::ios_base::binary | std::ios_base::in);
  verify_profile_header(in);

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
  if (opts.help || files.size() != 1) {
    print_usage();
    return RESULT_USAGE;
  }
  auto input_file = files.front();
  std::string output_path(opts.output_path ? std::string(opts.output_path.value())
    : std::string(input_file) + ".nsys-rep");
  try {
    convert_to_nsys_rep(input_file, output_path);
  } catch (std::exception const& e) {
    std::cerr << "Error converting " << input_file << ": " << e.what() << std::endl;
    return RESULT_FAILURE;
  }
  return RESULT_SUCCESS;
}
