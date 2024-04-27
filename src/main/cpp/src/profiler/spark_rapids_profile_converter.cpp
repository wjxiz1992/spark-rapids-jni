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

#include <iostream>
#include <optional>
#include <string>
#include <vector>

constexpr int MAIN_RESULT_SUCCESS = 0;
constexpr int MAIN_RESULT_FAILURE = 1;
constexpr int MAIN_RESULT_USAGE = 2;

struct program_options {
  std::optional<std::string_view> output_path;
  bool help = false;
};

void print_usage(std::string_view prog_name)
{
  std::cout << prog_name << " [OPTION]... profilebin" << std::endl;
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

int main(int argc, char* argv[])
{
  std::string_view prog_name(argv[0]);
  program_options opts;
  std::vector<std::string_view> files;
  if (argc < 2) {
    print_usage(prog_name);
    return MAIN_RESULT_USAGE;
  }
  std::vector<std::string_view> args(argv + 1, argv + argc);
  try {
    auto [ options, inputs ] = parse_options(args);
    opts = options;
    files = inputs;
  } catch (std::exception const &e) {
    std::cerr << prog_name << ": " << e.what() << std::endl;
    print_usage(prog_name);
    return MAIN_RESULT_USAGE;
  }
  if (opts.help || files.size() != 1) {
    print_usage(prog_name);
    return MAIN_RESULT_USAGE;
  }
  std::string output_path(opts.output_path ? std::string(opts.output_path.value())
    : std::string(files.front()) + ".nsys-rep");
  std::cout << "Processing " << files.front() << " into " << output_path << std::endl;
  return MAIN_RESULT_SUCCESS;
}
