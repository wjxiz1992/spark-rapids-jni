/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/device_uvector.hpp>

#include <cast_string.hpp>

#include <cstdint>
#include <cstring>
#include <limits>

using namespace cudf;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

struct FloatToStringTests : public cudf::test::BaseFixture {};

namespace {

template <typename FloatingPoint, typename Bits>
FloatingPoint from_bits(Bits const bits)
{
  static_assert(sizeof(FloatingPoint) == sizeof(Bits));
  FloatingPoint value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

TEST_F(FloatToStringTests, FromFloats32)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<float>{100.0f,
                                                  654321.25f,
                                                  -12761.125f,
                                                  0.f,
                                                  5.0f,
                                                  -4.0f,
                                                  std::numeric_limits<float>::quiet_NaN(),
                                                  123456789012.34f,
                                                  -0.0f};

  auto results = spark_rapids_jni::float_to_string(floats, cudf::get_default_stream());

  auto const expected = cudf::test::strings_column_wrapper{
    "100.0", "654321.25", "-12761.125", "0.0", "5.0", "-4.0", "NaN", "1.2345679E11", "-0.0"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, FromFloats64)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<double>{100.0d,
                                                   654321.25d,
                                                   -12761.125d,
                                                   1.123456789123456789d,
                                                   0.000000000000000000123456789123456789d,
                                                   0.0d,
                                                   5.0d,
                                                   -4.0d,
                                                   std::numeric_limits<double>::quiet_NaN(),
                                                   839542223232.794248339d,
                                                   -0.0d};

  auto results = spark_rapids_jni::float_to_string(floats, cudf::get_default_stream());

  auto const expected = cudf::test::strings_column_wrapper{"100.0",
                                                           "654321.25",
                                                           "-12761.125",
                                                           "1.1234567891234568",
                                                           "1.234567891234568E-19",
                                                           "0.0",
                                                           "5.0",
                                                           "-4.0",
                                                           "NaN",
                                                           "8.395422232327942E11",
                                                           "-0.0"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, FromFloats32ToJsonString)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<float>{100.0f,
                                                  -4.0f,
                                                  std::numeric_limits<float>::quiet_NaN(),
                                                  std::numeric_limits<float>::infinity(),
                                                  -std::numeric_limits<float>::infinity(),
                                                  -0.0f};

  auto const results = spark_rapids_jni::float_to_string(floats,
                                                         /*json_string=*/true,
                                                         cudf::get_default_stream(),
                                                         cudf::get_current_device_resource_ref());

  auto const expected = cudf::test::strings_column_wrapper{
    "100.0", "-4.0", "\"NaN\"", "\"Infinity\"", "\"-Infinity\"", "-0.0"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, FromFloats64ToJsonString)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<double>{100.0d,
                                                   -4.0d,
                                                   std::numeric_limits<double>::quiet_NaN(),
                                                   std::numeric_limits<double>::infinity(),
                                                   -std::numeric_limits<double>::infinity(),
                                                   -0.0d};

  auto const results = spark_rapids_jni::float_to_string(floats,
                                                         /*json_string=*/true,
                                                         cudf::get_default_stream(),
                                                         cudf::get_current_device_resource_ref());

  auto const expected = cudf::test::strings_column_wrapper{
    "100.0", "-4.0", "\"NaN\"", "\"Infinity\"", "\"-Infinity\"", "-0.0"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, Float32MatchesLegacyJavaJsonFormatting)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<float>{from_bits<float>(uint32_t{0x5e1b'372a}),
                                                  from_bits<float>(uint32_t{0x4ca5'a5a1}),
                                                  from_bits<float>(uint32_t{0xd487'ce1e}),
                                                  from_bits<float>(uint32_t{0x6a26'4afb}),
                                                  from_bits<float>(uint32_t{0xe8fd'd6aa}),
                                                  from_bits<float>(uint32_t{0x68fd'f7d7}),
                                                  from_bits<float>(uint32_t{0x7f7f'ffff})};

  auto const results = spark_rapids_jni::float_to_string(floats,
                                                         /*json_string=*/true,
                                                         cudf::get_default_stream(),
                                                         cudf::get_current_device_resource_ref());

  auto const expected = cudf::test::strings_column_wrapper{"2.79611359E18",
                                                           "8.6846728E7",
                                                           "-4.6662293E12",
                                                           "5.0258942E25",
                                                           "-9.5897485E24",
                                                           "9.5946444E24",
                                                           "3.4028235E38"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, Float64MatchesLegacyJavaJsonFormatting)
{
  auto const floats = cudf::test::fixed_width_column_wrapper<double>{
    from_bits<double>(uint64_t{0xc388'938a'82bc'a728}),
    from_bits<double>(uint64_t{0xc39d'dd74'67af'36d9}),
    from_bits<double>(uint64_t{0x4535'233e'b2d0'bfb4}),
    from_bits<double>(uint64_t{0x0f65'cf61'8f97'2b7e}),
    from_bits<double>(uint64_t{0x43d0'0000'0000'0000}),
    from_bits<double>(uint64_t{0x43e0'0000'0000'0000}),
    from_bits<double>(uint64_t{0x7fef'ffff'ffff'ffff})};

  auto const results = spark_rapids_jni::float_to_string(floats,
                                                         /*json_string=*/true,
                                                         cudf::get_default_stream(),
                                                         cudf::get_current_device_resource_ref());

  auto const expected = cudf::test::strings_column_wrapper{"-2.21363921575273728E17",
                                                           "-5.3800104640575648E17",
                                                           "2.5553881621949533E25",
                                                           "1.714867986910578E-234",
                                                           "4.6116860184273879E18",
                                                           "9.223372036854776E18",
                                                           "1.7976931348623157E308"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, LegacyJavaFormattingIsJsonOnly)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<float>{from_bits<float>(uint32_t{0x5e1b'372a})};

  auto const results  = spark_rapids_jni::float_to_string(floats, cudf::get_default_stream());
  auto const expected = cudf::test::strings_column_wrapper{"2.7961136E18"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, LegacyJavaDoubleFormattingIsJsonOnly)
{
  auto const doubles = cudf::test::fixed_width_column_wrapper<double>{
    from_bits<double>(uint64_t{0xc39d'dd74'67af'36d9})};

  auto const results  = spark_rapids_jni::float_to_string(doubles, cudf::get_default_stream());
  auto const expected = cudf::test::strings_column_wrapper{"-5.380010464057565E17"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}
