#include <gtest/gtest.h>

#include "seq/bessonov_e_multi_integration_trapezoid_method/include/ops_seq.hpp"

TEST(bessonov_e_multi_integration_trapezoid_method_seq, SingleDimensionalSinIntegration) {
  size_t dim = 1;
  std::vector<double> lower_limits = {0.0};
  std::vector<double> upper_limits = {1.0};
  std::vector<int> steps = {10};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);
  task.integrand = [](const std::vector<double>& point) { return std::sin(point[0]); };

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  double expected_result = 1.0 - std::cos(1.0);  // Integral of sin(x) from 0 to 1
  ASSERT_NEAR(result, expected_result, 1e-2);
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, OneDimensionalExpFunction) {
  size_t dim = 1;
  std::vector<double> lower_limits = {0.0};
  std::vector<double> upper_limits = {1.0};
  std::vector<int> steps = {100};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);
  task.integrand = [](const std::vector<double>& point) { return std::exp(point[0]); };

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  double expected_result = std::exp(1.0) - 1.0;  // Integral of exp(x) from 0 to 1
  ASSERT_NEAR(result, expected_result, 1e-2);
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, TwoDimensionalIntegration) {
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  std::vector<int> steps = {10, 10};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);
  task.integrand = [](const std::vector<double>& point) { return point[0] + point[1]; };

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  double expected_result = 1.0;
  ASSERT_NEAR(result, expected_result, 1e-2);
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, TwoDimensionalCosFunction) {
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {std::numbers::pi / 2, std::numbers::pi / 2};
  std::vector<int> steps = {50, 50};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);
  task.integrand = [](const std::vector<double>& point) { return std::cos(point[0]) * std::cos(point[1]); };

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  double expected_result = 1.0;  // Integral of cos(x)*cos(y) over [0, pi/2]x[0, pi/2]
  ASSERT_NEAR(result, expected_result, 1e-2);
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, ThreeDimensionalLinearFunction) {
  size_t dim = 3;
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  std::vector<int> steps = {20, 20, 20};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);
  task.integrand = [](const std::vector<double>& point) { return point[0] + point[1] + point[2]; };

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  double expected_result = 1.5;  // Integral of x+y+z over [0,1]x[0,1]x[0,1]
  ASSERT_NEAR(result, expected_result, 1e-2);
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, ThreeDimensionalIntegration) {
  size_t dim = 3;
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  std::vector<int> steps = {10, 10, 10};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);
  task.integrand = [](const std::vector<double>& point) { return point[0] * point[1] * point[2]; };

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  double expected_result = 0.125;  // Integral of x*y*z over [0,1]x[0,1]x[0,1]
  ASSERT_NEAR(result, expected_result, 1e-2);
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, FourDimensionalFunction) {
  size_t dim = 4;
  std::vector<double> lower_limits = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0, 1.0};
  std::vector<int> steps = {10, 10, 10, 10};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);
  task.integrand = [](const std::vector<double>& point) { return point[0] * point[1] + point[2] * point[3]; };

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  double expected_result = 0.5;  // Integral of x*y + z*w over [0,1]^4
  ASSERT_NEAR(result, expected_result, 1e-2);
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, ValidationTestInvalidDimension) {
  size_t dim = 0;  // Invalid dimension
  std::vector<double> lower_limits = {0.0};
  std::vector<double> upper_limits = {1.0};
  std::vector<int> steps = {10};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);

  ASSERT_FALSE(task.validation());
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, ValidationTestInvalidBounds) {
  size_t dim = 2;
  std::vector<double> lower_limits = {1.0, 0.0};
  std::vector<double> upper_limits = {0.0, 1.0};
  std::vector<int> steps = {10, 10};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);

  ASSERT_FALSE(task.validation());
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, ValidationTestValidData) {
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  std::vector<int> steps = {10, 10};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);

  ASSERT_TRUE(task.validation());
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, ValidationTestInvalidSteps) {
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  std::vector<int> steps = {10, -1};  // Invalid: negative steps
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);

  ASSERT_FALSE(task.validation());
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, ValidationTestInvalidCount) {
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0};// Invalid: count
  std::vector<int> steps = {10, 10};  
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential task(taskData);

  ASSERT_FALSE(task.validation());
}

