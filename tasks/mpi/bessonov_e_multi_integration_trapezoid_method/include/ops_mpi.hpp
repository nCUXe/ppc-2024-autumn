#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <memory>
#include <numbers>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_multi_integration_trapezoid_method_mpi {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  std::function<double(const std::vector<double>&)> integrand;

 private:
  size_t dim;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  std::vector<int> num_steps;

  double result;
};

class TestTaskParallel : public ppc::core::Task {
 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  std::function<double(const std::vector<double>&)> integrand;

 private:
  size_t dim;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  std::vector<int> num_steps;

  double result;
  boost::mpi::communicator world;
};

}  // namespace bessonov_e_multi_integration_trapezoid_method_mpi