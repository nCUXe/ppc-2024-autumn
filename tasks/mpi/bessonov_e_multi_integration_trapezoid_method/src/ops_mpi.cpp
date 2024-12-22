#include "mpi/bessonov_e_multi_integration_trapezoid_method/include/ops_mpi.hpp"

bool bessonov_e_multi_integration_trapezoid_method_mpi::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs.size() < 3 || taskData->outputs.size() != 1) {
    return false;
  }

  auto* dim_ptr = reinterpret_cast<size_t*>(taskData->inputs[0]);
  if (!dim_ptr || *dim_ptr == 0) {
    return false;
  }

  auto* lower_data = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* upper_data = reinterpret_cast<double*>(taskData->inputs[2]);
  auto* steps_data = reinterpret_cast<int*>(taskData->inputs[3]);

  if (!lower_data || !upper_data || !steps_data) {
    return false;
  }

  for (size_t i = 0; i < *dim_ptr; ++i) {
    if (lower_data[i] >= upper_data[i] || steps_data[i] <= 0) {
      return false;
    }
  }

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();

  dim = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  lower_bounds.assign(reinterpret_cast<double*>(taskData->inputs[1]),
                      reinterpret_cast<double*>(taskData->inputs[1]) + dim);
  upper_bounds.assign(reinterpret_cast<double*>(taskData->inputs[2]),
                      reinterpret_cast<double*>(taskData->inputs[2]) + dim);
  num_steps.assign(reinterpret_cast<int*>(taskData->inputs[3]), reinterpret_cast<int*>(taskData->inputs[3]) + dim);

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_mpi::TestTaskSequential::run() {
  internal_order_test();

  std::vector<double> step_sizes(dim);
  for (size_t i = 0; i < dim; ++i) {
    step_sizes[i] = (upper_bounds[i] - lower_bounds[i]) / num_steps[i];
  }

  size_t total_points = 1;
  for (int steps : num_steps) {
    total_points *= (steps + 1);
  }

  result = 0.0;
  std::vector<double> point(dim);

  for (size_t i = 0; i < total_points; ++i) {
    size_t temp = i;
    double weight = 1.0;

    for (size_t j = 0; j < dim; ++j) {
      point[j] = lower_bounds[j] + (temp % (num_steps[j] + 1)) * step_sizes[j];
      temp /= (num_steps[j] + 1);

      if (point[j] == lower_bounds[j] || point[j] == upper_bounds[j]) {
        weight *= 0.5;
      }
    }

    result += integrand(point) * weight;
  }

  result *= std::accumulate(step_sizes.begin(), step_sizes.end(), 1.0, std::multiplies<>());

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_mpi::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_mpi::TestTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.size() < 3 || taskData->outputs.size() != 1) {
      return false;
    }

    auto* dim_ptr = reinterpret_cast<size_t*>(taskData->inputs[0]);
    if (dim_ptr == nullptr || *dim_ptr == 0) {
      return false;
    }

    auto* lower_data = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* upper_data = reinterpret_cast<double*>(taskData->inputs[2]);
    auto* steps_data = reinterpret_cast<int*>(taskData->inputs[3]);

    if (lower_data == nullptr || upper_data == nullptr || steps_data == nullptr) {
      return false;
    }

    for (size_t i = 0; i < *dim_ptr; ++i) {
      if (lower_data[i] >= upper_data[i] || steps_data[i] <= 0) {
        return false;
      }
    }
  }
  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_mpi::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    dim = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    lower_bounds.assign(reinterpret_cast<double*>(taskData->inputs[1]),
                        reinterpret_cast<double*>(taskData->inputs[1]) + dim);
    upper_bounds.assign(reinterpret_cast<double*>(taskData->inputs[2]),
                        reinterpret_cast<double*>(taskData->inputs[2]) + dim);
    num_steps.assign(reinterpret_cast<int*>(taskData->inputs[3]), reinterpret_cast<int*>(taskData->inputs[3]) + dim);

    return true;
  }

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_mpi::TestTaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, dim, 0);

  lower_bounds.resize(dim);
  upper_bounds.resize(dim);
  num_steps.resize(dim);

  boost::mpi::broadcast(world, lower_bounds.data(), dim, 0);
  boost::mpi::broadcast(world, upper_bounds.data(), dim, 0);
  boost::mpi::broadcast(world, num_steps.data(), dim, 0);

  std::vector<double> step_sizes(dim);
  for (size_t i = 0; i < dim; ++i) {
    step_sizes[i] = (upper_bounds[i] - lower_bounds[i]) / num_steps[i];
  }

  size_t total_points = 1;
  for (int steps : num_steps) {
    total_points *= (steps + 1);
  }

  size_t points_per_proc = total_points / world.size();
  size_t start = world.rank() * points_per_proc;
  size_t end = (world.rank() == world.size() - 1) ? total_points : start + points_per_proc;

  double local_result = 0.0;
  std::vector<double> point(dim);

  for (size_t i = start; i < end; ++i) {
    size_t temp = i;
    double weight = 1.0;

    for (size_t j = 0; j < dim; ++j) {
      point[j] = lower_bounds[j] + (temp % (num_steps[j] + 1)) * step_sizes[j];
      temp /= (num_steps[j] + 1);

      if (point[j] == lower_bounds[j] || point[j] == upper_bounds[j]) {
        weight *= 0.5;
      }
    }

    local_result += integrand(point) * weight;
  }

  boost::mpi::reduce(world, local_result, result, std::plus<>(), 0);

  if (world.rank() == 0) {
    result *= std::accumulate(step_sizes.begin(), step_sizes.end(), 1.0, std::multiplies<>());
  }

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_mpi::TestTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  }
  return true;
}