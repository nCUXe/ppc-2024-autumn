#include "seq/bessonov_e_multi_integration_trapezoid_method/include/ops_seq.hpp"

bool bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::validation() {
  internal_order_test();

  // Check input size
  if (taskData->inputs.size() < 3) {
    return false;
  }

  // Check output size
  if (taskData->outputs.size() != 1) {
    return false;
  }

  // Check dimension
  size_t* dim_ptr = reinterpret_cast<size_t*>(taskData->inputs[0]);
  if (!dim_ptr || *dim_ptr == 0) {
    return false;
  }

  // Check lower and upper bounds
  double* lower_data = reinterpret_cast<double*>(taskData->inputs[1]);
  double* upper_data = reinterpret_cast<double*>(taskData->inputs[2]);
  int* steps_data = reinterpret_cast<int*>(taskData->inputs[3]);

  if (!lower_data || !upper_data || !steps_data) {
    return false;
  }

  // Check that sizes of lower_bounds, upper_bounds, and steps match dimension
  if (taskData->inputs_count.size() < 3 || taskData->inputs_count[0] != *dim_ptr ||
      taskData->inputs_count[1] != *dim_ptr || taskData->inputs_count[2] != *dim_ptr) {
    return false;
  }

  for (size_t i = 0; i < *dim_ptr; ++i) {
    if (lower_data[i] >= upper_data[i]) {
      return false;
    }
    if (steps_data[i] <= 0) {
      return false;
    }
  }

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  dim = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  lower_bounds.assign(reinterpret_cast<double*>(taskData->inputs[1]),
                      reinterpret_cast<double*>(taskData->inputs[1]) + dim);
  upper_bounds.assign(reinterpret_cast<double*>(taskData->inputs[2]),
                      reinterpret_cast<double*>(taskData->inputs[2]) + dim);
  num_steps.assign(reinterpret_cast<int*>(taskData->inputs[3]), reinterpret_cast<int*>(taskData->inputs[3]) + dim);

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::run() {
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
        weight *= 0.5;  // ”чет граничных точек
      }
    }

    double f_value = integrand(point);
    result += f_value * weight;
  }

  // ”чет объема €чеек
  result *= std::accumulate(step_sizes.begin(), step_sizes.end(), 1.0, std::multiplies<>());

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}