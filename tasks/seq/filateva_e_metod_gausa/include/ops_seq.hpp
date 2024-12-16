// Filateva Elizaveta Metod Gausa

#include <vector>

#include "core/task/include/task.hpp"

namespace filateva_e_metod_gausa_seq {

class MetodGausa : public ppc::core::Task {
 public:
  explicit MetodGausa(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  struct Matrix {
    double* pLine;
    double b;
  };

  int size;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> resh;
};

}  // namespace filateva_e_metod_gausa_seq