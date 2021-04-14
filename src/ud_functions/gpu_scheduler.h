#include "../server/static_config.h"
#include "gpu_scheduler_impl.h"
#include "null_gpu_scheduler.h"

#include <cstddef>
#include <memory>

namespace bbts {

using gpu_scheduler_t = std::conditional<static_config::enable_gpu, gpu_scheduler_impl_t, null_gpu_scheduler_t>::type;
using gpu_scheduler_ptr_t = std::shared_ptr<gpu_scheduler_t>;

} // namespace bbts