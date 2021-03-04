#pragma once
#include "memory_storage.h"
#include "nvme_storage.h"
#include <memory>

namespace bbts {

using storage_t = nvme_storage_t;
using storage_ptr_t = std::shared_ptr<storage_t>;

}