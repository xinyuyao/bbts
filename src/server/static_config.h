#pragma once


namespace bbts {

// the configuration set during compilation
struct static_config {

// are hooks enabled
#ifdef ENABLE_HOOKS
  static const bool enable_hooks = true;
#else
  static const bool enable_hooks = false;
#endif

// are we using gpu
#ifdef ENABLE_GPU
  static const bool enable_gpu = true;
#else 
  static const bool enable_gpu = true;
#endif

// are we using a buffer manager that relies on storage
#ifdef ENABLE_STORAGE
  static const bool enable_storage = true;
#else
  static const bool enable_storage = true;
#endif

};


}
