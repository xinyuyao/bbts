#include "ud_function.h"

void bbts::ud_impl_t::call_ud(const bbts::ud_impl_t::tensor_params_t &_params,
                              const tensor_args_t &_in,
                              tensor_args_t &_out) const {

  

    // check if this is a gpu based function
    if (is_gpu) {

        if constexpr (static_config::enable_gpu) {

            // call the function
            fn(_params, _in, _out);

            // sync the device
            auto error = cudaDeviceSynchronize();
            checkCudaErrors(error);
        }
        else {
            
            // throw an exception this is bad...
            throw std::runtime_error("Calling GPU function but the system was not" 
                                     " compiled with support for GPU.");
        }
        
    } else {

        // jut call the function
        fn(_params, _in, _out);
    }
}