#include <gtest/gtest.h>
#include <thread>
#include "../src/storage/storage.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/tensor/builtin_formats.h"

namespace bbts {

TEST(TestStorage, Test1) {

  // how many threads are going to be hammering the storage
  const int32_t num_threads = 5;
  const int32_t num_matrices = 10;

  // the thraeds that are going to hammer it
  std::vector<std::thread> threads;

  // create the storage
  storage_ptr_t storage = std::make_shared<storage_t>(nullptr);

  // make a tensor factory
  tensor_factory_ptr_t tf = std::make_shared<tensor_factory_t>();

  // grab the format impl_id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  // make the threads
  threads.reserve(num_threads);
  for(int t = 0; t < num_threads; ++t) {

    // run a bunch of threads
    threads.emplace_back([&storage, &tf, &num_matrices, fmt_id, t]() {

      for(uint32_t i = 0; i < num_matrices; i++) {
        
        // make the meta
        dense_tensor_meta_t dm{fmt_id, i * 100, i * 200};

        // get the size of the tensor we need to crate
        auto tensor_size = tf->get_tensor_size(dm);

        storage->local_transaction({}, {{i + t * num_matrices, false, tensor_size}}, [&](const storage_t::reservation_result_t &res) {

          // crate the tensor
          auto ts = res.create[0].tensor;

          // init the tensor
          auto &dt = tf->init_tensor(ts, dm).as<dense_tensor_t>();

          // write some memory into it
          for(int j = 0; j < (i * 100) * (i * 200); j++) {
            dt.data()[j] = (float) j;
          }
        });
      }

      // get the tensors and check them
      for(size_t i = 0; i < num_matrices; i++) {

        storage->local_transaction({ (tid_t) (i + t * num_matrices) }, {}, [&](const storage_t::reservation_result_t &res) {
          
          // get the dense tensor
          auto &dt = res.get[0].tensor->as<dense_tensor_t>();;

          // write some memory into it
          for(int j = 0; j < (i * 100) * (i * 200); j++) {
            EXPECT_LE(std::abs(dt.data()[j]  - (float) j), 0.0001f);
          }

        });

        // remove the tensor
        storage->remove_by_tid((tid_t) (i + t * num_matrices));
      }
    });
  }

  // wait for all the threads to finish
  for(auto &t : threads) {
    t.join();
  }
}

}