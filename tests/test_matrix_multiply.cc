
#include <chrono>
#include <iostream>
#include <mkl.h>
#include <mkl_cblas.h>
#include <sys/mman.h>

using namespace std::chrono;

int main() {

  int N = 10240;

  float *in1Data =
      (float *)mkl_malloc ( N * N * sizeof(float), 32 );
  float *in2Data =
      (float *)mkl_malloc (N * N * sizeof(float), 32 );
  float *outData =
      (float *)mkl_malloc (N * N * sizeof(float), 32);

  // make the random stream
  VSLStreamStatePtr stream;
  vslNewStream(&stream, VSL_BRNG_MCG31, 123);

  // the left and right boundary
  auto left = 0.0f;
  auto right = 1.0f;

  // create a bunch of random numbers
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int32_t)(N * N), in1Data,
               left, right);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int32_t)(N * N), in2Data,
               left, right);

  // delete the stream
  vslDeleteStream(&stream);

//   for(int idx = 0; idx < N * N; ++idx) {
//       in1Data[idx] = 1.0f;
//       in2Data[idx] = 1.0f;
//   }

  auto start = high_resolution_clock::now();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, in1Data,
              N, in2Data, N, 0.0f, outData, N);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  std::cout << 1e-6f * duration.count() << "s" << std::endl;
}