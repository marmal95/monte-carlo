#include "Algorithm.hpp"
#include <numeric>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


const std::size_t POINTS = 10'000'000'000;
const std::size_t THREADS_PER_BLOCK = 32;
const std::size_t NUM_BLOCKS = 640;
const std::size_t THREAD_ITERATIONS = POINTS / THREADS_PER_BLOCK / NUM_BLOCKS;


__global__ void monteCarlo_cuda(std::size_t* totals)
{
    __shared__ std::size_t counter[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    counter[threadIdx.x] = 0;

    curandState_t rng;
    curand_init(clock64(), tid, 0, &rng);

    for (int i = 0; i < THREAD_ITERATIONS; i++)
    {
        float x = curand_uniform(&rng);
        float y = curand_uniform(&rng);
        counter[threadIdx.x] += 1 - int(x * x + y * y);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        totals[blockIdx.x] = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            totals[blockIdx.x] += counter[i];
        }
    }
}

MonteCarloResult monteCarlo()
{
    thrust::host_vector<std::size_t> blocksCount(NUM_BLOCKS);
    thrust::device_vector<std::size_t> blocksCount_dev(NUM_BLOCKS);

    monteCarlo_cuda<<<NUM_BLOCKS, THREADS_PER_BLOCK >>>(blocksCount_dev.data().get());
    thrust::copy(blocksCount_dev.begin(), blocksCount_dev.end(), blocksCount.begin());

    const auto pointsInCircle = std::accumulate(blocksCount.cbegin(), blocksCount.cend(), std::size_t{});
    return { pointsInCircle, THREADS_PER_BLOCK * NUM_BLOCKS * THREAD_ITERATIONS };
}