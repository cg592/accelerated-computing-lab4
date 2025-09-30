#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <cassert>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem)

namespace matmul_l1 {

__global__ void matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c,
    int TILE_DIM_I,
    int TILE_DIM_J,
    int TILE_DIM_K) {
    
    int BLOCK_DIM_J = blockDim.x;
    int BLOCK_DIM_I = blockDim.y;
    int BLOCK_START_J = blockIdx.x * BLOCK_DIM_J;
    int BLOCK_START_I = blockIdx.y * BLOCK_DIM_I;
    int THREADS_PER_WARP = 32;
    int THREAD_OFFSET_J = threadIdx.x;
    int THREAD_OFFSET_I = threadIdx.y;

    extern __shared__ float shmem[];
    float* shared_A = shmem;
    float* shared_B = shared_A + TILE_DIM_I * TILE_DIM_K;
    float* shared_C = shared_B + TILE_DIM_K * TILE_DIM_J;

    float sum = 0.0;

    for (int BLOCK_START_K = 0; BLOCK_START_K < size_k; BLOCK_START_K += TILE_DIM_K) {
        // LOAD 
        assert(TILE_DIM_I == THREADS_PER_WARP && TILE_DIM_J == THREADS_PER_WARP && TILE_DIM_K == THREADS_PER_WARP);
        int thread_load_a_start_i = BLOCK_START_I + THREAD_OFFSET_I;
        int thread_load_a_start_k = BLOCK_START_K + threadIdx.x;
        shared_A[THREAD_OFFSET_I * TILE_DIM_K + threadIdx.x] = a[thread_load_a_start_i * size_k + thread_load_a_start_k];

        int thread_load_b_start_k = BLOCK_START_K + threadIdx.y;
        int thread_load_b_start_j = BLOCK_START_J + THREAD_OFFSET_J;
        shared_B[threadIdx.y * TILE_DIM_J + THREAD_OFFSET_J] = b[thread_load_b_start_k * size_j + thread_load_b_start_j];

        int thread_load_c_start_i = BLOCK_START_I + THREAD_OFFSET_I;
        int thread_load_c_start_j = BLOCK_START_J + THREAD_OFFSET_J;
        shared_C[THREAD_OFFSET_I * TILE_DIM_J + THREAD_OFFSET_J] = c[thread_load_c_start_i * size_j + thread_load_c_start_j];

        __syncthreads();

        // COMPUTE 
        for (int k = 0; k < TILE_DIM_K; k += 1) {
            float a_val = shared_A[THREAD_OFFSET_I * TILE_DIM_K + k];
            float b_val = shared_B[k * TILE_DIM_J + THREAD_OFFSET_J];
            sum += a_val * b_val;            

            int compute_i = BLOCK_START_I + THREAD_OFFSET_I;
            int compute_j = BLOCK_START_J + THREAD_OFFSET_J;
            int compute_k = BLOCK_START_K + k;
        }
        __syncthreads();
    }

    int thread_store_i = BLOCK_START_I + THREAD_OFFSET_I;
    int thread_store_j = BLOCK_START_J + THREAD_OFFSET_J;
    c[thread_store_i * size_j + thread_store_j] = sum;
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // X = J, Y = I
    
    int TILE_DIM_I = 32;
    int TILE_DIM_J = 32;
    int TILE_DIM_K = 32;
    int NUM_TILES_I = (size_i + TILE_DIM_I - 1) / TILE_DIM_I;
    int NUM_TILES_J = (size_j + TILE_DIM_J - 1) / TILE_DIM_J;

    dim3 num_blocks(NUM_TILES_J, NUM_TILES_I);
    dim3 block_size(TILE_DIM_J, TILE_DIM_I);

    int shmem_size_bytes = (TILE_DIM_I * TILE_DIM_K + TILE_DIM_K * TILE_DIM_J + TILE_DIM_I * TILE_DIM_J) * sizeof(float);

    // int shmem_size_bytes = (TILE_DIM_I * TILE_DIM_J) * sizeof(float);
    // int shmem_max_elem = (99 * 1024) / sizeof(float);
    // int shmem_elem_remaining = shmem_max_elem - (TILE_DIM_I * TILE_DIM_J);
    // int k_groups_remaining = shmem_elem_remaining / (32 * TILE_DIM_I + 32 * TILE_DIM_J); // 32 threads per warp
    // shmem_size_bytes += k_groups_remaining * 32 * (TILE_DIM_I + TILE_DIM_J) * sizeof(float);

    // std::cout << "shmem_max_elem: " << shmem_max_elem << std::endl;
    // std::cout << "shmem_elem_remaining: " << shmem_elem_remaining << std::endl;
    // std::cout << "k_groups_remaining: " << k_groups_remaining << std::endl;

    // std::cout << "shmem_size_bytes: " << shmem_size_bytes << std::endl;
    // std::cout << "percent of shmem used: " << (double)shmem_size_bytes / (100 * 1024) * 100 << "%" << std::endl;
    assert(shmem_size_bytes <= 100 * 1024);
    CUDA_CHECK(cudaFuncSetAttribute(matmul_l1,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shmem_size_bytes));

    // std::cout << "num_blocks: " << num_blocks.x << " " << num_blocks.y << std::endl;
    // std::cout << "block_size: " << block_size.x << " " << block_size.y << std::endl;

    matmul_l1<<<num_blocks, block_size, shmem_size_bytes>>>(size_i, size_j, size_k, a, b, c, TILE_DIM_I, TILE_DIM_J, TILE_DIM_K);

}

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

namespace matmul_l1_reg {

__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your GPU code here */
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */
}

}; // namespace matmul_l1_reg

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_k * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_gpu,
            b.data(),
            size_k * size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

        std::vector<float> c_out_host(size_i * size_j);
        CUDA_CHECK(cudaMemcpy(
            c_out_host.data(),
            c_gpu,
            size_i * size_j * sizeof(float),
            cudaMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t i = 0; i < size_i; ++i) {
            for (int32_t j = 0; j < size_j; ++j) {
                float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
                mse += diff * diff;
                ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
            }
        }
        mse /= size_i * size_j;
        ref_mean_square /= size_i * size_j;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulL1 {
    constexpr static char const *name = "matmul_l1";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1::launch_matmul_l1(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulL1>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}
