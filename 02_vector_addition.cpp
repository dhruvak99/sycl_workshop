#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>

using namespace sycl;
using namespace std;
using namespace chrono;

constexpr int OPERATIONS = 100;
constexpr int ITERATIONS = 5;

// Identical computation for both implementations
inline float compute(float value) {

    for (int j = 0; j < OPERATIONS; j++) {
        value = value * 1.00001f + 0.00001f;
    }

    return value;
}

// Calculate median
double median(vector<double> values) {

    sort(values.begin(), values.end());

    return values[values.size() / 2];
}

// Calculate checksum outside benchmark
double calculate_checksum(const vector<float>& data) {

    double checksum = 0.0;

    for (float value : data) {
        checksum += value;
    }

    return checksum;
}

int main() {

    const size_t N = 10000000;

    cout << "Initializing " << N << " elements...\n";

    vector<float> A(N);
    vector<float> B(N);

    vector<float> C_serial(N);
    vector<float> C_sycl(N);

    // Initialize inputs
    for (size_t i = 0; i < N; i++) {

        A[i] = 1.0f + (i % 100) * 0.01f;
        B[i] = 2.0f + (i % 50) * 0.02f;
    }

    // Create SYCL queue with profiling
    queue q{
        default_selector_v,
        property::queue::enable_profiling{}
    };

    // Create buffers outside benchmark
    buffer<float> bufA(A.data(), range<1>(N));
    buffer<float> bufB(B.data(), range<1>(N));
    buffer<float> bufC(C_sycl.data(), range<1>(N));

    vector<double> serial_times;
    vector<double> sycl_kernel_times;
    vector<double> sycl_total_times;

    // Warm-up execution
    event warmup = q.submit([&](handler& h) {

        auto a = bufA.get_access<access::mode::read>(h);
        auto b = bufB.get_access<access::mode::read>(h);
        auto c = bufC.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(N), [=](id<1> i) {

            c[i] = compute(a[i] + b[i]);

        });
    });

    warmup.wait();

    // Benchmark iterations
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {

        // --------------------------------
        // Serial execution
        // --------------------------------

        auto serial_start = high_resolution_clock::now();

        for (size_t i = 0; i < N; i++) {

            C_serial[i] = compute(A[i] + B[i]);

        }

        auto serial_end = high_resolution_clock::now();

        double serial_time =
            duration<double, milli>(
                serial_end - serial_start
            ).count();

        serial_times.push_back(serial_time);

        // --------------------------------
        // SYCL execution
        // --------------------------------

        auto sycl_start = high_resolution_clock::now();

        event e = q.submit([&](handler& h) {

            auto a = bufA.get_access<access::mode::read>(h);
            auto b = bufB.get_access<access::mode::read>(h);
            auto c = bufC.get_access<access::mode::write>(h);

            h.parallel_for(range<1>(N), [=](id<1> i) {

                c[i] = compute(a[i] + b[i]);

            });
        });

        e.wait();

        auto sycl_end = high_resolution_clock::now();

        double sycl_total_time =
            duration<double, milli>(
                sycl_end - sycl_start
            ).count();

        sycl_total_times.push_back(sycl_total_time);

        // --------------------------------
        // Kernel profiling
        // --------------------------------

        uint64_t kernel_start =
            e.get_profiling_info<
                info::event_profiling::command_start>();

        uint64_t kernel_end =
            e.get_profiling_info<
                info::event_profiling::command_end>();

        double kernel_time =
            (kernel_end - kernel_start) / 1e6;

        sycl_kernel_times.push_back(kernel_time);

        cout << "Completed iteration "
             << iteration + 1 << "/" << ITERATIONS << "\n";
    }

    // --------------------------------
    // Validation outside timing
    // --------------------------------

    bool correct = true;

    for (size_t i = 0; i < N; i++) {

        if (std::abs(C_serial[i] - C_sycl[i]) > 0.001f) {

            correct = false;
            break;
        }
    }

    double serial_checksum = calculate_checksum(C_serial);
    double sycl_checksum = calculate_checksum(C_sycl);

    // --------------------------------
    // Final results
    // --------------------------------

    double serial_median = median(serial_times);
    double kernel_median = median(sycl_kernel_times);
    double total_median = median(sycl_total_times);

    cout << "\n===== COMPUTE-INTENSIVE BENCHMARK =====\n";

    cout << "Elements: " << N << "\n";
    cout << "Operations per element: " << OPERATIONS << "\n";
    cout << "Iterations: " << ITERATIONS << "\n";

    cout << "\nValidation: "
         << (correct ? "PASSED" : "FAILED") << "\n";

    cout << "Serial checksum: " << serial_checksum << "\n";
    cout << "SYCL checksum: " << sycl_checksum << "\n";

    cout << "\nMedian Execution Times\n";

    cout << "Serial Execution: "
         << serial_median << " ms\n";

    cout << "SYCL Kernel Execution: "
         << kernel_median << " ms\n";

    cout << "SYCL Submission-to-Completion: "
         << total_median << " ms\n";

    cout << "\nSYCL Speedup: "
         << serial_median / kernel_median << "x\n";

    return correct ? 0 : 1;
}