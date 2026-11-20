import numpy as np
import dpnp as dp
import dpctl
import time
import statistics

# Matrix dimensions
N = 256
ITERATIONS = 5

print(f"Matrix size: {N} x {N}")

# Generate input matrices
np.random.seed(42)

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Convert matrices to DPNP arrays
queue = dpctl.SyclQueue()

dp_A = dp.asarray(A, sycl_queue=queue)
dp_B = dp.asarray(B, sycl_queue=queue)

python_times = []
numpy_times = []
dpnp_times = []

# Warm-up DPNP
warmup = dp.matmul(dp_A, dp_B)
queue.wait()

# Benchmark
for i in range(ITERATIONS):

    # Pure Python matrix multiplication
    start = time.perf_counter()

    python_result = [
        [
            sum(A[i, k] * B[k, j] for k in range(N))
            for j in range(N)
        ]
        for i in range(N)
    ]

    python_end = time.perf_counter()
    python_times.append((python_end - start) * 1000)

    # NumPy matrix multiplication
    start = time.perf_counter()

    numpy_result = np.matmul(A, B)

    numpy_end = time.perf_counter()
    numpy_times.append((numpy_end - start) * 1000)

    # DPNP matrix multiplication
    start = time.perf_counter()

    dpnp_result = dp.matmul(dp_A, dp_B)
    queue.wait()

    dpnp_end = time.perf_counter()
    dpnp_times.append((dpnp_end - start) * 1000)

    print(f"Completed iteration {i + 1}/{ITERATIONS}")

# Calculate median execution times
python_median = statistics.median(python_times)
numpy_median = statistics.median(numpy_times)
dpnp_median = statistics.median(dpnp_times)

# Convert results for validation
python_result = np.array(python_result, dtype=np.float32)
dpnp_result = dp.asnumpy(dpnp_result)

# Validate results
correct = (
    np.allclose(python_result, numpy_result, rtol=1e-4, atol=1e-4)
    and np.allclose(numpy_result, dpnp_result, rtol=1e-4, atol=1e-4)
)

print("\n===== MATRIX MULTIPLICATION BENCHMARK =====")

print(f"Matrix dimensions: {N} x {N}")
print(f"Iterations: {ITERATIONS}")

print("\nMedian Execution Times")

print(f"Python: {python_median:.3f} ms")
print(f"NumPy:  {numpy_median:.3f} ms")
print(f"DPNP:   {dpnp_median:.3f} ms")

print("\nValidation")
print(f"Result: {'PASSED' if correct else 'FAILED'}")

print("\nSpeedup")

print(f"DPNP vs Python: {python_median / dpnp_median:.2f}x")
print(f"DPNP vs NumPy:  {numpy_median / dpnp_median:.2f}x")