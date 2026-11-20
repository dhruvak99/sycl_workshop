import numpy as np
import dpnp as dp
import dpctl
import time
import statistics

N = 10_000_000
ITERATIONS = 5

print(f"Initializing {N:,} elements...")

# Create SYCL queue
queue = dpctl.SyclQueue()

# Initialize input data
data = np.ones(N, dtype=np.float32)

# Transfer data to SYCL device
dp_data = dp.asarray(data, sycl_queue=queue)

python_times = []
numpy_times = []
dpnp_times = []

# Warm-up
warmup_result = dp.sum(dp_data)
queue.wait()

for iteration in range(ITERATIONS):

    # 1. Standard Python
    start = time.perf_counter()

    python_result = sum(data)

    python_end = time.perf_counter()

    python_times.append(
        (python_end - start) * 1000
    )

    # 2. NumPy
    start = time.perf_counter()

    numpy_result = np.sum(data)

    numpy_end = time.perf_counter()

    numpy_times.append(
        (numpy_end - start) * 1000
    )

    # 3. DPNP (SYCL)
    start = time.perf_counter()

    dpnp_result = dp.sum(dp_data)
    queue.wait()

    dpnp_end = time.perf_counter()

    dpnp_times.append(
        (dpnp_end - start) * 1000
    )

    print(f"Completed iteration {iteration + 1}/{ITERATIONS}")


# Calculate median timings
python_median = statistics.median(python_times)
numpy_median = statistics.median(numpy_times)
dpnp_median = statistics.median(dpnp_times)

# Extract DPNP result
dpnp_value = dpnp_result.item()

# Display results
print("\n===== PARALLEL REDUCTION BENCHMARK =====")

print(f"Elements: {N:,}")
print(f"Iterations: {ITERATIONS}")

print("\nMedian Execution Times")

print(f"Python: {python_median:.3f} ms")
print(f"NumPy:  {numpy_median:.3f} ms")
print(f"DPNP:   {dpnp_median:.3f} ms")

print("\nResults")

print(f"Python Sum: {python_result}")
print(f"NumPy Sum:  {numpy_result}")
print(f"DPNP Sum:   {dpnp_value}")

# Validate results
correct = (
    np.isclose(python_result, numpy_result)
    and np.isclose(numpy_result, dpnp_value)
)

print(f"\nValidation: {'PASSED' if correct else 'FAILED'}")

print("\nSpeedup")

print(f"DPNP vs Python: {python_median / dpnp_median:.2f}x")
print(f"DPNP vs NumPy:  {numpy_median / dpnp_median:.2f}x")