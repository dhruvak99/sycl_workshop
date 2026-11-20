import numpy as np
import dpnp as dp
import dpctl

from PIL import Image
import time
import statistics

ITERATIONS = 5

# Load input image
image = Image.open("input.jpg").convert("RGB")
image_array = np.asarray(image, dtype=np.float32)

height, width, _ = image_array.shape

print(f"Image dimensions: {width} x {height}")
print(f"Total pixels: {width * height:,}")

# RGB weights for grayscale conversion
weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Create SYCL queue
queue = dpctl.SyclQueue()

# Transfer image to DPNP
dp_image = dp.asarray(image_array, sycl_queue=queue)
dp_weights = dp.asarray(weights, sycl_queue=queue)

python_times = []
numpy_times = []
dpnp_times = []

# Warm-up
warmup = dp.sum(dp_image * dp_weights, axis=2)
queue.wait()

for i in range(ITERATIONS):

    # Pure Python
    start = time.perf_counter()

    python_gray = [
        [
            0.299 * image_array[y, x, 0]
            + 0.587 * image_array[y, x, 1]
            + 0.114 * image_array[y, x, 2]
            for x in range(width)
        ]
        for y in range(height)
    ]

    python_end = time.perf_counter()
    python_times.append((python_end - start) * 1000)

    # NumPy
    start = time.perf_counter()

    numpy_gray = np.sum(image_array * weights, axis=2)

    numpy_end = time.perf_counter()
    numpy_times.append((numpy_end - start) * 1000)

    # DPNP
    start = time.perf_counter()

    dpnp_gray = dp.sum(dp_image * dp_weights, axis=2)
    queue.wait()

    dpnp_end = time.perf_counter()
    dpnp_times.append((dpnp_end - start) * 1000)

    print(f"Completed iteration {i + 1}/{ITERATIONS}")

# Median timings
python_median = statistics.median(python_times)
numpy_median = statistics.median(numpy_times)
dpnp_median = statistics.median(dpnp_times)

# Convert results for validation and saving
python_gray = np.asarray(python_gray, dtype=np.float32)
dpnp_gray = dp.asnumpy(dpnp_gray)

correct = (
    np.allclose(python_gray, numpy_gray, rtol=1e-4, atol=1e-3)
    and np.allclose(numpy_gray, dpnp_gray, rtol=1e-4, atol=1e-3)
)

# Save grayscale image
output = np.clip(dpnp_gray, 0, 255).astype(np.uint8)
Image.fromarray(output, mode="L").save("grayscale_output.png")

print("\n===== IMAGE PROCESSING BENCHMARK =====")

print(f"Image: {width} x {height}")
print(f"Pixels: {width * height:,}")

print("\nMedian Execution Times")
print(f"Python: {python_median:.3f} ms")
print(f"NumPy:  {numpy_median:.3f} ms")
print(f"DPNP:   {dpnp_median:.3f} ms")

print("\nValidation")
print(f"Result: {'PASSED' if correct else 'FAILED'}")

print("\nSpeedup")
print(f"DPNP vs Python: {python_median / dpnp_median:.2f}x")
print(f"DPNP vs NumPy:  {numpy_median / dpnp_median:.2f}x")

print("\nSaved: grayscale_output.png")
