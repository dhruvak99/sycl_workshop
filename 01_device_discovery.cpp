#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        sycl::device device{sycl::default_selector_v};

        std::cout << "===== SYCL DEVICE INFORMATION =====\n";

        std::cout << "Device Name: "
                  << device.get_info<sycl::info::device::name>()
                  << "\n";

        std::cout << "Vendor: "
                  << device.get_info<sycl::info::device::vendor>()
                  << "\n";

        std::cout << "Driver Version: "
                  << device.get_info<sycl::info::device::driver_version>()
                  << "\n";

        std::cout << "Device Type: ";

        if (device.is_cpu())
            std::cout << "CPU\n";
        else if (device.is_gpu())
            std::cout << "GPU\n";
        else if (device.is_accelerator())
            std::cout << "Accelerator\n";

        std::cout << "Maximum Compute Units: "
                  << device.get_info<sycl::info::device::max_compute_units>()
                  << "\n";

        std::cout << "Maximum Work Group Size: "
                  << device.get_info<sycl::info::device::max_work_group_size>()
                  << "\n";

        std::cout << "Global Memory: "
                  << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024)
                  << " MB\n";

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
