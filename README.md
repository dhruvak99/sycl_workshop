# Intel oneAPI & SYCL Workshop --- Setup Guide

**RV College of Engineering (RVCE)**\
**Date:** 20 November 2026\
**Time:** 2:00 PM -- 4:00 PM\
**Venue:** PG Laboratory\
**Resource Persons:** Rohith Arsha and Dhruva K

This README explains how to prepare your system before attending the
hands-on workshop. Please complete the installation steps before the
session so that we can spend the workshop time on the demonstrations and
project development.

------------------------------------------------------------------------

## 1. What you will install

The workshop uses:

-   **Intel oneAPI DPC++/C++ Compiler (`icpx`)** --- to compile SYCL C++
    programs.
-   **Python 3 and `venv`** --- to create an isolated Python
    environment.
-   **NumPy and Pillow** --- for numerical and image-processing
    examples.
-   **dpnp** --- a NumPy-like library that uses SYCL-backed
    data-parallel operations.
-   **Git** --- optional, but useful for obtaining and managing the
    workshop examples.

The demonstrations use CPU execution. **A dedicated GPU is not
required.**

## 2. Supported setup options

Choose the instructions that match your computer:

  -----------------------------------------------------------------------
  Operating system                    Recommended setup
  ----------------------------------- -----------------------------------
  Windows 10/11 (x64)                 Ubuntu 24.04 in **WSL 2**

  Ubuntu Linux (x64)                  Native Ubuntu installation

  Other Linux distributions           Check Intel oneAPI support for your
                                      distribution and toolkit release
                                      before installing
  -----------------------------------------------------------------------

> **Windows users:** Run the Linux installation commands inside the
> Ubuntu WSL terminal, not in PowerShell. PowerShell is used only for
> installing or launching WSL.

------------------------------------------------------------------------

## 3. Windows: Install Ubuntu using WSL 2

### 3.1 Install WSL

Open **PowerShell as Administrator** and run:

``` powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if prompted. Launch **Ubuntu 24.04** from the Start menu
and complete the first-time setup by creating a Linux username and
password.

Check that WSL 2 is being used. In PowerShell:

``` powershell
wsl --list --verbose
```

The Ubuntu distribution should show version `2`. If it shows version
`1`, convert it:

``` powershell
wsl --set-version Ubuntu-24.04 2
```

### 3.2 Update Ubuntu and install prerequisites

Run the following commands in the **Ubuntu WSL terminal**:

``` bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential wget gpg git python3 python3-venv python3-pip python3-full
```

Continue to [Section 4](#4-install-intel-oneapi-dpc-compiler).

------------------------------------------------------------------------

## 4. Install Intel oneAPI DPC++ Compiler

These commands are run in an Ubuntu terminal, whether it is WSL 2 or
native Ubuntu.

### 4.1 Add Intel's APT repository

``` bash
sudo apt install -y wget gpg

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
```

### 4.2 Install the compiler

For this workshop, install the DPC++/C++ compiler package:

``` bash
sudo apt install -y intel-oneapi-compiler-dpcpp-cpp
```

You do not need to install the entire oneAPI toolkit for the workshop
demonstrations.

### 4.3 Initialize and verify the compiler

Run this each time you open a new terminal before compiling SYCL
programs:

``` bash
source /opt/intel/oneapi/setvars.sh
```

Check the compiler:

``` bash
icpx --version
```

You should see Intel oneAPI DPC++/C++ Compiler information.

To check the available SYCL devices:

``` bash
sycl-ls
```

The exact device list depends on your hardware, drivers and environment.
In WSL, use CPU execution for this workshop; do not assume that GPU
execution is configured.

------------------------------------------------------------------------

## 5. Set up the Python environment

Create a folder for the workshop and a virtual environment:

``` bash
mkdir -p ~/sycl-workshop
cd ~/sycl-workshop

python3 -m venv sycl-env
source sycl-env/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pillow
```

Install DPNP from Intel's Python package index:

``` bash
python -m pip install --index-url https://software.repos.intel.com/python/pypi dpnp
```

Check that the packages can be imported:

``` bash
python -c "import numpy, PIL, dpnp, dpctl; print('NumPy:', numpy.__version__); print('DPNP:', dpnp.__version__)"
```

List available SYCL platforms and devices:

``` bash
python -m dpctl --full-list
```

If DPNP installation or device discovery fails, keep the error message
for the workshop team. Package availability and supported devices can
depend on the Python version, operating system, processor and Intel
runtime.

> Activate the environment whenever you work on the Python
> demonstrations:
>
> ``` bash
> cd ~/sycl-workshop
> source sycl-env/bin/activate
> ```

------------------------------------------------------------------------

## 6. Verify your setup

Before attending, open a terminal and run:

``` bash
source /opt/intel/oneapi/setvars.sh
icpx --version
sycl-ls
```

Then activate your Python environment and run:

``` bash
source ~/sycl-workshop/sycl-env/bin/activate
python -c "import numpy, PIL, dpnp, dpctl; print('Python dependencies OK')"
python -m dpctl --full-list
```

Your setup is ready if:

-   `icpx --version` prints compiler information.
-   `sycl-ls` lists at least one usable SYCL device.
-   The Python import test completes without an error.
-   `dpctl --full-list` displays the available SYCL platform/device
    information.

------------------------------------------------------------------------

## 7. What we will demonstrate

The hands-on session includes:

1.  **Device Discovery (SYCL C++)** --- inspect the available compute
    device.
2.  **Vector Processing (SYCL C++)** --- perform data-parallel
    computation.
3.  **Parallel Reduction (Python + DPNP)** --- calculate an aggregate
    over an array.
4.  **Matrix Multiplication (Python + DPNP)** --- multiply matrices
    using array operations.
5.  **Image Processing (Python + DPNP)** --- convert an image to
    grayscale.

The Python examples use DPNP operations backed by SYCL. They do not
require you to write custom SYCL kernels in Python.

## 8. Troubleshooting

### `icpx: command not found`

Initialize the oneAPI environment in the current terminal:

``` bash
source /opt/intel/oneapi/setvars.sh
```

If the file does not exist, check whether the compiler package installed
successfully.

### `sycl-ls` is not found

Initialize oneAPI first:

``` bash
source /opt/intel/oneapi/setvars.sh
```

If it is still unavailable, check the compiler/runtime installation.

### Python reports `No module named dpnp`

Activate the virtual environment and install DPNP:

``` bash
source ~/sycl-workshop/sycl-env/bin/activate
python -m pip install --index-url https://software.repos.intel.com/python/pypi dpnp
```

### DPNP cannot find a device

Check the device list:

``` bash
python -m dpctl --full-list
```

DPNP requires a compatible SYCL runtime and device. Do not assume that
installing the Python package automatically enables GPU support. For the
workshop, CPU execution is sufficient.

### WSL commands are not recognized

Make sure Linux commands such as `sudo`, `apt` and `source` are being
run inside the Ubuntu WSL terminal, not directly in PowerShell.

------------------------------------------------------------------------

## 9. Important notes

-   Use a **64-bit Intel/AMD x86-64 system** where possible. Check
    Intel's current product requirements for the exact compiler release
    and your processor/OS combination.
-   WSL 2 is the recommended Windows route for this workshop. The Linux
    commands above are executed inside Ubuntu.
-   The workshop examples are designed to run on a CPU. GPU acceleration
    is optional and is not part of the required setup.
-   Keep your laptop connected to power and bring your charger.
-   Complete the installation and verification before **20 November
    2026**.

## 10. Useful official resources

-   [Intel oneAPI
    Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
-   [Intel oneAPI Installation
    Guides](https://www.intel.com/content/www/us/en/developer/tools/oneapi/installation-guide.html)
-   [Intel DPC++/C++
    Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
-   [Microsoft: Install
    WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

------------------------------------------------------------------------

**See you at the workshop!**
