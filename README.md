# COE506_Project: GPU-Accelerated Point-in-Polygon Aggregation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements high-performance point-in-polygon aggregation using multiple GPU programming approaches. The problem involves determining which geographic points fall within specific polygon regions and aggregating values associated with those points. This is a common computational geometry problem with applications in GIS, spatial analysis, data visualization, and geographic data processing.

**Course:** COE-506 GPU Programming and Architecture
**Instructor:** Dr. Ayaz ul Hassan Khan
**Institution:** King Fahd University of Petroleum and Minerals

## Table of Contents

- [Features](#features)
- [Implementations](#implementations)
- [Performance Results](#performance-results)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results Verification](#results-verification)
- [Documentation](#documentation)
- [License](#license)

## Features

- **Multiple GPU Programming Paradigms**: Implementations using OpenACC, Numba CUDA, and CUDA C++
- **Performance Comparison**: Detailed benchmarking against naive CPU baseline
- **Profiling Integration**: NVIDIA Nsight Systems profiling for all implementations
- **Scalable Testing**: Three datasets of increasing size for scalability analysis
- **Verification System**: Automated correctness verification against baseline implementation
- **Google Colab Compatible**: Complete notebook for running on Google Colab with GPU support

## Implementations

### 1. Naive CPU Implementation (Baseline)
- **Language:** C
- **Approach:** Sequential CPU processing
- **Purpose:** Baseline for performance comparison
- **File:** `codes/naive_PointInPloy.c`

### 2. OpenACC Implementation
- **Language:** C with OpenACC directives
- **Approach:** Directive-based GPU programming
- **Advantages:** Portable, high-level abstraction
- **File:** `codes/openacc_code.c`

### 3. Numba CUDA Implementation
- **Language:** Python with Numba decorators
- **Approach:** Python-based GPU programming
- **Advantages:** Rapid prototyping, Python ecosystem integration
- **File:** `codes/numba_impl.py`

### 4. CUDA C++ Implementation
- **Language:** CUDA C++
- **Approach:** Low-level GPU programming with Thrust library
- **Advantages:** Maximum performance and control
- **File:** `codes/main_cuda.cu`

## Performance Results

All implementations were tested on NVIDIA Tesla T4 GPU with three datasets of increasing scale:

| Implementation | Dataset 1 (100K points, 1K regions) | Dataset 2 (500K points, 5K regions) | Dataset 3 (1M points, 10K regions) | Speedup vs CPU (Dataset 3) |
|----------------|-------------------------------------|-------------------------------------|-------------------------------------|---------------------------|
| **Naive CPU**  | 483.4 ms                            | 10,653.7 ms                         | 31,660.6 ms                         | 1.0× (baseline)           |
| **OpenACC**    | 566.6 ms                            | 643.6 ms                            | 929.5 ms                            | **34.1×**                 |
| **Numba CUDA** | 4,525.6 ms                          | 4,550.1 ms                          | 7,252.1 ms                          | 4.4×                      |
| **CUDA C++**   | 610.6 ms                            | 680.8 ms                            | 1,141.1 ms                          | **27.7×**                 |

**Note:** CUDA C++ times shown are aggregation-only. Total times including data loading: 1,733.9 ms (Dataset 1), 2,427.9 ms (Dataset 2), 4,200.3 ms (Dataset 3).

**Key Findings:**
- **OpenACC** achieves best performance with up to **34× speedup** on largest dataset
- **CUDA C++** provides **28× speedup** with fine-grained control
- GPU implementations scale significantly better with increasing dataset size
- OpenACC and CUDA C++ maintain consistent sub-second performance across all datasets
- Numba implementation shows room for optimization (grid size under-utilization warnings)
- See `preformance_results/` directory for detailed profiling logs

## Repository Structure

```
COE506_Project/
├── Colab_OpenACC_Numba_CUDA.ipynb    # Main Jupyter notebook for Google Colab
├── README.md                          # This file
├── DOCUMENTATION.md                   # Detailed technical documentation
├── LICENSE                            # MIT License
│
├── codes/                             # Source code implementations
│   ├── naive_PointInPloy.c           # CPU baseline implementation
│   ├── openacc_code.c                # OpenACC GPU implementation
│   ├── numba_impl.py                 # Numba CUDA implementation
│   └── main_cuda.cu                  # CUDA C++ implementation
│
├── data/                              # Input datasets
│   ├── points1.csv                   # 100,000 points
│   ├── polygons1.csv                 # 1,000 regions
│   ├── points2.csv                   # 500,000 points
│   ├── polygons2.csv                 # 5,000 regions
│   ├── points3.csv                   # 1,000,000 points
│   └── polygons3.csv                 # 10,000 regions
│
├── output_data/                       # Results from each implementation
│   ├── naive/                        # CPU baseline results
│   ├── openacc/                      # OpenACC results
│   ├── numba/                        # Numba results
│   └── cuda_c/                       # CUDA C++ results
│
├── preformance_results/              # Performance logs and metrics
│   ├── naive_nsys_output_log.txt
│   ├── openacc_nsys_output_log.txt
│   ├── numba_nsys_output_log.txt
│   └── cuda_nsys_output_log.txt
│
└── Profiling/                         # NVIDIA Nsight Systems profiles
    ├── naive/                        # CPU baseline profiles
    ├── openacc/                      # OpenACC profiles
    ├── numba/                        # Numba profiles
    └── cuda_c/                       # CUDA C++ profiles
```

## Setup Instructions

### Running on Google Colab (Recommended - Easiest Method)

**Quick Start (3 Simple Steps):**

1. **Open the notebook directly from GitHub:**
   - Go to: https://colab.research.google.com/github/kamelth/COE506_Project/blob/main/Colab_OpenACC_Numba_CUDA.ipynb
   - Or manually upload `Colab_OpenACC_Numba_CUDA.ipynb` to Colab

2. **Enable GPU runtime:**
   - Runtime → Change runtime type
   - Hardware accelerator → GPU (Tesla T4 recommended)
   - Save

3. **Run the setup cells:**
   - **Cell 1:** Check GPU availability (`nvidia-smi`)
   - **Cell 2:** Mount Google Drive (you'll be prompted to authorize)
   - **Cell 3:** Setup directories in your Google Drive
   - **Cell 4:** **Automatic Project Setup** 🚀
     - The notebook will automatically clone the project from GitHub
     - Copy all source code and data files to your Google Drive
     - Skip this step if files already exist (smart detection)
     - No manual file uploads needed!

4. **Continue with the rest of the notebook:**
   - Install NVIDIA HPC SDK (one-time, ~7 minutes)
   - Run all implementations and see the results!

**What Happens Automatically:**
- ✅ Project cloned from GitHub
- ✅ Source code copied to Google Drive (`COE506_Project/codes/`)
- ✅ Data files copied to Google Drive (`COE506_Project/data/`)
- ✅ Directory structure created automatically
- ✅ Temporary files cleaned up

**Alternative Method (Manual Upload):**
If you prefer to manually upload files:
1. Create a folder named `COE506_Project` in your Google Drive
2. Upload `codes/` and `data/` directories maintaining the structure
3. The notebook will detect existing files and skip the automatic clone

### Running Locally (Requires NVIDIA GPU)

#### Prerequisites
- NVIDIA GPU with CUDA capability 3.5 or higher
- CUDA Toolkit 11.0 or later
- NVIDIA HPC SDK (for OpenACC)
- Python 3.7+ with Numba and NumPy
- GCC/G++ compiler

#### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kamelth/COE506_Project.git
   cd COE506_Project
   ```

2. **Install NVIDIA HPC SDK:**
   ```bash
   # For Ubuntu/Debian
   curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
   echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] \
     https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | \
     sudo tee /etc/apt/sources.list.d/nvhpc.list
   sudo apt-get update
   sudo apt-get install -y nvhpc-22-11
   ```

3. **Install Python dependencies:**
   ```bash
   pip install numba numpy pandas
   ```

4. **Load NVIDIA HPC SDK modules:**
   ```bash
   source /usr/share/modules/init/bash
   module use /opt/nvidia/hpc_sdk/modulefiles
   module load nvhpc/22.11
   ```

## Usage

### Compiling the Implementations

#### Naive CPU Implementation:
```bash
nvc -acc -ta=multicore -Minfo=accel -fast \
    -I/usr/local/cuda/include/nvtx3 \
    -L/usr/local/cuda/lib64 -lnvToolsExt \
    codes/naive_PointInPloy.c -o codes/naive_PointInPloy
```

#### OpenACC Implementation:
```bash
nvc -acc -ta=tesla -Minfo=accel -fast \
    -I/usr/local/cuda/include/nvtx3 \
    -L/usr/local/cuda/lib64 -lnvToolsExt \
    codes/openacc_code.c -o codes/openacc_code
```

#### CUDA C++ Implementation:
```bash
nvcc -std=c++14 -arch=sm_75 --extended-lambda -O2 \
    codes/main_cuda.cu -o codes/main_cuda
```

### Running the Implementations

#### Example for Dataset 1:
```bash
# Naive CPU
./codes/naive_PointInPloy data/points1.csv data/polygons1.csv output_data/naive/out_1.csv

# OpenACC
./codes/openacc_code data/points1.csv data/polygons1.csv output_data/openacc/out_1.csv

# Numba
python codes/numba_impl.py data/points1.csv data/polygons1.csv output_data/numba/out_1.csv

# CUDA C++
./codes/main_cuda data/points1.csv data/polygons1.csv output_data/cuda_c/out_1.csv
```

### Profiling with NVIDIA Nsight Systems

```bash
nsys profile --force-overwrite true -o profile_output \
    ./codes/main_cuda data/points1.csv data/polygons1.csv output_data/cuda_c/out_1.csv
```

## Datasets

The project includes three datasets of increasing scale for performance analysis:

- **Dataset 1:** 100,000 points × 1,000 regions
- **Dataset 2:** 500,000 points × 5,000 regions
- **Dataset 3:** 1,000,000 points × 10,000 regions

### Data Format

**Points CSV:**
```
latitude,longitude,value
40.7128,-74.0060,10.5
34.0522,-118.2437,15.2
...
```

**Polygons CSV:**
```
region_id,num_vertices,vertex1_lat,vertex1_lon,vertex2_lat,vertex2_lon,...
0,4,40.0,−75.0,41.0,−75.0,41.0,−74.0,40.0,−74.0
...
```

## Results Verification

All GPU implementations have been verified against the naive CPU baseline for correctness. The verification script compares output CSV files:

```python
# Run verification (included in notebook)
# All implementations match baseline with 100% accuracy
```

## Documentation

For detailed technical documentation including:
- Algorithm implementation details
- GPU optimization strategies
- Performance analysis methodology
- Compilation instructions
- Troubleshooting guide

Please see [DOCUMENTATION.md](DOCUMENTATION.md)

## Contributing

This is an academic project for COE-506 course. If you'd like to contribute improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Team

**COE506 Project Team**
Course: GPU Programming and Architecture
Instructor: Dr. Ayaz ul Hassan Khan

## Acknowledgments

- NVIDIA Corporation for CUDA Toolkit and HPC SDK
- Google Colab for providing free GPU resources
- Dr. Ayaz ul Hassan Khan for course guidance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/
2. OpenACC Specification: https://www.openacc.org/
3. Numba Documentation: https://numba.pydata.org/
4. Point-in-Polygon Algorithms: https://en.wikipedia.org/wiki/Point_in_polygon

---

**Note:** This project demonstrates GPU acceleration techniques for academic and educational purposes. For production use, additional optimization and error handling may be required.
