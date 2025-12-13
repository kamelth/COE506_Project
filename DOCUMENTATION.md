# COE506 Project - Technical Documentation

## Table of Contents

1. [Problem Description](#problem-description)
2. [Algorithm Overview](#algorithm-overview)
3. [Implementation Details](#implementation-details)
4. [GPU Optimization Strategies](#gpu-optimization-strategies)
5. [Compilation and Build Instructions](#compilation-and-build-instructions)
6. [Performance Analysis](#performance-analysis)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Problem Description

### Point-in-Polygon Aggregation

The point-in-polygon problem is a fundamental computational geometry problem that determines whether a given point lies inside, outside, or on the boundary of a polygon. This project extends the basic problem to perform aggregation:

**Input:**
- A set of N points, each with coordinates (latitude, longitude) and an associated value
- A set of M polygons (regions), each defined by a list of vertices

**Output:**
- For each polygon: count of points inside the polygon and sum of their values

**Applications:**
- Geographic Information Systems (GIS)
- Spatial data analysis
- Electoral district analysis
- Environmental monitoring
- Urban planning

### Computational Complexity

- **Naive approach:** O(N × M × V) where V is average vertices per polygon
- **Challenge:** With N=1M points and M=10K regions, this requires ~10 billion operations
- **Solution:** GPU parallelization to process multiple points simultaneously

---

## Algorithm Overview

### Ray Casting Algorithm

The project uses the **ray casting algorithm** to determine point-in-polygon containment:

1. Cast a ray from the point to infinity (typically along x-axis)
2. Count intersections with polygon edges
3. If count is odd → point is inside; if even → point is outside

```c
bool point_in_polygon(float px, float py, float* vertices, int num_vertices) {
    bool inside = false;
    for (int i = 0, j = num_vertices - 1; i < num_vertices; j = i++) {
        float xi = vertices[i*2], yi = vertices[i*2+1];
        float xj = vertices[j*2], yj = vertices[j*2+1];

        bool intersect = ((yi > py) != (yj > py)) &&
                         (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}
```

### Aggregation Strategy

For each point:
1. Test against all polygons
2. If inside a polygon, increment count and add value
3. Store results per polygon

---

## Implementation Details

### 1. Naive CPU Implementation

**File:** `codes/naive_PointInPloy.c`

**Key Characteristics:**
- Sequential processing of all points
- Nested loops: outer loop over points, inner loop over regions
- No parallelization
- Serves as correctness baseline

**Code Structure:**
```c
void aggregate_cpu(Points* points, Regions* regions, Results* results) {
    for (int p = 0; p < num_points; p++) {
        for (int r = 0; r < num_regions; r++) {
            if (point_in_polygon(points[p], regions[r])) {
                results[r].count++;
                results[r].sum += points[p].value;
            }
        }
    }
}
```

---

### 2. OpenACC Implementation

**File:** `codes/openacc_code.c`

**Parallelization Strategy:**
- `#pragma acc data`: Manages GPU memory transfers
- `#pragma acc parallel loop`: Parallelizes outer loop over points
- `#pragma acc routine seq`: Marks helper functions for GPU execution

**Key Directives:**
```c
#pragma acc routine seq
bool point_in_polygon(...) { ... }

#pragma acc parallel loop gang vector(256) \
    copyin(points[0:num_points], polygons[0:total_vertices]) \
    copy(results[0:num_regions])
for (int p = 0; p < num_points; p++) {
    // Process point p
}
```

**Memory Management:**
- `copyin`: Transfer input data to GPU (read-only)
- `copy`: Transfer results to/from GPU
- `create`: Allocate temporary GPU memory

**Compilation:**
```bash
nvc -acc -ta=tesla -Minfo=accel codes/openacc_code.c -o openacc_code
```

**Advantages:**
- Portable across different accelerators
- Minimal code changes from CPU version
- Compiler handles low-level details

---

### 3. Numba CUDA Implementation

**File:** `codes/numba_impl.py`

**Parallelization Strategy:**
- `@cuda.jit`: JIT compilation for GPU kernels
- Grid-stride loop pattern for scalability
- NumPy arrays for data management

**Kernel Structure:**
```python
@cuda.jit
def aggregate_kernel(points_lat, points_lon, points_val,
                     polygon_vertices, results_count, results_sum):
    # Thread index calculation
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Grid-stride loop
    for p in range(idx, num_points, stride):
        lat, lon, val = points_lat[p], points_lon[p], points_val[p]
        for r in range(num_regions):
            if point_in_polygon(lat, lon, polygon_vertices[r]):
                cuda.atomic.add(results_count, r, 1)
                cuda.atomic.add(results_sum, r, val)
```

**Memory Transfer:**
```python
# Host to device
d_points_lat = cuda.to_device(points_lat)
d_results_count = cuda.device_array(num_regions, dtype=np.int32)

# Launch kernel
threads_per_block = 256
blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
aggregate_kernel[blocks_per_grid, threads_per_block](...)

# Device to host
results_count = d_results_count.copy_to_host()
```

**Advantages:**
- Python ecosystem integration
- Rapid prototyping
- Automatic memory management
- Interactive development

---

### 4. CUDA C++ Implementation

**File:** `codes/main_cuda.cu`

**Parallelization Strategy:**
- Thrust library for high-performance data structures
- Custom CUDA kernel with optimized memory access
- Atomic operations for thread-safe aggregation

**Kernel Implementation:**
```cuda
__global__ void aggregate_kernel(
    float* points_lat, float* points_lon, float* points_val,
    int num_points, RegionDevice* regions, int num_regions,
    int* results_count, float* results_sum) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int p = idx; p < num_points; p += stride) {
        float lat = points_lat[p];
        float lon = points_lon[p];
        float val = points_val[p];

        for (int r = 0; r < num_regions; r++) {
            if (point_in_polygon_device(lat, lon, regions[r])) {
                atomicAdd(&results_count[r], 1);
                atomicAdd(&results_sum[r], val);
            }
        }
    }
}
```

**Device Function:**
```cuda
__device__ bool point_in_polygon_device(
    float px, float py, const RegionDevice& region) {
    bool inside = false;
    for (int i = 0, j = region.num_vertices - 1; i < region.num_vertices; j = i++) {
        float xi = region.lats[i], yi = region.lons[i];
        float xj = region.lats[j], yj = region.lons[j];

        bool intersect = ((yi > py) != (yj > py)) &&
                         (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}
```

**Thrust Data Structures:**
```cpp
thrust::device_vector<float> d_points_lat(points_lat);
thrust::device_vector<int> d_results_count(num_regions, 0);
```

**Compilation:**
```bash
nvcc -std=c++14 -arch=sm_75 --extended-lambda -O2 \
     codes/main_cuda.cu -o main_cuda
```

**Advantages:**
- Maximum performance
- Fine-grained control over GPU resources
- Access to latest CUDA features
- Optimized memory patterns

---

## GPU Optimization Strategies

### 1. Memory Coalescing

**Problem:** Non-coalesced memory access reduces bandwidth utilization

**Solution:** Structure data for sequential access patterns
```cpp
// Good: Structure of Arrays (SoA)
float* lat = ...;  // lat[0], lat[1], lat[2], ...
float* lon = ...;  // lon[0], lon[1], lon[2], ...

// Bad: Array of Structures (AoS)
struct Point { float lat, lon; };  // lat[0], lon[0], lat[1], lon[1], ...
```

### 2. Atomic Operations

**Challenge:** Multiple threads updating same memory location

**Solution:** Atomic operations ensure thread-safe updates
```cuda
atomicAdd(&results_count[r], 1);      // Thread-safe increment
atomicAdd(&results_sum[r], val);      // Thread-safe addition
```

**Trade-off:** Atomics can serialize execution; use when necessary

### 3. Grid-Stride Loops

**Purpose:** Handle arrays larger than maximum threads

**Pattern:**
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = idx; i < N; i += stride) {
    // Process element i
}
```

**Advantages:**
- Scalability across different GPU sizes
- Automatic load balancing
- Kernel works for any input size

### 4. Occupancy Optimization

**Threads per block:** 256 (good balance between occupancy and resources)

**Calculation:**
```python
threads_per_block = 256
blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
```

**Considerations:**
- More threads → better latency hiding
- Fewer threads → more resources per thread
- Optimal value depends on kernel characteristics

---

## Compilation and Build Instructions

### Prerequisites

#### For All Implementations:
- NVIDIA GPU with Compute Capability 3.5+
- CUDA Toolkit 11.0 or later
- GCC 7.3 or later

#### OpenACC:
- NVIDIA HPC SDK 22.11 or later

#### Numba:
- Python 3.7+
- Numba 0.54+
- NumPy 1.20+

### Installation

#### 1. NVIDIA HPC SDK (Ubuntu/Debian)

```bash
# Add NVIDIA HPC SDK repository
curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg

echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] \
  https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | \
  sudo tee /etc/apt/sources.list.d/nvhpc.list

# Install
sudo apt-get update
sudo apt-get install -y nvhpc-22-11

# Setup environment
source /usr/share/modules/init/bash
module use /opt/nvidia/hpc_sdk/modulefiles
module load nvhpc/22.11
```

#### 2. Python Environment

```bash
# Create virtual environment
python3 -m venv gpu_env
source gpu_env/bin/activate

# Install dependencies
pip install numba numpy pandas
```

### Compilation Commands

#### Naive CPU Implementation:
```bash
nvc -acc -ta=multicore -Minfo=accel -fast \
    -I/usr/local/cuda/include/nvtx3 \
    -L/usr/local/cuda/lib64 -lnvToolsExt \
    codes/naive_PointInPloy.c -o codes/naive_PointInPloy
```

**Flags:**
- `-acc`: Enable OpenACC
- `-ta=multicore`: Target multicore CPU (for naive version)
- `-Minfo=accel`: Print accelerator information
- `-fast`: Aggressive optimizations
- `-lnvToolsExt`: Link NVTX profiling library

#### OpenACC Implementation:
```bash
nvc -acc -ta=tesla -Minfo=accel -fast \
    -I/usr/local/cuda/include/nvtx3 \
    -L/usr/local/cuda/lib64 -lnvToolsExt \
    codes/openacc_code.c -o codes/openacc_code
```

**Flags:**
- `-ta=tesla`: Target NVIDIA Tesla GPU
- Output shows GPU code generation details

#### CUDA C++ Implementation:
```bash
# Auto-detect GPU architecture
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//')
nvcc -std=c++14 -arch=sm_${GPU_ARCH} --extended-lambda -O2 \
     codes/main_cuda.cu -o codes/main_cuda
```

**Flags:**
- `-std=c++14`: C++14 standard
- `-arch=sm_75`: GPU architecture (75 = Turing/T4)
- `--extended-lambda`: Allow lambda in device code
- `-O2`: Optimization level 2

---

## Performance Analysis

### Profiling with NVIDIA Nsight Systems

#### Command:
```bash
nsys profile --force-overwrite true \
     -o profiling_output \
     ./codes/main_cuda data/points1.csv data/polygons1.csv output.csv
```

#### Generated Files:
- `profiling_output.nsys-rep`: Binary profile data
- Open in NVIDIA Nsight Systems GUI for visualization

#### Key Metrics:
- **Kernel Duration:** Time spent in GPU kernel
- **Memory Transfers:** Host↔Device transfer time
- **GPU Utilization:** Percentage of time GPU is active
- **Occupancy:** Ratio of active warps to maximum warps

### Performance Comparison Matrix

| Metric | Naive CPU | OpenACC | Numba | CUDA C++ |
|--------|-----------|---------|-------|----------|
| Development Time | Fast | Fast | Medium | Slow |
| Portability | High | High | Medium | Low |
| Performance | Baseline | Good | Good | Best |
| Debugging | Easy | Medium | Medium | Hard |
| Fine-tuning | Limited | Medium | Medium | Maximum |

### Scalability Analysis

**Actual Performance Results (NVIDIA Tesla T4 GPU):**

| Implementation | Dataset 1 (100K, 1K) | Dataset 2 (500K, 5K) | Dataset 3 (1M, 10K) | Speedup (Dataset 3) |
|----------------|---------------------|---------------------|---------------------|---------------------|
| **Naive CPU**  | 474 ms              | 11,513 ms           | 31,485 ms           | 1.0× (baseline)     |
| **CUDA C++**   | 425 ms              | 688 ms              | 874 ms              | **36.0×** 🏆        |
| **OpenACC**    | 693 ms              | 635 ms              | 961 ms              | **32.8×**           |
| **Numba CUDA** | 2,272 ms            | 1,319 ms            | 2,075 ms            | **15.2×**           |

**Performance improvement factor vs naive CPU (Dataset 3):**

- **CUDA C++:** 36.0× speedup (best performance - optimized low-level implementation)
- **OpenACC:** 32.8× speedup (excellent performance with high-level directives)
- **Numba:** 15.2× speedup (good performance for Python-based GPU programming)

**Key Observations:**
- **CUDA C++ achieves best overall speedup (36×)** through low-level optimizations
- **OpenACC delivers excellent performance (32.8×)** with minimal code modifications
- **Numba provides solid Python-based GPU acceleration (15.2×)** for rapid development
- All GPU implementations dramatically outperform CPU baseline
- Performance advantage increases with dataset size (better scalability)
- Both CUDA C++ and OpenACC maintain sub-second performance across all datasets

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce dataset size
- Process data in batches
- Use unified memory
- Reduce threads per block

#### 2. Compilation Errors with nvcc

**Symptom:**
```
error: identifier "__host__" is undefined
```

**Solution:**
- Ensure CUDA Toolkit is properly installed
- Check GPU architecture compatibility
- Verify CUDA version matches driver

#### 3. OpenACC Not Using GPU

**Symptom:**
```
Accelerator kernel not generated
```

**Solutions:**
- Check `-Minfo=accel` output for warnings
- Ensure data directives are correct
- Verify GPU target: `-ta=tesla`
- Check for unsupported operations in loop

#### 4. Incorrect Results

**Symptom:**
- Output doesn't match baseline
- NaN or inf values

**Solutions:**
- Verify atomic operations for race conditions
- Check floating-point precision issues
- Validate input data format
- Run verification script

### Debugging Tips

#### Enable CUDA Error Checking:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

#### OpenACC Environment Variables:
```bash
export ACC_DEVICE_NUM=0              # Select GPU 0
export NV_ACC_DEBUG=1                # Enable debug output
export NV_ACC_TIME=1                 # Print timing info
```

#### Numba Debugging:
```python
from numba import cuda
cuda.detect()  # Print detected CUDA devices

# Enable debug output
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'  # CPU simulation mode
```

---

## Advanced Usage

### Custom Datasets

#### Creating Synthetic Data:

```python
import numpy as np
import pandas as pd

# Generate random points
num_points = 100000
points = pd.DataFrame({
    'latitude': np.random.uniform(30, 45, num_points),
    'longitude': np.random.uniform(-125, -65, num_points),
    'value': np.random.uniform(0, 100, num_points)
})
points.to_csv('custom_points.csv', index=False)

# Generate random polygons
def generate_polygon(center_lat, center_lon, radius, num_vertices):
    angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
    vertices = []
    for angle in angles:
        lat = center_lat + radius * np.cos(angle)
        lon = center_lon + radius * np.sin(angle)
        vertices.extend([lat, lon])
    return vertices

num_regions = 1000
polygons_data = []
for i in range(num_regions):
    center = (np.random.uniform(30, 45), np.random.uniform(-125, -65))
    vertices = generate_polygon(*center, 0.5, 4)
    polygons_data.append([i, 4] + vertices)

polygons_df = pd.DataFrame(polygons_data)
polygons_df.to_csv('custom_polygons.csv', index=False, header=False)
```

### Batch Processing

For very large datasets that don't fit in GPU memory:

```python
def process_in_batches(points, polygons, batch_size=50000):
    results = initialize_results(len(polygons))

    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        batch_results = gpu_aggregate(batch, polygons)
        results += batch_results

    return results
```

### Performance Tuning

#### Optimal Thread Configuration:

```cpp
// Experiment with different configurations
int configs[][2] = {
    {128, -1},  // 128 threads per block, auto blocks
    {256, -1},
    {512, -1},
    {1024, -1}
};

for (auto config : configs) {
    int threads = config[0];
    int blocks = (num_points + threads - 1) / threads;

    // Time kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    aggregate_kernel<<<blocks, threads>>>(...);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Threads: %d, Time: %.3f ms\n", threads, milliseconds);
}
```

---

## Contact and Support

For questions or issues:
- Check existing [GitHub Issues](https://github.com/yourusername/COE506_Project/issues)
- Create new issue with detailed description
- Include: GPU model, CUDA version, error messages, minimal reproducible example

## Additional Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [OpenACC Best Practices](https://www.openacc.org/resources)
- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [GPU Gems (Free Online Book)](https://developer.nvidia.com/gpugems/gpugems3/contributors)

---

*Last Updated: December 2025*
