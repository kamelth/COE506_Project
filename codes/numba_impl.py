"""
GPU-Accelerated Point-in-Polygon Aggregation using Numba CUDA

This implementation uses the following optimization techniques:
1. Custom CUDA Kernel - Parallelizes point processing across GPU threads
2. Memory Coalescing - Structures data for efficient GPU memory access
3. Atomic Operations - Thread-safe reductions for count and sum per region
4. Flattened Data Structures - Better memory access patterns on GPU
5. Device Memory Management - Minimizes CPU-GPU transfers

Author: GPU-optimized implementation
"""

import numpy as np
import csv
import time
from numba import cuda, float32, int32

# ===================================================
# 1. DATA STRUCTURES & CONSTANTS
# ===================================================

# GPU Configuration for T4 GPU
# T4 GPU specifications:
# - Compute Capability: 7.5
# - Number of SMs: 40
# - Max threads per block: 1024
# - Optimal threads per block: 256 (good balance of occupancy and register usage)
THREADS_PER_BLOCK = 256

# ===================================================
# 2. LOAD POINT DATA FROM CSV
# ===================================================

def load_points(filename):
    """Load points from CSV file (lat, lon, value)."""
    points = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 3:
                lat, lon, val = float(row[0]), float(row[1]), float(row[2])
                points.append([lat, lon, val])
    return np.array(points, dtype=np.float32)

# ===================================================
# 3. LOAD REGIONS FROM CSV
# ===================================================

def load_regions(filename):
    """
    Load regions from CSV file (region_id, x, y).
    Returns flattened arrays for efficient GPU access:
    - polygon_vertices: concatenated vertex coordinates [x0, y0, x1, y1, ...]
    - polygon_sizes: number of vertices per polygon
    - polygon_offsets: starting index in polygon_vertices for each polygon
    """
    regions_dict = {}
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 3:
                region_id = int(row[0])
                x, y = float(row[1]), float(row[2])
                
                if region_id not in regions_dict:
                    regions_dict[region_id] = []
                regions_dict[region_id].append([x, y])
    
    # Sort by region_id to ensure consistent ordering
    sorted_regions = sorted(regions_dict.items())
    
    # Flatten polygons for GPU: interleave x and y coordinates
    # This enables coalesced memory access when reading polygon vertices
    polygon_vertices = []
    polygon_sizes = []
    polygon_offsets = []
    
    offset = 0
    for region_id, vertices in sorted_regions:
        polygon_sizes.append(len(vertices))
        polygon_offsets.append(offset)
        
        # Flatten: [x0, y0, x1, y1, ...] for coalesced access
        for v in vertices:
            polygon_vertices.append(v[0])  # x
            polygon_vertices.append(v[1])  # y
        
        offset += len(vertices) * 2  # 2 floats per vertex
    
    return (np.array(polygon_vertices, dtype=np.float32),
            np.array(polygon_sizes, dtype=np.int32),
            np.array(polygon_offsets, dtype=np.int32))

# ===================================================
# 4. POINT-IN-POLYGON (GPU Device Function)
# ===================================================

@cuda.jit(device=True)
def point_in_polygon_device(x, y, polygon_vertices, polygon_size, polygon_offset):
    """
    GPU device function for point-in-polygon test using ray casting algorithm.
    
    Memory Coalescing: Accesses polygon vertices sequentially, which is
    coalesced when multiple threads check different polygons or when
    threads in a warp access consecutive vertices.
    """
    inside = 0
    n = polygon_size
    
    # Ray casting algorithm
    for i in range(n):
        j = (i + n - 1) % n  # previous vertex
        
        # Get vertex coordinates (coalesced access pattern)
        idx_i = polygon_offset + i * 2
        idx_j = polygon_offset + j * 2
        
        xi = polygon_vertices[idx_i]      # x coordinate
        yi = polygon_vertices[idx_i + 1]  # y coordinate
        xj = polygon_vertices[idx_j]      # x coordinate
        yj = polygon_vertices[idx_j + 1]  # y coordinate
        
        # Check if ray crosses edge
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        
        if intersect:
            inside = 1 - inside  # toggle
    
    return inside

# ===================================================
# 5. GPU AGGREGATION KERNEL (Custom CUDA Kernel)
# ===================================================

@cuda.jit
def aggregate_gpu_kernel(points_lat, points_lon, points_value, num_points,
                         polygon_vertices, polygon_sizes, polygon_offsets, num_regions,
                         count_per_region, sum_per_region):
    """
    Custom CUDA kernel for parallel point-in-polygon aggregation.
    
    OPTIMIZATION TECHNIQUES:
    1. Parallelization: Each thread processes one point (parallel over points)
    2. Memory Coalescing: 
       - Points are stored in separate arrays (lat, lon, value) for coalesced reads
       - Threads with consecutive indices access consecutive memory locations
    3. Early Exit: Breaks when point is found in a polygon (matches C implementation)
    4. Atomic Operations: Thread-safe updates to count and sum arrays
    
    Memory Access Pattern:
    - Thread i reads points_lat[i], points_lon[i], points_value[i] (coalesced)
    - Threads check polygons sequentially (may not be fully coalesced, but acceptable)
    - Atomic writes to count_per_region[r] and sum_per_region[r] (serialized but necessary)
    """
    # Get thread index (each thread processes one point)
    idx = cuda.grid(1)
    
    # Bounds check
    if idx >= num_points:
        return
    
    # Coalesced memory read: consecutive threads read consecutive memory locations
    x = points_lon[idx]   # longitude
    y = points_lat[idx]   # latitude
    value = points_value[idx]
    
    # Check each region until point is found (early exit optimization)
    for r in range(num_regions):
        polygon_size = polygon_sizes[r]
        polygon_offset = polygon_offsets[r]
        
        # Point-in-polygon test
        if point_in_polygon_device(x, y, polygon_vertices, polygon_size, polygon_offset):
            # Atomic operations for thread-safe reduction
            # Note: These serialize writes but are necessary for correctness
            # Syntax: cuda.atomic.add(array, index, value)
            cuda.atomic.add(count_per_region, r, 1)
            cuda.atomic.add(sum_per_region, r, value)
            break  # Early exit: point can only be in one region

# ===================================================
# 6. HOST FUNCTION: GPU AGGREGATION
# ===================================================

def aggregate_gpu(points, polygon_vertices, polygon_sizes, polygon_offsets):
    """
    Host function that orchestrates GPU computation.
    
    Memory Management:
    - Transfers data to GPU device memory
    - Launches CUDA kernel with optimal grid/block configuration
    - Transfers results back to CPU
    """
    num_points = len(points)
    num_regions = len(polygon_sizes)
    
    # Separate point arrays for coalesced memory access
    points_lat = points[:, 0].astype(np.float32)
    points_lon = points[:, 1].astype(np.float32)
    points_value = points[:, 2].astype(np.float32)
    
    # Allocate device memory
    d_points_lat = cuda.to_device(points_lat)
    d_points_lon = cuda.to_device(points_lon)
    d_points_value = cuda.to_device(points_value)
    d_polygon_vertices = cuda.to_device(polygon_vertices)
    d_polygon_sizes = cuda.to_device(polygon_sizes)
    d_polygon_offsets = cuda.to_device(polygon_offsets)
    d_count = cuda.device_array(num_regions, dtype=np.int32)
    d_sum = cuda.device_array(num_regions, dtype=np.float32)
    
    # Initialize device arrays to zero
    d_count[:] = 0
    d_sum[:] = 0.0
    
    # Calculate grid dimensions for T4 GPU kernel launch
    # Unified configuration: 256 threads per block
    # Calculate number of blocks needed to cover all points
    threads_per_block = THREADS_PER_BLOCK
    blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
    
    # Launch kernel with grid/block configuration
    aggregate_gpu_kernel[blocks_per_grid, threads_per_block](
        d_points_lat, d_points_lon, d_points_value, num_points,
        d_polygon_vertices, d_polygon_sizes, d_polygon_offsets, num_regions,
        d_count, d_sum
    )
    
    # Synchronize to ensure kernel completes
    cuda.synchronize()
    
    # Copy results back to host
    count_per_region = d_count.copy_to_host()
    sum_per_region = d_sum.copy_to_host()
    
    return count_per_region, sum_per_region

# ===================================================
# 7. WRITE OUTPUT
# ===================================================

def write_output(filename, count_per_region, sum_per_region):
    """Write aggregation results to CSV file."""
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['region', 'count', 'sum', 'average'])
        
        for r in range(len(count_per_region)):
            count = int(count_per_region[r])
            sum_val = float(sum_per_region[r])
            avg = sum_val / count if count > 0 else 0.0
            writer.writerow([r, count, sum_val, avg])

# ===================================================
# 8. MAIN
# ===================================================

def main():
    import sys
    
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <points.csv> <regions.csv> <output.csv>")
        return 1
    
    points_file = sys.argv[1]
    regions_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Load data
    print("Loading points...")
    points = load_points(points_file)
    print(f"Loaded {len(points)} points")
    
    print("Loading regions...")
    polygon_vertices, polygon_sizes, polygon_offsets = load_regions(regions_file)
    print(f"Loaded {len(polygon_sizes)} regions")
    
    # GPU Aggregation
    print("Running GPU aggregation...")
    start_time = time.time()
    
    count_per_region, sum_per_region = aggregate_gpu(
        points, polygon_vertices, polygon_sizes, polygon_offsets
    )
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0
    
    print(f"GPU Aggregation Time = {elapsed_ms:.3f} ms")
    
    # Write output
    print(f"Writing results to {output_file}...")
    write_output(output_file, count_per_region, sum_per_region)
    
    print("Done!")
    return 0

if __name__ == "__main__":
    exit(main())

