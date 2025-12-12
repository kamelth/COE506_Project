#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

// ===================================================
// 1. DATA STRUCTURES
// ===================================================

struct Point {
    float lat;
    float lon;
    float value;
};

struct Polygon {
    int n;             // number of vertices
    float* px;         // x coordinates (device pointer)
    float* py;         // y coordinates (device pointer)
};

// Host-only polygon structure
struct RegionHost {
    int n;
    thrust::host_vector<float> px;
    thrust::host_vector<float> py;
};

// ===================================================
// 2. LOAD POINT DATA FROM CSV
// ===================================================

thrust::host_vector<Point> loadPoints(const char* filename, int* outCount)
{
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: could not open %s\n", filename);
        exit(1);
    }

    char line[256];
    if (!fgets(line, sizeof(line), f)) { // skip header
        fprintf(stderr, "Error reading header of file %s\n", filename);
        exit(1);
    }

    thrust::host_vector<Point> points;
    points.reserve(1000);

    while (fgets(line, sizeof(line), f)) {
        Point p;
        if (sscanf(line, "%f,%f,%f", &p.lat, &p.lon, &p.value) == 3) {
            points.push_back(p);
        }
    }

    fclose(f);
    *outCount = points.size();
    return points;
}

// ===================================================
// 3. LOAD REGIONS FROM CSV
// ===================================================

thrust::host_vector<RegionHost> loadRegions(const char* filename, int* outCount)
{
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: could not open %s\n", filename);
        exit(1);
    }

    char line[256];
    if (!fgets(line, sizeof(line), f)) { // skip header
        fprintf(stderr, "Error reading header of file %s\n", filename);
        exit(1);
    }

    thrust::host_vector<RegionHost> regions;
    regions.reserve(50);

    int currentRegion = -1;
    RegionHost current;
    current.n = 0;

    while (fgets(line, sizeof(line), f)) {
        int regionId;
        float x, y;

        if (sscanf(line, "%d,%f,%f", &regionId, &x, &y) != 3)
            continue;

        if (regionId != currentRegion) {
            if (currentRegion >= 0 && current.n > 0)
                regions.push_back(current);

            currentRegion = regionId;
            current.n = 0;
            current.px.clear();
            current.py.clear();
            current.px.reserve(10);
            current.py.reserve(10);
        }

        current.px.push_back(x);
        current.py.push_back(y);
        current.n++;
    }

    if (currentRegion >= 0 && current.n > 0)
        regions.push_back(current);

    fclose(f);
    *outCount = regions.size();
    return regions;
}

// ===================================================
// 4. POINT-IN-POLYGON (Ray Casting) - DEVICE FUNCTION
// ===================================================

__device__ int pointInPolygon(float x, float y, const float* polygon_vertices,
                              int polygon_size, int polygon_offset)
{
    int inside = 0;
    int n = polygon_size;

    for (int i = 0; i < n; i++) {
        int j = (i + n - 1) % n;
        
        int idx_i = polygon_offset + i * 2;
        int idx_j = polygon_offset + j * 2;
        
        float xi = polygon_vertices[idx_i];
        float yi = polygon_vertices[idx_i + 1];
        float xj = polygon_vertices[idx_j];
        float yj = polygon_vertices[idx_j + 1];

        int intersect = ((yi > y) != (yj > y)) &&
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12f) + xi);

        if (intersect) inside = !inside;
    }

    return inside;
}

// ===================================================
// 5. GPU CONFIGURATION FOR T4 GPU
// ===================================================

// T4 GPU specifications:
// - Compute Capability: 7.5
// - Number of SMs: 40
// - Max threads per block: 1024
// - Optimal threads per block: 256 (good balance of occupancy and register usage)
#define THREADS_PER_BLOCK 256

// ===================================================
// 6. CUDA KERNEL FOR AGGREGATION
// ===================================================

__global__ void aggregateKernel(const float* points_lon, const float* points_lat, 
                                const float* points_value, int N,
                                const float* polygon_vertices,
                                const int* polygon_sizes, const int* polygon_offsets,
                                int R, int* countPerRegion, float* sumPerRegion)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    // Coalesced memory read: consecutive threads read consecutive memory locations
    float x = points_lon[idx];
    float y = points_lat[idx];
    float value = points_value[idx];
    
    for (int r = 0; r < R; r++) {
        int polygon_size = polygon_sizes[r];
        int polygon_offset = polygon_offsets[r];
        
        if (pointInPolygon(x, y, polygon_vertices, polygon_size, polygon_offset)) {
            atomicAdd(&countPerRegion[r], 1);
            atomicAdd(&sumPerRegion[r], value);
            break;
        }
    }
}

// ===================================================
// 7. AGGREGATION WITH CUDA
// ===================================================

void aggregateCUDA(const thrust::host_vector<Point>& h_points, int N,
                   const thrust::host_vector<RegionHost>& h_regions_host, int R,
                   thrust::host_vector<int>& h_countPerRegion,
                   thrust::host_vector<float>& h_sumPerRegion)
{
    // Separate point arrays for coalesced memory access (Structure of Arrays)
    thrust::host_vector<float> h_points_lat(N);
    thrust::host_vector<float> h_points_lon(N);
    thrust::host_vector<float> h_points_value(N);
    
    for (int i = 0; i < N; i++) {
        h_points_lat[i] = h_points[i].lat;
        h_points_lon[i] = h_points[i].lon;
        h_points_value[i] = h_points[i].value;
    }
    
    // Transfer point data to device (coalesced layout)
    thrust::device_vector<float> d_points_lat = h_points_lat;
    thrust::device_vector<float> d_points_lon = h_points_lon;
    thrust::device_vector<float> d_points_value = h_points_value;
    
    // Flatten polygon vertices into single array [x0, y0, x1, y1, ...]
    int total_vertices = 0;
    for (int r = 0; r < R; r++) {
        total_vertices += h_regions_host[r].n;
    }
    
    thrust::host_vector<float> h_polygon_vertices(total_vertices * 2);
    thrust::host_vector<int> h_polygon_sizes(R);
    thrust::host_vector<int> h_polygon_offsets(R);
    
    int offset = 0;
    for (int r = 0; r < R; r++) {
        h_polygon_sizes[r] = h_regions_host[r].n;
        h_polygon_offsets[r] = offset;
        
        // Flatten: [x0, y0, x1, y1, ...] for coalesced access
        for (int v = 0; v < h_regions_host[r].n; v++) {
            h_polygon_vertices[offset + v * 2] = h_regions_host[r].px[v];
            h_polygon_vertices[offset + v * 2 + 1] = h_regions_host[r].py[v];
        }
        
        offset += h_regions_host[r].n * 2;
    }
    
    // Transfer polygon data to device
    thrust::device_vector<float> d_polygon_vertices = h_polygon_vertices;
    thrust::device_vector<int> d_polygon_sizes = h_polygon_sizes;
    thrust::device_vector<int> d_polygon_offsets = h_polygon_offsets;
    
    // Output arrays
    thrust::device_vector<int> d_countPerRegion(R, 0);
    thrust::device_vector<float> d_sumPerRegion(R, 0.0f);
    
    // Get raw pointers
    const float* d_points_lat_ptr = thrust::raw_pointer_cast(d_points_lat.data());
    const float* d_points_lon_ptr = thrust::raw_pointer_cast(d_points_lon.data());
    const float* d_points_value_ptr = thrust::raw_pointer_cast(d_points_value.data());
    const float* d_polygon_vertices_ptr = thrust::raw_pointer_cast(d_polygon_vertices.data());
    const int* d_polygon_sizes_ptr = thrust::raw_pointer_cast(d_polygon_sizes.data());
    const int* d_polygon_offsets_ptr = thrust::raw_pointer_cast(d_polygon_offsets.data());
    int* d_count_ptr = thrust::raw_pointer_cast(d_countPerRegion.data());
    float* d_sum_ptr = thrust::raw_pointer_cast(d_sumPerRegion.data());

    // Calculate grid dimensions for T4 GPU
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel with unified configuration
    aggregateKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_points_lon_ptr, d_points_lat_ptr, d_points_value_ptr, N,
        d_polygon_vertices_ptr, d_polygon_sizes_ptr, d_polygon_offsets_ptr, R,
        d_count_ptr, d_sum_ptr
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    h_countPerRegion = d_countPerRegion;
    h_sumPerRegion = d_sumPerRegion;
}

// ===================================================
// 8. WRITE OUTPUT
// ===================================================

void writeOutput(const char* filename,
                 const thrust::host_vector<int>& count,
                 const thrust::host_vector<float>& sum,
                 int R)
{
    FILE* f = fopen(filename, "w");
    fprintf(f, "region,count,sum,average\n");

    for (int r = 0; r < R; r++) {
        float avg = (count[r] > 0) ? sum[r] / count[r] : 0.0f;
        fprintf(f, "%d,%d,%f,%f\n", r, count[r], sum[r], avg);
    }

    fclose(f);
}

// ===================================================
// 9. MAIN
// ===================================================

int main(int argc, char* argv[])
{
    if (argc != 4) {
        printf("Usage: %s <points_csv> <regions_csv> <output_csv>\n", argv[0]);
        return 1;
    }

    const char* pointsFile = argv[1];
    const char* regionsFile = argv[2];
    const char* outputFile = argv[3];

    int N, R;

    auto t_start_load = std::chrono::high_resolution_clock::now();
    thrust::host_vector<Point> points = loadPoints(pointsFile, &N);
    thrust::host_vector<RegionHost> regions = loadRegions(regionsFile, &R);
    auto t_end_load = std::chrono::high_resolution_clock::now();

    printf("Loaded %d points and %d regions\n", N, R);

    thrust::host_vector<int> countPerRegion(R);
    thrust::host_vector<float> sumPerRegion(R);

    auto t_start_compute = std::chrono::high_resolution_clock::now();
    aggregateCUDA(points, N, regions, R, countPerRegion, sumPerRegion);
    auto t_end_compute = std::chrono::high_resolution_clock::now();

    double ms_load = std::chrono::duration<double, std::milli>(t_end_load - t_start_load).count();
    double ms_compute = std::chrono::duration<double, std::milli>(t_end_compute - t_start_compute).count();

    printf("Data Loading Time = %.3f ms\n", ms_load);
    printf("CUDA Aggregation Time = %.3f ms\n", ms_compute);
    printf("Total Time = %.3f ms\n", ms_load + ms_compute);

    writeOutput(outputFile, countPerRegion, sumPerRegion, R);

    return 0;
}

