
/*
 * GPU-Accelerated Point-in-Polygon Aggregation using OpenACC
 *
 * This implementation uses the following optimization techniques:
 * 1. OpenACC Parallel Loop - Parallelizes point processing across GPU threads
 * 2. Memory Coalescing - Structures data for efficient GPU memory access
 * 3. Atomic Operations - Thread-safe reductions for count and sum per region
 * 4. Flattened Data Structures - Better memory access patterns on GPU
 * 5. Device Memory Management - Minimizes CPU-GPU transfers with data directives
 *
 * Compile with: pgcc -acc -Minfo=accel -o aggregate aggregate.c -lm
 * Or with: nvc -acc -Minfo=accel -o aggregate aggregate.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ===================================================
// 1. GPU CONFIGURATION FOR T4 GPU
// ===================================================

// T4 GPU specifications:
// - Compute Capability: 7.5
// - Number of SMs: 40
// - Max threads per block: 1024
// - Optimal threads per block: 256 (good balance of occupancy and register usage)
// Note: OpenACC will use this as a hint via vector_length directive
#define THREADS_PER_BLOCK 256

// ===================================================
// 2. DATA STRUCTURES
// ===================================================

typedef struct {
    float lat;
    float lon;
    float value;
} Point;

typedef struct {
    float *vertices;      // Flattened: [x0, y0, x1, y1, ...]
    int *sizes;           // Number of vertices per polygon
    int *offsets;         // Starting index in vertices array
    int num_regions;
} Regions;

// ===================================================
// 3. LOAD POINT DATA FROM CSV
// ===================================================

Point* load_points(const char *filename, int *num_points) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening points file: %s\n", filename);
        return NULL;
    }
    
    // Skip header
    char line[1024];
    fgets(line, sizeof(line), f);
    
    // Count lines first
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        count++;
    }
    
    // Allocate memory
    Point *points = (Point*)malloc(count * sizeof(Point));
    if (!points) {
        fclose(f);
        return NULL;
    }
    
    // Rewind and read data
    rewind(f);
    fgets(line, sizeof(line), f); // Skip header again
    
    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < count) {
        float lat, lon, val;
        if (sscanf(line, "%f,%f,%f", &lat, &lon, &val) == 3) {
            points[idx].lat = lat;
            points[idx].lon = lon;
            points[idx].value = val;
            idx++;
        }
    }
    
    fclose(f);
    *num_points = idx;
    return points;
}

// ===================================================
// 4. LOAD REGIONS FROM CSV
// ===================================================

Regions* load_regions(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening regions file: %s\n", filename);
        return NULL;
    }
    
    // Skip header
    char line[1024];
    fgets(line, sizeof(line), f);
    
    // First pass: count regions and vertices
    int max_region_id = -1;
    int total_vertices = 0;
    
    while (fgets(line, sizeof(line), f)) {
        int region_id;
        float x, y;
        if (sscanf(line, "%d,%f,%f", &region_id, &x, &y) == 3) {
            if (region_id > max_region_id) {
                max_region_id = region_id;
            }
            total_vertices++;
        }
    }
    
    int num_regions = max_region_id + 1;
    
    // Allocate temporary storage for each region
    int *vertex_counts = (int*)calloc(num_regions, sizeof(int));
    float **temp_vertices = (float**)calloc(num_regions, sizeof(float*));
    
    for (int i = 0; i < num_regions; i++) {
        temp_vertices[i] = (float*)malloc(total_vertices * 2 * sizeof(float));
    }
    
    // Second pass: read vertices
    rewind(f);
    fgets(line, sizeof(line), f); // Skip header
    
    while (fgets(line, sizeof(line), f)) {
        int region_id;
        float x, y;
        if (sscanf(line, "%d,%f,%f", &region_id, &x, &y) == 3) {
            int idx = vertex_counts[region_id];
            temp_vertices[region_id][idx * 2] = x;
            temp_vertices[region_id][idx * 2 + 1] = y;
            vertex_counts[region_id]++;
        }
    }
    
    fclose(f);
    
    // Create flattened structure
    Regions *regions = (Regions*)malloc(sizeof(Regions));
    regions->num_regions = num_regions;
    regions->sizes = (int*)malloc(num_regions * sizeof(int));
    regions->offsets = (int*)malloc(num_regions * sizeof(int));
    regions->vertices = (float*)malloc(total_vertices * 2 * sizeof(float));
    
    int offset = 0;
    for (int r = 0; r < num_regions; r++) {
        regions->sizes[r] = vertex_counts[r];
        regions->offsets[r] = offset;
        
        // Copy vertices to flattened array
        memcpy(&regions->vertices[offset], temp_vertices[r], 
               vertex_counts[r] * 2 * sizeof(float));
        
        offset += vertex_counts[r] * 2;
        free(temp_vertices[r]);
    }
    
    free(temp_vertices);
    free(vertex_counts);
    
    return regions;
}

// ===================================================
// 5. POINT-IN-POLYGON (Device Function)
// ===================================================

#pragma acc routine seq
int point_in_polygon(float x, float y, float *polygon_vertices, 
                     int polygon_size, int polygon_offset) {
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
        
        if (intersect) {
            inside = 1 - inside;
        }
    }
    
    return inside;
}

// ===================================================
// 6. GPU AGGREGATION USING OpenACC
// ===================================================

void aggregate_gpu(Point *points, int num_points, Regions *regions,
                   int *count_per_region, float *sum_per_region) {
    
    int num_regions = regions->num_regions;
    
    // Separate arrays for coalesced access
    float *points_lat = (float*)malloc(num_points * sizeof(float));
    float *points_lon = (float*)malloc(num_points * sizeof(float));
    float *points_value = (float*)malloc(num_points * sizeof(float));
    
    for (int i = 0; i < num_points; i++) {
        points_lat[i] = points[i].lat;
        points_lon[i] = points[i].lon;
        points_value[i] = points[i].value;
    }
    
    // Initialize output arrays
    for (int r = 0; r < num_regions; r++) {
        count_per_region[r] = 0;
        sum_per_region[r] = 0.0f;
    }
    
    // Get pointers for cleaner code
    float *polygon_vertices = regions->vertices;
    int *polygon_sizes = regions->sizes;
    int *polygon_offsets = regions->offsets;
    
    // OpenACC parallel region
    // Copy data to device and keep it there during computation
    #pragma acc data copyin(points_lat[0:num_points], \
                            points_lon[0:num_points], \
                            points_value[0:num_points], \
                            polygon_vertices[0:regions->offsets[num_regions-1] + regions->sizes[num_regions-1]*2], \
                            polygon_sizes[0:num_regions], \
                            polygon_offsets[0:num_regions]) \
                     copy(count_per_region[0:num_regions], \
                          sum_per_region[0:num_regions])
    {
        // Parallel loop over all points
        // Unified configuration for T4 GPU: vector_length(256) hints the compiler
        // to use 256 threads per block (gang) for optimal T4 performance
        #pragma acc parallel loop gang vector vector_length(THREADS_PER_BLOCK)
        for (int idx = 0; idx < num_points; idx++) {
            float x = points_lon[idx];
            float y = points_lat[idx];
            float value = points_value[idx];
            
            // Check each region sequentially (per thread)
            for (int r = 0; r < num_regions; r++) {
                int polygon_size = polygon_sizes[r];
                int polygon_offset = polygon_offsets[r];
                
                if (point_in_polygon(x, y, polygon_vertices, 
                                    polygon_size, polygon_offset)) {
                    // Atomic operations for thread-safe updates
                    #pragma acc atomic update
                    count_per_region[r] += 1;
                    
                    #pragma acc atomic update
                    sum_per_region[r] += value;
                    
                    break; // Early exit
                }
            }
        }
    }
    
    // Free temporary arrays
    free(points_lat);
    free(points_lon);
    free(points_value);
}

// ===================================================
// 7. WRITE OUTPUT
// ===================================================

void write_output(const char *filename, int *count_per_region, 
                  float *sum_per_region, int num_regions) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening output file: %s\n", filename);
        return;
    }
    
    fprintf(f, "region,count,sum,average\n");
    
    for (int r = 0; r < num_regions; r++) {
        int count = count_per_region[r];
        float sum_val = sum_per_region[r];
        float avg = (count > 0) ? (sum_val / count) : 0.0f;
        
        fprintf(f, "%d,%d,%.6f,%.6f\n", r, count, sum_val, avg);
    }
    
    fclose(f);
}

// ===================================================
// 8. FREE MEMORY
// ===================================================

void free_regions(Regions *regions) {
    if (regions) {
        free(regions->vertices);
        free(regions->sizes);
        free(regions->offsets);
        free(regions);
    }
}

// ===================================================
// 9. MAIN
// ===================================================

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <points.csv> <regions.csv> <output.csv>\n", 
                argv[0]);
        return 1;
    }
    
    const char *points_file = argv[1];
    const char *regions_file = argv[2];
    const char *output_file = argv[3];
    
    // Load data
    printf("Loading points...\n");
    int num_points;
    Point *points = load_points(points_file, &num_points);
    if (!points) {
        fprintf(stderr, "Failed to load points\n");
        return 1;
    }
    printf("Loaded %d points\n", num_points);
    
    printf("Loading regions...\n");
    Regions *regions = load_regions(regions_file);
    if (!regions) {
        fprintf(stderr, "Failed to load regions\n");
        free(points);
        return 1;
    }
    printf("Loaded %d regions\n", regions->num_regions);
    
    // Allocate output arrays
    int *count_per_region = (int*)malloc(regions->num_regions * sizeof(int));
    float *sum_per_region = (float*)malloc(regions->num_regions * sizeof(float));
    
    // GPU Aggregation
    printf("Running GPU aggregation...\n");
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    aggregate_gpu(points, num_points, regions, count_per_region, sum_per_region);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("GPU Aggregation Time = %.3f ms\n", elapsed_ms);
    
    // Write output
    printf("Writing results to %s...\n", output_file);
    write_output(output_file, count_per_region, sum_per_region, regions->num_regions);
    
    // Cleanup
    free(points);
    free_regions(regions);
    free(count_per_region);
    free(sum_per_region);
    
    printf("Done!\n");
    return 0;
}

