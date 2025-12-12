#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <nvToolsExt.h>   // <-- ADD NVTX

// ===================================================
// 1. DATA STRUCTURES
// ===================================================

typedef struct {
    float lat;
    float lon;
    float value;
} Point;

typedef struct {
    int n;
    float* px;
    float* py;
} Polygon;

// ===================================================
// 2. LOAD POINT DATA FROM CSV
// ===================================================

Point* loadPoints(const char* filename, int* outCount)
{
    nvtxRangePushA("loadPoints");

    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: could not open %s\n", filename);
        exit(1);
    }

    char line[256];
    fgets(line, sizeof(line), f); // skip header

    int capacity = 1000;
    int count = 0;
    Point* points = (Point*)malloc(capacity * sizeof(Point));

    while (fgets(line, sizeof(line), f)) {
        if (count == capacity) {
            capacity *= 2;
            points = (Point*)realloc(points, capacity * sizeof(Point));
        }

        float lat, lon, val;
        sscanf(line, "%f,%f,%f", &lat, &lon, &val);

        points[count].lat = lat;
        points[count].lon = lon;
        points[count].value = val;

        count++;
    }

    fclose(f);
    *outCount = count;

    nvtxRangePop();
    return points;
}

// ===================================================
// 3. LOAD REGIONS FROM CSV
// ===================================================

Polygon* loadRegions(const char* filename, int* outCount)
{
    nvtxRangePushA("loadRegions");

    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: could not open %s\n", filename);
        exit(1);
    }

    char line[256];
    fgets(line, sizeof(line), f); // skip header

    int capacity = 50;
    int count = 0;
    Polygon* regions = (Polygon*)malloc(capacity * sizeof(Polygon));

    int currentRegion = -1;
    int vertexCount = 0;
    int vertexCapacity = 0;
    float* tempX = NULL;
    float* tempY = NULL;

    while (fgets(line, sizeof(line), f)) {
        int regionId;
        float x, y;

        if (sscanf(line, "%d,%f,%f", &regionId, &x, &y) != 3) {
            continue;
        }

        if (regionId != currentRegion) {
            if (currentRegion >= 0 && vertexCount > 0) {
                if (count == capacity) {
                    capacity *= 2;
                    regions = (Polygon*)realloc(regions, capacity * sizeof(Polygon));
                }
                regions[count].n = vertexCount;
                regions[count].px = tempX;
                regions[count].py = tempY;
                count++;
            }

            currentRegion = regionId;
            vertexCount = 0;
            vertexCapacity = 10;
            tempX = (float*)malloc(vertexCapacity * sizeof(float));
            tempY = (float*)malloc(vertexCapacity * sizeof(float));
        }

        if (vertexCount == vertexCapacity) {
            vertexCapacity *= 2;
            tempX = (float*)realloc(tempX, vertexCapacity * sizeof(float));
            tempY = (float*)realloc(tempY, vertexCapacity * sizeof(float));
        }

        tempX[vertexCount] = x;
        tempY[vertexCount] = y;
        vertexCount++;
    }

    if (currentRegion >= 0 && vertexCount > 0) {
        if (count == capacity) {
            capacity *= 2;
            regions = (Polygon*)realloc(regions, capacity * sizeof(Polygon));
        }
        regions[count].n = vertexCount;
        regions[count].px = tempX;
        regions[count].py = tempY;
        count++;
    }

    fclose(f);
    *outCount = count;

    nvtxRangePop();
    return regions;
}

// ===================================================
// 4. POINT-IN-POLYGON
// ===================================================

int pointInPolygon(float x, float y, const Polygon* poly)
{
    // Optional: annotate small internal function
    // nvtxRangePushA("pointInPolygon");

    int inside = 0;
    int n = poly->n;

    for (int i = 0, j = n - 1; i < n; j = i++) {
        float xi = poly->px[i], yi = poly->py[i];
        float xj = poly->px[j], yj = poly->py[j];

        int intersect =
            ((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12f) + xi);

        if (intersect)
            inside = !inside;
    }

    // nvtxRangePop();
    return inside;
}

// ===================================================
// 5. AGGREGATION
// ===================================================

void aggregateCPU(Point* points, int N,
                  Polygon* regions, int R,
                  int* countPerRegion, float* sumPerRegion)
{
    nvtxRangePushA("aggregateCPU");

    for (int i = 0; i < N; i++) {
        float x = points[i].lon;
        float y = points[i].lat;

        for (int r = 0; r < R; r++) {
            if (pointInPolygon(x, y, &regions[r])) {
                countPerRegion[r] += 1;
                sumPerRegion[r] += points[i].value;
                break;
            }
        }
    }

    nvtxRangePop();
}

// ===================================================
// 6. WRITE OUTPUT
// ===================================================

void writeOutput(const char* filename,
                 int* count, float* sum, int R)
{
    nvtxRangePushA("writeOutput");

    FILE* f = fopen(filename, "w");
    fprintf(f, "region,count,sum,average\n");

    for (int r = 0; r < R; r++) {
        float avg = (count[r] > 0) ? sum[r] / count[r] : 0.0f;
        fprintf(f, "%d,%d,%f,%f\n", r, count[r], sum[r], avg);
    }

    fclose(f);

    nvtxRangePop();
}

// ===================================================
// 7. MAIN
// ===================================================

int main(int argc, char** argv)
{
    nvtxRangePushA("main");

    if (argc < 4) {
        printf("Usage: %s <points.csv> <regions.csv> <output.csv>\n", argv[0]);
        return 1;
    }

    const char* pointsFile  = argv[1];
    const char* regionsFile = argv[2];
    const char* outputFile  = argv[3];

    int N, R;

    nvtxRangePushA("load_data");
    Point* points = loadPoints(pointsFile, &N);
    Polygon* regions = loadRegions(regionsFile, &R);
    nvtxRangePop(); // load_data

    int* countPerRegion = (int*)calloc(R, sizeof(int));
    float* sumPerRegion = (float*)calloc(R, sizeof(float));

    nvtxRangePushA("CPU_Aggregation");
    clock_t start = clock();
    aggregateCPU(points, N, regions, R, countPerRegion, sumPerRegion);
    clock_t end = clock();
    nvtxRangePop();

    double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    printf("CPU Aggregation Time = %.3f ms\n", ms);

    nvtxRangePushA("write_output");
    writeOutput(outputFile, countPerRegion, sumPerRegion, R);
    nvtxRangePop();

    nvtxRangePushA("cleanup");
    for (int r = 0; r < R; r++) {
        free(regions[r].px);
        free(regions[r].py);
    }
    free(regions);
    free(points);
    free(countPerRegion);
    free(sumPerRegion);
    nvtxRangePop();

    nvtxRangePop(); // main

    return 0;
}


