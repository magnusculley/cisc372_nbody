#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "vector.h"
#include "config.h"

#define THREADS_PER_BLOCK 256
#define CUDA_CHECK(err)                                                   \
    {                                                                     \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(1);                                                      \
        }                                                                 \
    }

// Device memory pointers (external, defined in nbody.c)
extern vector3 *d_hVel, *d_hPos;
extern double *d_mass;

extern "C"
{

    // Kernel to compute pairwise accelerations
    __global__ void accumulateForces(vector3 *pos, vector3 *acc, double *mass, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;

        for (int j = 0; j < n; j++)
        {
            if (i == j)
                continue;

            vector3 distance;
            distance[0] = pos[i][0] - pos[j][0];
            distance[1] = pos[i][1] - pos[j][1];
            distance[2] = pos[i][2] - pos[j][2];

            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;

            acc[i][0] += accelmag * distance[0] / magnitude;
            acc[i][1] += accelmag * distance[1] / magnitude;
            acc[i][2] += accelmag * distance[2] / magnitude;
        }
    }

    // kernel to update velocity and position
    __global__ void moveBodies(vector3 *pos, vector3 *vel, vector3 *acc, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;

        vel[i][0] += acc[i][0] * INTERVAL;
        vel[i][1] += acc[i][1] * INTERVAL;
        vel[i][2] += acc[i][2] * INTERVAL;

        pos[i][0] += vel[i][0] * INTERVAL;
        pos[i][1] += vel[i][1] * INTERVAL;
        pos[i][2] += vel[i][2] * INTERVAL;
    }

    // initialize device memory
    void initDeviceMemory(int numObjects)
    {
        CUDA_CHECK(cudaMalloc(&d_hPos, sizeof(vector3) * numObjects));
        CUDA_CHECK(cudaMalloc(&d_hVel, sizeof(vector3) * numObjects));
        CUDA_CHECK(cudaMalloc(&d_mass, sizeof(double) * numObjects));
    }

    // free device memory
    void freeDeviceMemory()
    {
        CUDA_CHECK(cudaFree(d_hPos));
        CUDA_CHECK(cudaFree(d_hVel));
        CUDA_CHECK(cudaFree(d_mass));
    }

    // copy data to device
    void copyToDevice(vector3 *hPos, vector3 *hVel, double *hMass, int numObjects)
    {
        CUDA_CHECK(cudaMemcpy(d_hPos, hPos, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_hVel, hVel, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, hMass, sizeof(double) * numObjects, cudaMemcpyHostToDevice));
    }

    // copy data from device
    void copyFromDevice(vector3 *hPos, vector3 *hVel, int numObjects)
    {
        CUDA_CHECK(cudaMemcpy(hPos, d_hPos, sizeof(vector3) * numObjects, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hVel, d_hVel, sizeof(vector3) * numObjects, cudaMemcpyDeviceToHost));
    }

    // Compute: Updates positions and velocities based on gravity
    void compute(vector3 *hPos, vector3 *hVel, double *hMass, int numObjects)
    {
        // Allocate temporary device acceleration array
        vector3 *d_acc;
        CUDA_CHECK(cudaMalloc(&d_acc, sizeof(vector3) * numObjects));
        CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(vector3) * numObjects));

        // Copy data to device
        copyToDevice(hPos, hVel, hMass, numObjects);

        // Calculate grid/block dimensions
        int blocks = (numObjects + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Accumulate forces
        accumulateForces<<<blocks, THREADS_PER_BLOCK>>>(d_hPos, d_acc, d_mass, numObjects);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update positions and velocities
        moveBodies<<<blocks, THREADS_PER_BLOCK>>>(d_hPos, d_hVel, d_acc, numObjects);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        copyFromDevice(hPos, hVel, numObjects);

        // Free temporary acceleration array
        CUDA_CHECK(cudaFree(d_acc));
    }
} // extern "C"
