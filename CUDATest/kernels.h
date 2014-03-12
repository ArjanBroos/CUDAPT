#pragma once

#include "camera.h"
#include "cuda_inc.h"
#include "color.h"
#include "sphere.h"
#include "scene.h"
#include "curand_kernel.h"

void LaunchInitRNG(curandState* state, unsigned long seed);
__global__ void InitRNG(curandState* state, unsigned long seed);
void LaunchInitScene(Scene** scene);
__global__ void InitScene(Scene** scene);
void LaunchTraceRays(const Camera* cam, const Scene* scene, Color* result, curandState* rng, unsigned width, unsigned height, unsigned tileSize);
__global__ void TraceRays(const Camera* cam, const Scene* scene, Color* result, curandState* rng, unsigned width);
void LaunchConvert(const Color* result, unsigned char* pixelData, unsigned width, unsigned height, unsigned tileSize);
__global__ void Convert(const Color* result, unsigned char* pixelData, unsigned width);
void LaunchDestroyScene(Scene* scene);
__global__ void DestroyScene(Scene* scene);