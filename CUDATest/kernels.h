#pragma once

#include "camera.h"
#include "cuda_inc.h"
#include "color.h"
#include "sphere.h"
#include "scene.h"
#include "curand_kernel.h"
#define	 MAX_DEPTH 4

void LaunchInitRNG(curandState* state, unsigned long seed, unsigned width, unsigned height, unsigned tileSize);
__global__ void InitRNG(curandState* state, unsigned long seed, unsigned width);
void LaunchInitScene(Scene** scene);
__global__ void InitScene(Scene** scene);
void LaunchInitResult(Color* result, unsigned width, unsigned height, unsigned tileSize);
__global__ void InitResult(Color* result, unsigned width);
void LaunchTraceRays(const Camera* cam, const Scene* scene, Color* result, curandState* rng, unsigned width, unsigned height, unsigned tileSize);
__global__ void TraceRays(const Camera* cam, const Scene* scene, Color* result, curandState* rng, unsigned width);

void LaunchChangeLight(Scene* scene);
__global__ void ChangeLight(Scene* scene);

void LaunchAddBlock(const Camera* cam, Scene* scene);
__global__ void AddBlock(const Camera* cam, Scene* scene);
void LaunchRemoveBlock(const Camera* cam, Scene* scene);
__global__ void RemoveBlock(const Camera* cam, Scene* scene);

void LaunchConvert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width, unsigned height, unsigned tileSize);
__global__ void Convert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width);
void LaunchDestroyScene(Scene* scene);
__global__ void DestroyScene(Scene* scene);

__device__ unsigned char Clamp255(float s);