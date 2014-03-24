#pragma once

#include "camera.h"
#include "cuda_inc.h"
#include "color.h"
#include "sphere.h"
#include "scene.h"
#include "curand_kernel.h"
#include "builder.h"
#define	 MAX_DEPTH 4

void LaunchInitRNG(curandState* state, unsigned long seed, unsigned width, unsigned height, unsigned tileSize);
__global__ void InitRNG(curandState* state, unsigned long seed, unsigned width);
void LaunchInitScene(Scene** scene);
__global__ void InitScene(Scene** scene);
void LaunchInitBuilder(Builder** builder);
__global__ void InitBuilder(Builder** builder);
void LaunchInitResult(Color* result, unsigned width, unsigned height, unsigned tileSize);
__global__ void InitResult(Color* result, unsigned width);
void LaunchTraceRays(const Camera* cam, const Scene* scene, Color* result, curandState* rng, unsigned width, unsigned height, unsigned tileSize);
__global__ void TraceRays(const Camera* cam, const Scene* scene, Color* result, curandState* rng, unsigned width);

void LaunchAddBlock(const Camera* cam, Scene* scene, Builder* builder);
__global__ void AddBlock(const Camera* cam, Scene* scene, Builder* builder);
void LaunchRemoveBlock(const Camera* cam, Scene* scene);
__global__ void RemoveBlock(const Camera* cam, Scene* scene);

void LaunchBuilderNextBuildType(Builder* builder);
__global__ void BuilderNextBuildType(Builder* builder);
void LaunchBuilderNextMaterialType(Builder* builder);
__global__ void BuilderNextMaterialType(Builder* builder);
void LaunchBuilderNextColor(Builder* builder);
__global__ void BuilderNextColor(Builder* builder);
void LaunchBuilderIncrease(Builder* builder);
__global__ void BuilderIncrease(Builder* builder);
void LaunchBuilderDecrease(Builder* builder);
__global__ void BuilderDecrease(Builder* builder);

void LaunchConvert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width, unsigned height, unsigned tileSize);
__global__ void Convert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width);
void LaunchDestroyScene(Scene* scene);
__global__ void DestroyScene(Scene* scene);

__device__ unsigned char Clamp255(float s);