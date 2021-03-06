#ifndef KERNELS_H
#define KERNELS_H

#include "camera.h"
#include "cuda_inc.h"
#include "color.h"
#include "sphere.h"
#include "scene.h"
#include "curand_kernel.h"
#include "builder.h"
#include <string>
#define	 MAX_DEPTH 4

//void LaunchCreateRoomScene2(Scene* scene, curandState* rng);
__device__ void CreateRoomScene2(Scene* scene, curandState* rng);
__device__ void PreAddBlock(Point loc, Scene* scene, int type);
void LaunchInitRNG(curandState* state, unsigned long seed, unsigned width, unsigned height, unsigned tileSize);
__global__ void InitRNG(curandState* state, unsigned long seed, unsigned width);

void LaunchInitScene(Scene** scene, curandState* rng);
__global__ void InitScene(Scene** scene, curandState* rng);
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

void LaunchSaveBlocks(Scene* scene);
bool checkOverwrite(std::string worldName);
void writeWorldFile(std::string worldName, std::stringstream &contents);
__global__ void SaveBlock(Scene* scene, Node* nextNode, Point* loc, Color* col, float* albedo, float* intensity, MaterialType* mat, ShapeType* shape, ObjectType* type);
__global__ void InitSaveBlocks(Scene* scene, Node* nextNode);
int LaunchCountObjects(Scene* scene);
__global__ void NumberOfBlocks(Scene* scene, int* nObjects);

void LaunchLoadBlocks(Scene* scene);
void LaunchEmptyScene(Scene* scene);
__global__ void EmptyScene(Scene* scene);
void LaunchLoadBlock(Scene* scene, Point loc, Color col, float albedo, float intensity, MaterialType mat, ShapeType shape, ObjectType type);
__global__ void LoadBlock(Scene* scene, Point loc, Color col, float albedo, float intensity, MaterialType mat, ShapeType shape, ObjectType type);

void LaunchBuilderNextBuildType(Builder* builder);
__global__ void BuilderNextBuildType(Builder* builder);
void LaunchBuilderNextShapeType(Builder* builder);
__global__ void BuilderNextShapeType(Builder* builder);
void LaunchBuilderNextMaterialType(Builder* builder);
__global__ void BuilderNextMaterialType(Builder* builder);
void LaunchBuilderSetPresetColor(Builder* builder, unsigned index);
__global__ void BuilderSetPresetColor(Builder* builder, unsigned index);
void LaunchBuilderIncrease(Builder* builder);
__global__ void BuilderIncrease(Builder* builder);
void LaunchBuilderDecrease(Builder* builder);
__global__ void BuilderDecrease(Builder* builder);

void LaunchIncreaseDayLight(Scene* scene);
__global__ void IncreaseDayLight(Scene* scene);
void LaunchDecreaseDayLight(Scene* scene);
__global__ void DecreaseDayLight(Scene* scene);

void LaunchRepositionCamera(Camera* cam);
__global__ void RepositionCamera(Camera* cam);
void LaunchSetFocalPoint(Camera* cam, const Scene* scene);
__global__ void SetFocalPoint(Camera* cam, const Scene* scene);

void LaunchConvert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width, unsigned height, unsigned tileSize);
__global__ void Convert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width);
void LaunchDestroyScene(Scene* scene);
__global__ void DestroyScene(Scene* scene);

__device__ unsigned char Clamp255(float s);

#endif
