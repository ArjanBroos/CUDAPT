#ifndef KERNELS_H
#define KERNELS_H

#include "camera.h"
#include "color.h"
#include "sphere.h"
#include "scene.h"
#include "builder.h"
#include <string>
#define	 MAX_DEPTH 4

//void LaunchCreateRoomScene2(Scene* scene, curandState* rng);
void CreateRoomScene2(Scene* scene, curandState* rng);
void PreAddBlock(Point loc, Scene* scene, int type);

void LaunchInitScene(Scene* scene);
void LaunchInitBuilder(Builder** builder);
void InitBuilder(Builder** builder);
void LaunchInitResult(Color* result, unsigned width, unsigned height);
void InitResult(Color* result, unsigned width);
void LaunchTraceRays(const Camera* cam, const Scene* scene, Color* result, unsigned width, unsigned height);
void TraceRays(const unsigned x, const unsigned y, const Camera* cam, const Scene* scene, Color* result, unsigned width);

void LaunchAddBlock(const Camera* cam, Scene* scene, Builder* builder);
void AddBlock(const Camera* cam, Scene* scene, Builder* builder);
void LaunchRemoveBlock(const Camera* cam, Scene* scene);
void RemoveBlock(const Camera* cam, Scene* scene);

void LaunchSaveBlocks(Scene* scene);
bool checkOverwrite(std::string worldName);
void writeWorldFile(std::string worldName, std::stringstream &contents);
void SaveBlock(Scene* scene, Node** nextNode, Point &loc, Color &col, float &albedo, float &intensity, MaterialType &mat, ShapeType &shape, ObjectType &type);
void InitSaveBlocks(Scene* scene, Node* nextNode);
int LaunchCountObjects(Scene* scene);
void NumberOfBlocks(Scene* scene, int* nObjects);

void LaunchLoadBlocks(Scene* scene);
void LaunchEmptyScene(Scene* scene);
void EmptyScene(Scene* scene);
void LaunchLoadBlock(Scene* scene, Point loc, Color col, float albedo, float intensity, MaterialType mat, ShapeType shape, ObjectType type);
void LoadBlock(Scene* scene, Point loc, Color col, float albedo, float intensity, MaterialType mat, ShapeType shape, ObjectType type);

void LaunchBuilderNextBuildType(Builder* builder);
void BuilderNextBuildType(Builder* builder);
void LaunchBuilderNextShapeType(Builder* builder);
void BuilderNextShapeType(Builder* builder);
void LaunchBuilderNextMaterialType(Builder* builder);
void BuilderNextMaterialType(Builder* builder);
void LaunchBuilderSetPresetColor(Builder* builder, unsigned index);
void BuilderSetPresetColor(Builder* builder, unsigned index);
void LaunchBuilderIncrease(Builder* builder);
void BuilderIncrease(Builder* builder);
void LaunchBuilderDecrease(Builder* builder);
void BuilderDecrease(Builder* builder);

void LaunchIncreaseDayLight(Scene* scene);
void IncreaseDayLight(Scene* scene);
void LaunchDecreaseDayLight(Scene* scene);
void DecreaseDayLight(Scene* scene);

void LaunchRepositionCamera(Camera* cam);
void RepositionCamera(Camera* cam);
void LaunchSetFocalPoint(Camera* cam, const Scene* scene);
void SetFocalPoint(Camera* cam, const Scene* scene);

void LaunchConvert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width, unsigned height);
void Convert(const unsigned x, const unsigned y, const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width);
void LaunchDestroyScene(Scene* scene);
void DestroyScene(Scene* scene);

unsigned char Clamp255(float s);

#endif
