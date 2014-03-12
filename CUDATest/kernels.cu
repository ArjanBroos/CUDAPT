#include "kernels.h"
#include "sphere.h"
#include "intrec.h"
#include "lambertmaterial.h"
#include "arealight.h"
#include "plane.h"

void LaunchInitRNG(curandState* state, unsigned long seed) {
	InitRNG<<<1, 1>>>(state, seed);
}

__global__ void InitRNG(curandState* state, unsigned long seed) {
	curand_init(seed, 0, 0, state);
}

void LaunchInitScene(Scene** pScene) {
	InitScene<<<1,1>>>(pScene);
}

__global__ void InitScene(Scene** pScene) {
	*pScene = new Scene();
	Scene* scene = *pScene;

	Sphere*				sphereShape1	= new Sphere(Point(0.0f, 0.4f, 0.0f), 2.f);
	LambertMaterial*	sphereMat1		= new LambertMaterial(Color(1.f, 0.f, 0.f), 1.f);
	scene->AddPrimitive(new Primitive(sphereShape1, sphereMat1));
	Sphere*				sphereShape2	= new Sphere(Point(3.f, 0.2f, 0.f), 1.f);
	LambertMaterial*	sphereMat2		= new LambertMaterial(Color(0.f, 0.f, 1.f), 1.f);
	scene->AddPrimitive(new Primitive(sphereShape2, sphereMat2));
	Plane*				planeShape1		= new Plane(Point(), Vector(0.f, 1.f, 0.f));
	LambertMaterial*	planeMat1		= new LambertMaterial(Color(0.f, 1.f, 0.f), 1.f);
	scene->AddPrimitive(new Primitive(planeShape1, planeMat1));
	Sphere*				lightShape1		= new Sphere(Point(-1.f, 2.f, 1.f), 1.5f);
	scene->AddLight(new AreaLight(lightShape1));
}

void LaunchTraceRays(const Camera* cam, const Scene* scene, Color* result, curandState* rng, unsigned width, unsigned height, unsigned tileSize) {
	dim3 grid(width / tileSize, height / tileSize);
	dim3 block(tileSize, tileSize);
	TraceRays<<<grid, block>>>(cam, scene, result, rng, width);
}

__global__ void TraceRays(const Camera* cam, const Scene* scene, Color* result, curandState* rng, unsigned width) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned i = y * width + x;
	const Ray ray = cam->GetRay(x, y);

	result[i] = Color();

	IntRec intRec;
	if (scene->Intersect(ray, intRec)) {
		if (intRec.prim)
			result[i] = intRec.prim->GetMaterial()->GetColor();
		if (intRec.light)
			result[i] = intRec.light->Le();
	}
}

void LaunchConvert(const Color* result, unsigned char* pixelData, unsigned width, unsigned height, unsigned tileSize) {
	dim3 grid(width / tileSize, height / tileSize);
	dim3 block(tileSize, tileSize);
	Convert<<<grid, block>>>(result, pixelData, width);
}

__global__ void Convert(const Color* result, unsigned char* pixelData, unsigned width) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned i = y * width + x;
	const unsigned pdi = i*4;

	pixelData[pdi]		= (unsigned char)(result[i].r * 255);
	pixelData[pdi+1]	= (unsigned char)(result[i].g * 255);
	pixelData[pdi+2]	= (unsigned char)(result[i].b * 255);
	pixelData[pdi+3]	= 255;
}

void LaunchDestroyScene(Scene* scene) {
	DestroyScene<<<1,1>>>(scene);
}

__global__ void DestroyScene(Scene* scene) {
	delete scene;
	scene = NULL;
}

