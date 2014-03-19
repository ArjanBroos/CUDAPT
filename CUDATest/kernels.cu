#include "kernels.h"
#include "sphere.h"
#include "intrec.h"
#include "lambertmaterial.h"
#include "arealight.h"
#include "plane.h"
#include "mirrormaterial.h"
#include "box.h"

void LaunchInitRNG(curandState* state, unsigned long seed, unsigned width, unsigned height, unsigned tileSize) {
	dim3 grid(width / tileSize, height / tileSize);
	dim3 block(tileSize, tileSize);
	InitRNG<<<grid, block>>>(state, seed, width);
}

__global__ void InitRNG(curandState* state, unsigned long seed, unsigned width) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned i = y * width + x;
	curand_init(seed - i, 0, 0, &state[i]);
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
	MirrorMaterial*		sphereMat2		= new MirrorMaterial(Color(1.f, 1.f, 1.f), 0.8f);
	scene->AddPrimitive(new Primitive(sphereShape2, sphereMat2));
	Sphere*				sphereShape3	= new Sphere(Point(-2.f, 8.f, -20.f), 6.f);
	MirrorMaterial*		sphereMat3		= new MirrorMaterial(Color(1.f, 1.f, 1.f), 0.9f);
	scene->AddPrimitive(new Primitive(sphereShape3, sphereMat3));

	Box*				boxShape1		= new Box(Point(-1.5f, 0.5f, 3.f), 1.f);
	LambertMaterial*	boxMat1			= new LambertMaterial();
	scene->AddPrimitive(new Primitive(boxShape1, boxMat1));

	Plane*				planeShape1		= new Plane(Point(), Vector(0.f, 1.f, 0.f));
	LambertMaterial*	planeMat1		= new LambertMaterial(Color(1.f, 1.f, 0.3f), 1.f);
	scene->AddPrimitive(new Primitive(planeShape1, planeMat1));

	Sphere*				lightShape1		= new Sphere(Point(-2.f, 3.f, 1.f), 1.5f);
	scene->AddLight(new AreaLight(lightShape1));
	Sphere*				lightShape2		= new Sphere(Point(5.f, 1.f, -3.f), 0.8f);
	scene->AddLight(new AreaLight(lightShape2, Color(1.f, 0.5f, 0.5f), 1.5f));
	Box*				lightShape3		= new Box(Point(3.f, 0.f, 5.f), 0.8f);
	scene->AddLight(new AreaLight(lightShape3, Color(1.f, 1.f, 1.f), 2.f));
}

void LaunchInitResult(Color* result, unsigned width, unsigned height, unsigned tileSize) {
	dim3 grid(width / tileSize, height / tileSize);
	dim3 block(tileSize, tileSize);
	InitResult<<<grid, block>>>(result, width);
}

__global__ void InitResult(Color* result, unsigned width) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned i = y * width + x;
	result[i] = Color();
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

	Ray ray = cam->GetJitteredRay(x, y, rng);
	const unsigned maxDepth = 4;
	IntRec intRec;
	Color color(1.f, 1.f, 1.f);

	for (unsigned depth = 0; depth <= maxDepth; depth++) {
		if (depth == maxDepth) {
			color *= Color();
			break;
		}

		if (!scene->Intersect(ray, intRec)) {
			color *= Color(0.2f, 0.2f, 0.3f);
			break;
		}

		if (intRec.light) {
			color *= intRec.light->Le();
			break;
		}

		const Material* mat = intRec.prim->GetMaterial();
		const Shape*	shape = intRec.prim->GetShape();
		const Point		p = ray(intRec.t);
		const Vector	n = shape->GetNormal(p);

		const Vector in = ray.d;
		const Vector out = mat->GetSample(in, n, &rng[i]);
		ray = Ray(p, out);

		color *= mat->GetColor();
		color *= mat->GetMultiplier(in, out, n);
	}

	result[i] += color;
}

void LaunchConvert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width, unsigned height, unsigned tileSize) {
	dim3 grid(width / tileSize, height / tileSize);
	dim3 block(tileSize, tileSize);
	Convert<<<grid, block>>>(result, pixelData, iteration, width);
}

__global__ void Convert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned i = y * width + x;
	const unsigned pdi = i*4;

	const float r = (result[i].r / iteration) * 255;
	const float g = (result[i].g / iteration) * 255;
	const float b = (result[i].b / iteration) * 255;

	pixelData[pdi]		= Clamp255(r);
	pixelData[pdi+1]	= Clamp255(g);
	pixelData[pdi+2]	= Clamp255(b);
	pixelData[pdi+3]	= 255;
}

void LaunchDestroyScene(Scene* scene) {
	DestroyScene<<<1,1>>>(scene);
}

__global__ void DestroyScene(Scene* scene) {
	delete scene;
	scene = NULL;
}

__device__ unsigned char Clamp255(float s) {
	if (s < 0.f) s = 0.f;
	if (s > 255.f) s = 255.f;
	return (unsigned char)s;
}