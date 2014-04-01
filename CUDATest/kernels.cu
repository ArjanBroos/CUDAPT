#include "kernels.h"
#include "sphere.h"
#include "intrec.h"
#include "lambertmaterial.h"
#include "arealight.h"
#include "plane.h"
#include "mirrormaterial.h"
#include "box.h"
#include "point.h"
#include "cuda_runtime.h"
#include "device_functions.h"

void LaunchInitRNG(curandState* state, unsigned long seed, unsigned width, unsigned height, unsigned tileSize) {
	dim3 grid(width / tileSize, height / tileSize);
	dim3 block(tileSize, tileSize);
	InitRNG<<<grid, block>>>(state, seed, width);
}

__global__ void InitRNG(curandState* state, unsigned long seed, unsigned width) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned i = y * width + x;
	curand_init(seed - i*x, 0, 0, &state[i]);
}

void LaunchInitScene(Scene** pScene) {
	InitScene<<<1,1>>>(pScene);
}

__device__ void PreAddBlock(Point* loc, Scene* scene, int type) {
	if(type == 0) {
		Box*				boxShape	= new Box(*loc);
		LambertMaterial*	boxMat		= new LambertMaterial(Color(1.f, 0.f, 0.f), 1.f);
		Primitive*			boxPrim		= new Primitive(boxShape, boxMat, &boxShape->bounds[0]);
		boxPrim->type = PRIMITIVE;
		scene->AddObject(boxPrim);
	}
	if(type == 1) {
		Box*				boxShape	= new Box(*loc);
		AreaLight*			boxLight	= new AreaLight(boxShape,Color(1.f,1.f,1.f),10.f,loc);
		boxLight->type = LIGHT;
		scene->AddObject(boxLight);
	}
	if(type == 2) {
		Box*				boxShape	= new Box(*loc);
		MirrorMaterial*		boxMat		= new MirrorMaterial(Color(1.f, 1.f, 1.f), .9f);
		Primitive*			boxPrim		= new Primitive(boxShape, boxMat, &boxShape->bounds[0]);
		boxPrim->type = PRIMITIVE;
		scene->AddObject(boxPrim);
	}
}

__device__ void CreateRoomScene(Scene* scene) {
	// Onderste rand
	PreAddBlock(&Point(1.f,0.f,1.f), scene, 0);
	PreAddBlock(&Point(1.f,0.f,2.f), scene, 0);
	PreAddBlock(&Point(1.f,0.f,3.f), scene, 0);
	PreAddBlock(&Point(1.f,0.f,4.f), scene, 0);
	PreAddBlock(&Point(1.f,0.f,5.f), scene, 0);
	PreAddBlock(&Point(1.f,0.f,6.f), scene, 0);
	PreAddBlock(&Point(2.f,0.f,6.f), scene, 0);
	PreAddBlock(&Point(2.f,0.f,1.f), scene, 0);
	PreAddBlock(&Point(3.f,0.f,1.f), scene, 0);
	PreAddBlock(&Point(4.f,0.f,1.f), scene, 0);
	PreAddBlock(&Point(5.f,0.f,1.f), scene, 0);
	PreAddBlock(&Point(6.f,0.f,1.f), scene, 0);
	PreAddBlock(&Point(6.f,0.f,2.f), scene, 0);
	PreAddBlock(&Point(6.f,0.f,3.f), scene, 0);

	// Middelste rand
	PreAddBlock(&Point(1.f,1.f,1.f), scene, 0);
	PreAddBlock(&Point(1.f,1.f,2.f), scene, 0);
	PreAddBlock(&Point(1.f,1.f,3.f), scene, 0);
	PreAddBlock(&Point(1.f,1.f,4.f), scene, 0);
	PreAddBlock(&Point(1.f,1.f,5.f), scene, 0);
	PreAddBlock(&Point(1.f,1.f,6.f), scene, 0);
	PreAddBlock(&Point(2.f,1.f,6.f), scene, 0);
	PreAddBlock(&Point(2.f,1.f,1.f), scene, 0);
	PreAddBlock(&Point(3.f,1.f,1.f), scene, 0);
	PreAddBlock(&Point(4.f,1.f,1.f), scene, 0);
	PreAddBlock(&Point(5.f,1.f,1.f), scene, 0);
	PreAddBlock(&Point(6.f,1.f,1.f), scene, 0);
	PreAddBlock(&Point(6.f,1.f,2.f), scene, 0);
	PreAddBlock(&Point(6.f,1.f,3.f), scene, 0);

	// Bovenste rand
	PreAddBlock(&Point(1.f,2.f,1.f), scene, 0);
	PreAddBlock(&Point(1.f,2.f,2.f), scene, 0);
	PreAddBlock(&Point(1.f,2.f,3.f), scene, 0);
	PreAddBlock(&Point(1.f,2.f,4.f), scene, 0);
	PreAddBlock(&Point(1.f,2.f,5.f), scene, 0);
	PreAddBlock(&Point(1.f,2.f,6.f), scene, 0);
	PreAddBlock(&Point(2.f,2.f,6.f), scene, 0);
	PreAddBlock(&Point(2.f,2.f,1.f), scene, 0);
	PreAddBlock(&Point(3.f,2.f,1.f), scene, 0);
	PreAddBlock(&Point(4.f,2.f,1.f), scene, 0);
	PreAddBlock(&Point(5.f,2.f,1.f), scene, 0);
	PreAddBlock(&Point(6.f,2.f,1.f), scene, 0);
	PreAddBlock(&Point(6.f,2.f,2.f), scene, 0);
	PreAddBlock(&Point(6.f,2.f,3.f), scene, 0);

	// Plafond
	PreAddBlock(&Point(2.f,2.f,5.f), scene, 0);
	PreAddBlock(&Point(2.f,2.f,4.f), scene, 0);

	PreAddBlock(&Point(5.f,2.f,3.f), scene, 0);
	PreAddBlock(&Point(4.f,2.f,3.f), scene, 0);
	PreAddBlock(&Point(3.f,2.f,3.f), scene, 0);
	PreAddBlock(&Point(2.f,2.f,3.f), scene, 0);

	PreAddBlock(&Point(5.f,2.f,2.f), scene, 0);
	PreAddBlock(&Point(4.f,2.f,2.f), scene, 0);
	PreAddBlock(&Point(3.f,2.f,2.f), scene, 0);
	PreAddBlock(&Point(2.f,2.f,2.f), scene, 0);

	// Lampjes
	PreAddBlock(&Point(2.f,0.f,2.f), scene, 1);
	PreAddBlock(&Point(2.f,0.f,3.f), scene, 1);
	PreAddBlock(&Point(3.f,0.f,2.f), scene, 1);
	PreAddBlock(&Point(2.f,1.f,2.f), scene, 1);
	PreAddBlock(&Point(2.f,1.f,3.f), scene, 1);
	PreAddBlock(&Point(3.f,1.f,2.f), scene, 1);

	// Schaduw blok
	PreAddBlock(&Point(4.f,0.f,4.f), scene, 0);

	// Mirror wall
	PreAddBlock(&Point(10.f,0.f,3.f), scene, 2);
	PreAddBlock(&Point(10.f,0.f,4.f), scene, 2);
	PreAddBlock(&Point(10.f,0.f,5.f), scene, 2);
	PreAddBlock(&Point(10.f,0.f,6.f), scene, 2);
	PreAddBlock(&Point(10.f,0.f,7.f), scene, 2);
	PreAddBlock(&Point(10.f,1.f,3.f), scene, 2);
	PreAddBlock(&Point(10.f,1.f,4.f), scene, 2);
	PreAddBlock(&Point(10.f,1.f,5.f), scene, 2);
	PreAddBlock(&Point(10.f,1.f,6.f), scene, 2);
	PreAddBlock(&Point(10.f,1.f,7.f), scene, 2);
	PreAddBlock(&Point(10.f,2.f,3.f), scene, 2);
	PreAddBlock(&Point(10.f,2.f,4.f), scene, 2);
	PreAddBlock(&Point(10.f,2.f,5.f), scene, 2);
	PreAddBlock(&Point(10.f,2.f,6.f), scene, 2);
	PreAddBlock(&Point(10.f,2.f,7.f), scene, 2);
	PreAddBlock(&Point(10.f,3.f,3.f), scene, 2);
	PreAddBlock(&Point(10.f,3.f,4.f), scene, 2);
	PreAddBlock(&Point(10.f,3.f,5.f), scene, 2);
	PreAddBlock(&Point(10.f,3.f,6.f), scene, 2);
	PreAddBlock(&Point(10.f,3.f,7.f), scene, 2);
	PreAddBlock(&Point(10.f,4.f,3.f), scene, 2);
	PreAddBlock(&Point(10.f,4.f,4.f), scene, 2);
	PreAddBlock(&Point(10.f,4.f,5.f), scene, 2);
	PreAddBlock(&Point(10.f,4.f,6.f), scene, 2);
	PreAddBlock(&Point(10.f,4.f,7.f), scene, 2);
	PreAddBlock(&Point(9.f,0.f,4.f), scene, 0);
}

__global__ void InitScene(Scene** pScene) {
	*pScene = new Scene();
	Scene* scene = *pScene;

	Plane*				planeShape		= new Plane(Point(0,0,0), Vector(0.f, 1.f, 0.f));
	LambertMaterial*	planeMat		= new LambertMaterial(Color(1.f, 1.f, 1.f), .9f);
	Primitive*			plane			= new Primitive(planeShape, planeMat, &planeShape->p);
	plane->type = PLANE;
	scene->AddPlane(plane);

	CreateRoomScene(scene);
}

void LaunchInitBuilder(Builder** builder) {
	InitBuilder<<<1,1>>>(builder);
}

__global__ void InitBuilder(Builder** builder) {
	*builder = new Builder();
}

void LaunchRepositionCamera(Camera* cam) {
	RepositionCamera<<<1,1>>>(cam);
}

__global__ void RepositionCamera(Camera* cam) {
	cam->Reposition();
}

void LaunchAddBlock(const Camera* cam, Scene* scene, Builder* builder) {
	AddBlock<<<1,1>>>(cam, scene, builder);
}

__global__ void AddBlock(const Camera* cam, Scene* scene, Builder* builder) {
	const Ray ray = cam->GetCenterRay();
	IntRec intRec;
	if (scene->Intersect(ray, intRec)) {
		Point* location;
		if (intRec.prim)
			location = builder->GetPosition(intRec.prim, ray(intRec.t));
		if (intRec.light)
			location = builder->GetPosition((AreaLight*)intRec.light, ray(intRec.t));

		Shape* shape = builder->GetShape(*location);
		Object* object = builder->GetObject(shape, location);
		//scene->AddPrimitive((Primitive*)object);
		scene->AddObject(object);
	}
}

void LaunchRemoveBlock(const Camera* cam, Scene* scene) {
	RemoveBlock<<<1,1>>>(cam, scene);
}

__global__ void RemoveBlock(const Camera* cam, Scene* scene) {
	Ray ray = cam->GetCenterRay();
	IntRec intRec;
	if (scene->Intersect(ray, intRec)) {
		if(intRec.light) {
			scene->RemoveObject(intRec.light);
		}
		if(intRec.prim && intRec.prim->type == PRIMITIVE) {
			scene->RemoveObject(intRec.prim);
		}
	}
}

void LaunchBuilderNextBuildType(Builder* builder) {
	BuilderNextBuildType<<<1,1>>>(builder);
}

__global__ void BuilderNextBuildType(Builder* builder) {
	builder->NextBuildType();
}

void LaunchBuilderNextShapeType(Builder* builder) {
	BuilderNextShapeType<<<1,1>>>(builder);
}

__global__ void BuilderNextShapeType(Builder* builder) {
	builder->NextShapeType();
}

void LaunchBuilderNextMaterialType(Builder* builder) {
	BuilderNextMaterialType<<<1,1>>>(builder);
}

__global__ void BuilderNextMaterialType(Builder* builder) {
	builder->NextMaterialType();
}

void LaunchBuilderSetPresetColor(Builder* builder, unsigned index)  {
	BuilderSetPresetColor<<<1,1>>>(builder, index);
}

__global__ void BuilderSetPresetColor(Builder* builder, unsigned index)  {
	builder->SetPresetColor(index);
}

void LaunchBuilderIncrease(Builder* builder)  {
	BuilderIncrease<<<1,1>>>(builder);
}

__global__ void BuilderIncrease(Builder* builder)  {
	builder->IncreaseAorI(0.2f);
}

void LaunchBuilderDecrease(Builder* builder)  {
	BuilderDecrease<<<1,1>>>(builder);
}

__global__ void BuilderDecrease(Builder* builder) {
	builder->DecreaseAorI(0.2f);
}

void LaunchIncreaseDayLight(Scene* scene) {
	IncreaseDayLight<<<1,1>>>(scene);
}

__global__ void IncreaseDayLight(Scene* scene) {
	scene->IncreaseDayLight(.1f);
}

void LaunchDecreaseDayLight(Scene* scene) {
	DecreaseDayLight<<<1,1>>>(scene);
} 

__global__ void DecreaseDayLight(Scene* scene) {
	scene->DecreaseDayLight(.1f);
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
	Color black = Color();
	Color env = scene->GetDayLight();

	Ray ray = cam->GetJitteredRay(x, y, rng);
	//const unsigned maxDepth = 4;
	IntRec intRec;
	Color color(1.f, 1.f, 1.f);
	float rr = 1.f;

	for (unsigned depth = 0; depth <= MAX_DEPTH; depth++) {
		// Enforce maximum depth
		if (depth == MAX_DEPTH) {
			color = black;
			break;
		}

		// If ray leaves the scene
		if (!scene->Intersect(ray, intRec)) {
			color *= env;
			break;
		}

		// If the ray hits a light
		if (intRec.light) {
			color *= intRec.light->Le();
			break;
		}

		// If ray hit a surface in the scene
		const Material* mat = intRec.prim->GetMaterial();
		const Shape*	shape = intRec.prim->GetShape();
		const Point		p = ray(intRec.t);
		const Vector	n = shape->GetNormal(p);

		//const Vector in = ray.d;
		const Vector out = mat->GetSample(ray.d, n, &rng[i]);
		const float multiplier = mat->GetMultiplier(ray.d, out, n);
		ray = Ray(p, out);

		color *= mat->GetColor();
		color *= multiplier;

		// Russian roulette
		if (curand_uniform(&rng[i]) > rr) {
			color = black;
			break;
		}
		rr *= multiplier;
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