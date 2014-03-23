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

__global__ void InitScene(Scene** pScene) {
	*pScene = new Scene();
	Scene* scene = *pScene;

	Plane*				planeShape		= new Plane(Point(0,0,0), Vector(0.f, 1.f, 0.f));
	LambertMaterial*	planeMat		= new LambertMaterial(Color(1.f, 1.f, 0.3f), 1.f);
	Primitive*			plane			= new Primitive(planeShape, planeMat, &planeShape->p);
	plane->type = PLANE;
	scene->AddPlane(plane);

	Box*				sphereShape1	= new Box(Point(0.f, 0.f, 0.f));
	LambertMaterial*	sphereMat1		= new LambertMaterial(Color(1.f, 0.f, 0.f), 1.f);
	Primitive*			spherePrim1		= new Primitive(sphereShape1, sphereMat1, &sphereShape1->bounds[0]);
	spherePrim1->type = PRIMITIVE;
	scene->AddObject(spherePrim1);

	Box*				sphereShape2	= new Box(Point(127.f, 0.f, 0.f));
	LambertMaterial*	sphereMat2		= new LambertMaterial(Color(1.f, 0.f, 0.f), 1.f);
	Primitive*			spherePrim2		= new Primitive(sphereShape2, sphereMat2, &sphereShape2->bounds[0]);
	spherePrim2->type = PRIMITIVE;
	scene->AddObject(spherePrim2);

	Box*				sphereShape3	= new Box(Point(127.f, 0.f, 127.f));
	LambertMaterial*	sphereMat3		= new LambertMaterial(Color(1.f, 0.f, 0.f), 1.f);
	Primitive*			spherePrim3		= new Primitive(sphereShape3, sphereMat3, &sphereShape3->bounds[0]);
	spherePrim3->type = PRIMITIVE;
	scene->AddObject(spherePrim3);

	Box*				sphereShape4	= new Box(Point(0.f, 0.f, 127.f));
	LambertMaterial*	sphereMat4		= new LambertMaterial(Color(1.f, 0.f, 0.f), 1.f);
	Primitive*			spherePrim4		= new Primitive(sphereShape4, sphereMat4, &sphereShape4->bounds[0]);
	spherePrim4->type = PRIMITIVE;
	scene->AddObject(spherePrim4);
}

void LaunchChangeLight(Scene* scene) {
	ChangeLight<<<1,1>>>(scene);
}

__global__ void ChangeLight(Scene* scene) {
	if(scene->nextLight) {
		scene->nextLight = !scene->nextLight;
		scene->nextR = 1.f;
		scene->nextG = 0.f;
		scene->nextB = 0.f;
	} else {
		scene->nextLight = !scene->nextLight;
		scene->nextR = 1.f;
		scene->nextG = 1.f;
		scene->nextB = 1.f;
	}
}

void LaunchAddBlock(const Camera* cam, Scene* scene) {
	AddBlock<<<1,1>>>(cam, scene);
}

__global__ void AddBlock(const Camera* cam, Scene* scene) {
	Ray ray = cam->GetCenterRay();
	IntRec intRec;
	if (scene->Intersect(ray, intRec)) {
		const Shape*	shape;
		Point*			newLoc;
		Vector			n;
		Point			p = ray(intRec.t);
		if(intRec.light) {
			shape = ((AreaLight*)(intRec.light))->shape;
			n = shape->GetNormal(p);
			newLoc = new Point((int)(intRec.light->loc->x+n.x+.5f), (int)(intRec.light->loc->y+n.y+.5f), (int)(intRec.light->loc->z+n.z+.5f)); 
		}
		if(intRec.prim) {
			shape = intRec.prim->GetShape();
			n = shape->GetNormal(p);
			if(intRec.prim->type == PLANE)
				newLoc = new Point(Point((int)p.x, (int)p.y, (int)p.z));
			else {
				newLoc = new Point((int)(intRec.prim->loc->x+n.x+.5f), (int)(intRec.prim->loc->y+n.y+.5f), (int)(intRec.prim->loc->z+n.z+.5f)); 
			}
		}
		if(!scene->nextLight) {
			Box*				sphereShape1	= new Box(*newLoc);
			LambertMaterial*	sphereMat1		= new LambertMaterial(Color(scene->nextR, scene->nextG, scene->nextB), 1.f);
			Primitive*			spherePrim1		= new Primitive(sphereShape1, sphereMat1, &sphereShape1->bounds[0]);
			spherePrim1->type = PRIMITIVE;
			scene->AddObject(spherePrim1);
		} else {
			Box*				lightShape1	= new Box(*newLoc);
			AreaLight*			light1 = new AreaLight(lightShape1, Color(scene->nextR, scene->nextG, scene->nextB), scene->nextE, &lightShape1->bounds[0]);
			light1->type = LIGHT;
			scene->AddObject(light1);
		}
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
	__shared__ Color black;
	__shared__ Color env;
	if(threadIdx.x == 0) {
		black = Color();
		env = Color(0.2f, 0.2f, 0.3f);
	}
	__syncthreads();

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