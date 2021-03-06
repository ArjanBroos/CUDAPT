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
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

//void LaunchCreateRoomScene2(Scene* scene, curandState* rng) {
//	CreateRoomScene2<<<1,1>>>(scene, rng);
//}

__device__ void CreateRoomScene2(Scene* scene, curandState* rng) {
	int size = 100;
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			int x = i;
			int y = curand_uniform(rng) * 5;
			int z = j;
			if(curand_uniform(rng) > .1f)
				PreAddBlock(Point(x,y,z), scene, 2);
			else
				PreAddBlock(Point(x,y,z), scene, 1);
		}
	}
}

__device__ void PreAddBlock(Point loc, Scene* scene, int type) {
		if(type == 0) {
			Box*				boxShape	= new Box(loc);
			LambertMaterial*	boxMat		= new LambertMaterial(Color(1.f, 0.f, 0.f), 1.f);
			Primitive*			boxPrim		= new Primitive(boxShape, boxMat);
			scene->AddObject(boxPrim);
		}
		if(type == 1) {
			Box*				boxShape	= new Box(loc);
			AreaLight*			boxLight	= new AreaLight(boxShape,Color(1.f,1.f,1.f),10.f);
			scene->AddObject(boxLight);
		}
		if(type == 2) {
			Box*				boxShape	= new Box(loc);
			MirrorMaterial*		boxMat		= new MirrorMaterial(Color(1.f, 1.f, 1.f), .9f);
			Primitive*			boxPrim		= new Primitive(boxShape, boxMat);
			scene->AddObject(boxPrim);
		}
	}

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

void LaunchInitScene(Scene** pScene, curandState* rng) {
	InitScene<<<1,1>>>(pScene, rng);
}

__global__ void InitScene(Scene** pScene, curandState* rng) {
	*pScene = new Scene();
	Scene* scene = *pScene;

	Plane*				planeShape		= new Plane(Point(0,0,0), Vector(0.f, 1.f, 0.f));
	LambertMaterial*	planeMat		= new LambertMaterial(Color(1.f, 1.f, 1.f), .9f);
	Primitive*			plane			= new Primitive(planeShape, planeMat);
	scene->AddPlane(plane);
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

void LaunchSetFocalPoint(Camera* cam, const Scene* scene) {
    SetFocalPoint<<<1,1>>>(cam, scene);
}

__global__ void SetFocalPoint(Camera* cam, const Scene* scene) {
    const Ray ray = cam->GetCenterRay();
    IntRec intRec;
    if (scene->Intersect(ray, intRec)) {
        cam->fpoint = max(1.f,intRec.t - 1.f);
    }
}

void LaunchRemoveBlock(const Camera* cam, Scene* scene) {
	RemoveBlock<<<1,1>>>(cam, scene);
}

__global__ void RemoveBlock(const Camera* cam, Scene* scene) {
	Ray ray = cam->GetCenterRay();
	IntRec intRec;
	if (scene->Intersect(ray, intRec)) {
		if(intRec.light) scene->RemoveObject(intRec.light);
		if(intRec.prim) scene->RemoveObject(intRec.prim);
	}
}

void LaunchSaveBlocks(Scene* scene) {
	// Ask user for a world name (for the filename)
	std::cout << "How would you like to name this world?" << std::endl;
	std::string worldName;
	std::getline(std::cin, worldName);
	worldName += ".wrld";
	if(checkOverwrite(worldName)) return;
	
	//std::ofstream writeWorldFile(worldName);

	std::stringstream contents;
	contents.clear();
	contents.str(std::string());

	std::cout << "Saving the world to file: " << worldName << ", this can take a few minutes" << std::endl;

	// Start saving process

	// Declare pointers and allocate device pointers
	int h_nBlocks, *d_nBlocks;
	h_nBlocks = 2;
	Node* d_nextNode;
	Point *d_loc, loc;
	Color *d_col, col;
	float *d_albedo, *d_intensity, albedo, intensity;
	MaterialType	*d_mat, mat;
	ShapeType *d_shape, shape;
	ObjectType *d_type,  type;
	cudaMalloc(&d_nextNode, sizeof(Node*));
	cudaMalloc(&d_nBlocks, sizeof(int));
	cudaMalloc(&d_loc, sizeof(Point));
	cudaMalloc(&d_col, sizeof(Color));
	cudaMalloc(&d_albedo, sizeof(float));
	cudaMalloc(&d_intensity, sizeof(float));
	cudaMalloc(&d_mat, sizeof(MaterialType));
	cudaMalloc(&d_shape, sizeof(ShapeType));
	cudaMalloc(&d_type, sizeof(ObjectType));

	// Query the number of objects we have to save
	NumberOfBlocks<<<1,1>>>(scene,d_nBlocks);
	cudaMemcpy(&h_nBlocks, d_nBlocks, sizeof(int), cudaMemcpyDeviceToHost);

	// Reset d_nextNode to root of the octree
	InitSaveBlocks<<<1,1>>>(scene, d_nextNode);

	// Save each object into the world file
	for(int i = 0; i < h_nBlocks; i++) {
		// Fill the variables with the correct block data
		SaveBlock<<<1,1>>>(scene, d_nextNode, d_loc, d_col, d_albedo, d_intensity, d_mat, d_shape, d_type);
		// Copy the data back to the host
		cudaMemcpy(&loc, d_loc, sizeof(Point), cudaMemcpyDeviceToHost);
		cudaMemcpy(&col, d_col, sizeof(Color), cudaMemcpyDeviceToHost);
		cudaMemcpy(&albedo, d_albedo, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&intensity, d_intensity, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&mat, d_mat, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&shape, d_shape, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&type, d_type, sizeof(int), cudaMemcpyDeviceToHost);
		// Write the block data in the world file
		//if (writeWorldFile.is_open())
		//{
		contents << "NewObject" << std::endl;
		contents << "Location: " << loc.x << " " << loc.y << " "  << loc.z << std::endl;
		contents << "Color: " << col.r << " " << col.g << " "  << col.b << std::endl;
		contents << "Albedo: " << albedo << std::endl;
		contents << "Intensity: " << intensity << std::endl;
		contents << "Material: " << mat << std::endl;
		contents << "Shape: " << shape << std::endl;
		contents << "ObjectType: " << type << std::endl;
		//}
		//else std::cout << "Unable to open file " << worldName << std::endl;
	}
	
	writeWorldFile(worldName, contents);
	//writeWorldFile.close();

	//Free pointers on device
	cudaFree(d_type);
	cudaFree(d_mat);
	cudaFree(d_shape);
	cudaFree(d_albedo);
	cudaFree(d_intensity);
	cudaFree(d_col);
	cudaFree(d_loc);
	cudaFree(d_nextNode);
	cudaFree(d_nBlocks);

	std::cout << "Done" << std::endl; 
}
__global__ void InitSaveBlocks(Scene* scene, Node* nextNode) {
	*nextNode = *scene->octree;
}

__global__ void SaveBlock(Scene* scene, Node* nextNode, Point* loc, Color* col, float* albedo, float* intensity, MaterialType* mat, ShapeType* shape, ObjectType* type) {
	// Point to the correct next leaf node
	*nextNode = *scene->octree->NextLeaf(nextNode);
	const Object* obj = nextNode->object;

	// Fill the variables with the block data
	*loc = *obj->GetCornerPoint();
	*type = obj->GetType();
	if(obj->GetType() == OBJ_PRIMITIVE) {
		*col = ((Primitive*) obj)->GetMaterial()->GetColor();
		*albedo = ((Primitive*) obj)->GetMaterial()->GetAlbedo();
		*mat = ((Primitive*) obj)->GetMaterial()->GetType();
		*shape = ((Primitive*) obj)->GetShape()->GetType();
		*intensity = -1.f;
	}
	if(obj->GetType() == OBJ_LIGHT) {
		*col = ((AreaLight*) obj)->c;
		*intensity = ((AreaLight*) obj)->i;
		*shape = ((AreaLight*) obj)->GetShape()->GetType();
		*albedo = -1.f;
		*mat = (MaterialType) 0;
	}
}

void LaunchEmptyScene(Scene* scene) {
	EmptyScene<<<1,1>>>(scene);
}

__global__ void EmptyScene(Scene* scene) {
	delete scene->octree;
	scene->octree = new Node(Point(0,0,0), Point(scene->size-1,scene->size-1,scene->size-1));
}

void LaunchLoadBlock(Scene* scene, Point loc, Color col, float albedo, float intensity, MaterialType mat, ShapeType shape, ObjectType type) {
	LoadBlock<<<1,1>>>(scene, loc, col, albedo, intensity, mat, shape, type);
}

__global__ void LoadBlock(Scene* scene, Point loc, Color col, float albedo, float intensity, MaterialType mat, ShapeType shape, ObjectType type) {
	Object* newObject;
	if(type == OBJ_PRIMITIVE) {
		Shape* shapeObj;
		if(shape == ST_CUBE)
			shapeObj = new Box(loc);
		if(shape == ST_SPHERE)
			shapeObj = new Sphere(loc);
		Material* matObj;
		if(mat == MT_DIFFUSE)
			matObj = new LambertMaterial(col, albedo);
		if(mat == MT_MIRROR)
			matObj = new MirrorMaterial(col, albedo);
		newObject = new Primitive(shapeObj, matObj);
	}

	if(type == OBJ_LIGHT) {
		Shape* shapeObj;
		if(shape == ST_CUBE)
			shapeObj = new Box(loc);
		if(shape == ST_SPHERE)
			shapeObj = new Sphere(loc);
		newObject = new AreaLight(shapeObj,col,intensity);
	}
	scene->AddObject(newObject);
}

int LaunchCountObjects(Scene* scene) {
	int h_nBlocks, *d_nBlocks;
	cudaMalloc(&d_nBlocks, sizeof(int));
	// Query the number of objects we have to save
	NumberOfBlocks<<<1,1>>>(scene,d_nBlocks);
	cudaMemcpy(&h_nBlocks, d_nBlocks, sizeof(int), cudaMemcpyDeviceToHost);
	return h_nBlocks;
}

__global__ void NumberOfBlocks(Scene* scene, int* nObjects) {
	*nObjects = scene->GetNumberOfObjects();
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
    Ray ray;
    if(cam->anti && cam->dof)
        ray = cam->GetAaDofRay(x, y, rng);
    else if(cam-> anti && !cam->dof)
        ray = cam->GetAaRay(x,y,rng);
    else if(!cam-> anti && cam->dof)
        ray = cam->GetDofRay(x,y,rng);
    else if(!cam-> anti && !cam->dof)
        ray = cam->GetNormalRay(x,y);
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
