#include "scene.h"
#include "math.h"

__device__ Scene::Scene() : primCounter(0), lightCounter(0), objectCounter(0), planeCounter(0), nextId(0) {
	primitives = new Primitive*[MAX_PRIMITIVES];
	objects = new Object*[MAX_OBJECTS];
	lights = new Light*[MAX_LIGHTS];
	octree = (*new Node(Point(0,0,0), Point(63,63,63), nextId));
}

__device__ Scene::~Scene() {
	for (unsigned i = 0; i < primCounter; i++)
		delete primitives[i];
	delete[] primitives;
	for (unsigned i = 0; i < lightCounter; i++)
		delete lights[i];
	delete[] lights;
}

__device__ void Scene::AddPrimitive(Primitive* primitive) {
	if (primCounter < MAX_PRIMITIVES - 1)
		primitives[primCounter++] = primitive;
}

__device__ void Scene::AddPlane(Primitive* plane) {
	if (planeCounter < MAX_PLANES - 1)
		planes[planeCounter++] = plane;
}

__device__ void Scene::AddLight(Light* light) {
	if (lightCounter < MAX_LIGHTS - 1)
		lights[lightCounter++] = light;
}

// Adds an object to the octree of the scene
__device__ void Scene::AddObject(Object* object) {
	if (objectCounter < MAX_PRIMITIVES - 1) {
		objects[objectCounter] = object;
		octree.Insert(objects[objectCounter], nextId);
		objectCounter++;
	}
}

__device__ bool Scene::Intersect(const Ray& ray, IntRec& intRec) const {
	intRec.t = INFINITY;
	float t;
	bool intersected = false;

	//// Check intersection with primitives
	//for (unsigned i = 0; i < primCounter; i++) {
	//	if (primitives[i]->GetShape()->Intersect(ray, t) && t < intRec.t) {
	//		intRec.prim = primitives[i];
	//		intRec.t = t;
	//		intersected = true;
	//	}
	//}

	//// Check intersection with lights
	//for (unsigned i = 0; i < lightCounter; i++) {
	//	if (lights[i]->Intersect(ray, t) && t < intRec.t) {
	//		intRec.prim = NULL;
	//		intRec.light = lights[i];
	//		intRec.t = t;
	//		intersected = true;
	//	}
	//}
	for (unsigned i = 0; i < planeCounter; i++) {
		if(planes[i]->Intersect(ray,t) && t < intRec.t) {
			intRec.prim = (Primitive*) planes[i];
			intRec.light = false;
			intRec.t = t;
			intersected = true;
		}
	}

	if(octree.Intersect(ray, intRec))
		intersected = true;

	return intersected;
}

__device__ bool Scene::IntersectOctree(const Ray& ray, IntRec& intRec, bool& intersected) const {
	//bool intersect = false;
	//intRec.t = INFINITY;

	// Query the octree for intersection
	if(octree.Intersect(ray, intRec))
		intersected = true;

	return intersected;
}