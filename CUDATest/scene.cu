#include "scene.h"
#include "math.h"

__device__ Scene::Scene() : primCounter(0), lightCounter(0) {
	primitives = new Primitive*[MAX_PRIMITIVES];
	lights = new Light*[MAX_LIGHTS];
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

__device__ void Scene::AddLight(Light* light) {
	if (lightCounter < MAX_LIGHTS - 1)
		lights[lightCounter++] = light;
}

__device__ bool Scene::Intersect(const Ray& ray, IntRec& intRec) const {
	float mint = INFINITY;
	float t;
	bool intersected = false;

	// Check intersection with primitives
	for (unsigned i = 0; i < primCounter; i++) {
		if (primitives[i]->GetShape()->Intersect(ray, t) && t < mint) {
			mint = t;
			intRec.prim = primitives[i];
			intRec.t = t;
			intersected = true;
		}
	}

	// Check intersection with lights
	for (unsigned i = 0; i < lightCounter; i++) {
		if (lights[i]->Intersect(ray, t) && t < mint) {
			mint = t;
			intRec.prim = NULL;
			intRec.light = lights[i];
			intRec.t = t;
			intersected = true;
		}
	}

	return intersected;
}