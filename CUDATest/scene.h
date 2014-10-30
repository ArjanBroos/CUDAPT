#ifndef SCENE_H
#define SCENE_H

#include "geometry.h"
#include "primitive.h"
#include "intrec.h"
#include "light.h"
#include "box.h"
#include "lambertmaterial.h"
#include "octree.h"
#include "plane.h"
#include "color.h"

#define MAX_PRIMITIVES	1024
#define MAX_OBJECTS	1024
#define MAX_PLANES	8
#define MAX_LIGHTS		1024

class Scene {
public:
    Scene();
    ~Scene();
    void AddPrimitive(Primitive* primitive);
    void AddPlane(Primitive* plane);
    void AddLight(Light* light);
	// Adds object to the octree of the scene
    void AddObject(Object* object);
    void RemoveObject(Object* object);
    void IncreaseDayLight(float amount);
    void DecreaseDayLight(float amount);
    const Color GetDayLight() const;
    bool	Intersect(const Ray& ray, IntRec& intRec) const;
    bool	IntersectOctree(const Ray& ray, IntRec& intRec, bool& intersected) const;
    int GetNumberOfObjects() const;
	int			size;
	Node*		octree;

private:
	unsigned	primCounter;
	unsigned	lightCounter;
	Primitive**	primitives;
	Light**		lights;

	unsigned	planeCounter;
	unsigned	objectCounter;
	Primitive**	planes;
	Object**	objects;
	Color		dayLight;
};

#endif
