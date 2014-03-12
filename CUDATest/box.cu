#include "box.h"
#include <cassert>

__device__ Box::Box(Point& p)
{
	bounds[0] = p;
	bounds[1]= Point(p.x+1.f,p.y+1.f,p.z+1.f);
}

__device__ Box::Box(Point& p, float s)
{
	bounds[0] = p;
	bounds[1]= Point(p.x+s,p.y+s,p.z+s);
}

__device__ Box::Box(Point& p, Point& q)
{
	bounds[0] = p;
	bounds[1]= q;
}

__device__ bool Box::Intersect(const Ray& ray, float& t) const
{
	float tmin, tmax, tymin, tymax, tzmin, tzmax;

	//Check x-direction
	if (ray.d.x >= 0) {
		tmin = (bounds[0].x - ray.o.x) / ray.d.x;
		tmax = (bounds[1].x - ray.o.x) / ray.d.x;
	}
	else {
		tmin = (bounds[1].x - ray.o.x) / ray.d.x;
		tmax = (bounds[0].x - ray.o.x) / ray.d.x;
	}
	//Check y-direction
	if (ray.d.y >= 0) {
		tymin = (bounds[0].y - ray.o.y) / ray.d.y;
		tymax = (bounds[1].y - ray.o.y) / ray.d.y;
	}
	else {
		tymin = (bounds[1].y - ray.o.y) / ray.d.y;
		tymax = (bounds[0].y - ray.o.y) / ray.d.y;
	}
	//Compare to previous interval
	if ( (tmin > tymax) || (tymin > tmax) )
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	//Check z-direction
	if (ray.d.z >= 0) {
		tzmin = (bounds[0].z - ray.o.z) / ray.d.z;
		tzmax = (bounds[1].z - ray.o.z) / ray.d.z;
	}
	else {
		tzmin = (bounds[1].z - ray.o.z) / ray.d.z;
		tzmax = (bounds[0].z - ray.o.z) / ray.d.z;
	}
	//Compare to previous interval
	if ( (tmin > tzmax) || (tzmin > tmax) )
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;
	if ( (tmin < ray.maxt) && (tmax > ray.mint) ) {
		t = tmin;
		return true;
	}
	return false;
}

// Returns the normal of this sphere at point p
__device__ Vector	Box::GetNormal(const Point& p) const
{
	const float epsilon = 1e-5f;
	if( fabsf(p.x - bounds[0].x) < epsilon) {
		return Vector(-1.f,0.f,0.f);
	}
	if( fabsf(p.y - bounds[0].y) < epsilon) {
		return Vector(0.f,-1.f,0.f);
	}
	if( fabsf(p.z - bounds[0].z) < epsilon) {
		return Vector(0.f,0.f,-1.f);
	}
	if( fabsf(p.x - bounds[1].x) < epsilon) {
		return Vector(1.f,0.f,0.f);
	}
	if( fabsf(p.y - bounds[1].y) < epsilon) {
		return Vector(0.f,1.f,0.f);
	}
	if( fabsf(p.z - bounds[1].z) < epsilon) {
		return Vector(0.f,0.f,1.f);
	}

	return Vector(.0f,.0f,.0f);
}