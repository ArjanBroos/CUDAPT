#include "box.h"
#include <cassert>

__device__ Box::Box(const Point& p)
{
	bounds[0] = p;
	bounds[1]= Point(p.x+1.f,p.y+1.f,p.z+1.f);
}

__device__ Box::Box(const Point& p, float s)
{
	bounds[0] = p;
	bounds[1]= Point(p.x+s,p.y+s,p.z+s);
}

__device__ Box::Box(const Point& p, const Point& q)
{
	bounds[0] = p;
	bounds[1]= q;
}

__device__ bool Box::Intersect(const Ray& ray, float& t) const
{
	float tmin, tmax, tminn, tmaxn;

	tmin = (bounds[ray.sign[0]].x - ray.o.x) * ray.inv.x;
	tmax = (bounds[1-ray.sign[0]].x - ray.o.x) * ray.inv.x;
	tminn = (bounds[ray.sign[1]].y - ray.o.y) * ray.inv.y;
	tmaxn = (bounds[1-ray.sign[1]].y - ray.o.y) * ray.inv.y;

	//Compare to previous interval
	if ( (tmin > tmaxn) || (tminn > tmax) )
		return false;
	if (tminn > tmin)
		tmin = tminn;
	if (tmaxn < tmax)
		tmax = tmaxn;

	tminn = (bounds[ray.sign[2]].z - ray.o.z) * ray.inv.z;
	tmaxn = (bounds[1-ray.sign[2]].z - ray.o.z) * ray.inv.z;
	//Compare to previous interval
	if ( (tmin > tmaxn) || (tminn > tmax) )
		return false;
	if (tminn > tmin)
		tmin = tminn;
	if (tmaxn < tmax)
		tmax = tmaxn;
	if ( (tmin < ray.maxt) && (tmax > ray.mint) && (tmax > 0)) {
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

// Returns the type of this shape
__device__ ShapeType Box::GetType() const {
	return ST_CUBE;
}

// Return the corner point of object
__device__ const Point*		Box::GetCornerPoint() const {
	return &bounds[0];
}