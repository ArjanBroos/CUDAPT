#include "plane.h"

// Initializes a plane through (0, 0, 0) with normal (0, 1, 0)
__device__ Plane::Plane() : p(Point(0.f, 0.f, 0.f)), n(Vector(0.f, 1.f, 0.f)) {
}

// Initializes a plane through p with normal n
__device__ Plane::Plane(const Point& p, const Vector& n) : p(p), n(n) {
}

// Returns true when this shape intersects ray
// If so, output parameter t becomes the distance along ray to the closest intersection
__device__ bool Plane::Intersect(const Ray& ray, float& t) const {
	const float nDotD = Dot(n, ray.d);
	// If ray and plane are perpendicular
	if (fabsf(nDotD) < 1e-10f)
		return false;

	// Find intersection point between ray and plane
	t = Dot(p - ray.o, n) / nDotD;
	if (t < ray.mint || t > ray.maxt)
		return false; // Does not intersect with the right part of the ray

	return true;
}
	
// Returns the normal of this shape at point p
__device__ Vector Plane::GetNormal(const Point& p) const {
	return n;
}