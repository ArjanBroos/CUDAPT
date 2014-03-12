#include "sphere.h"
#include <random>
#include <ctime>

// Initializes a sphere at (0, 0, 0) with radius 1
__device__ Sphere::Sphere() : c(Point(0.f, 0.f, 0.f)), r(1.f) {
}

// Initializes a sphere at c with radius r
__device__ Sphere::Sphere(const Point& c, float r) : c(c), r(r) {
}

// Returns true when this sphere intersects ray
// If so, output parameter t becomes the distance along ray to the closest intersection
__device__ bool Sphere::Intersect(const Ray& ray, float& t) const {
	// Use abc-formula to solve for the two possible intersection points

	Vector v = ray.o - c;
	float a = Dot(ray.d, ray.d);
	float b = Dot(ray.d, v);
	float c = Dot(v, v) - r*r;
	float d = b*b - a*c;

	if (d < 0.f) return false; // No intersection

	float D = sqrtf(d);

	// s1 is always smaller than s2 and we want to return the closest valid intersection
	float s1 = (-b - D) / a;
	if (s1 >= ray.mint && s1 <= ray.maxt) {
		t = s1;
		return true;
	}
	float s2 = (-b + D) / a;
	if (s2 >= ray.mint && s2 <= ray.maxt) {
		t = s2;
		return true;
	}
	
	return false;
}

// Returns the normal of this sphere at point p
__device__ Vector Sphere::GetNormal(const Point& p) const {
	return Normalize(p - c);
}