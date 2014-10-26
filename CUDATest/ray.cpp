#include "ray.h"

// Initializes a ray with origin o and direction d, mint and maxt for intersection optional
Ray::Ray(const Point& o, const Vector& d, float ior, float mint, float maxt) : o(o), d(d), ior(ior), mint(mint), maxt(maxt) {
	inv = Vector(1.f/d.x, 1.f/d.y, 1.f/d.z);
	sign[0] = (inv.x < 0);
	sign[1] = (inv.y < 0);
	sign[2] = (inv.z < 0);
}

// Returns o + t*d
Point Ray::operator()(float t) const {
	return o + d*t;
}