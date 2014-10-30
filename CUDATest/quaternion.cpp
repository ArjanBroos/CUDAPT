#include "quaternion.h"
#include "geometry.h"
#include "math.h"

Quaternion::Quaternion(Vector& n, float a) {
	v = n * sin(a/2.f);
	w = cos(a/2.f);
}

float Quaternion::Length() const {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + w * w);
}

Quaternion Quaternion::Normalize() const {
	Quaternion result;

	float l = Length();

	result.v = v / l;
	result.w = w / l;

	return result;
}

Quaternion Quaternion::Inverted() const {
	Quaternion q;
	q.w = w;
	q.v = -v;
	return q;
}

Quaternion Quaternion::operator*(const Quaternion& q) const
{
	Quaternion r;

	r.w = w*q.w + Dot(v, q.v);
	r.v = v*q.w + q.v*w + Cross(v, q.v);

	return r;
}

Vector Quaternion::operator*(const Vector& V) const
{
	Quaternion p;
	p.w = 0;
	p.v = V;

	// Could do it this way:
	/*
	const Quaternion& q = (*this);
	return (q * p * q.Inverted()).v;
	*/

	// But let's optimize it a bit instead.
	Vector vcV = Cross(v, V);
	return V + vcV*(2*w) + Cross(v, vcV)*2;
}