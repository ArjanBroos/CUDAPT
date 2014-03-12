#include "point.h"

// Initialize point as (0, 0, 0)
Point::Point() : x(0.f), y(0.f), z(0.f) {
}

// Initialize point as (x, y, z)
Point::Point(float x, float y, float z) : x(x), y(y), z(z) {
}

Vector Point::operator-(const Point& p) const {
	return Vector(x - p.x, y - p.y, z - p.z);
}

Point Point::operator+(const Vector& v) const {
	return Point(x + v.x, y + v.y, z + v.z);
}

Point& Point::operator+=(const Vector& v) {
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

Point Point::operator-(const Vector& v) const {
	return Point(x - v.x, y - v.y, z - v.z);
}

Point& Point::operator-=(const Vector& v) {
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

Point Point::operator+(const Point& p) const {
	return Point(x + p.x, y + p.y, z + p.z);
}

Point& Point::operator+=(const Point& p) {
	x += p.x;
	y += p.y;
	z += p.z;
	return *this;
}

Point Point::operator*(float s) const {
	return Point(x * s, y * s, z * s);
}

Point& Point::operator*=(float s) {
	x *= s;
	y *= s;
	z *= s;
	return *this;
}

Point Point::operator/(float s) const {
	return Point(x / s, y / s, z / s);
}

Point& Point::operator/=(float s) {
	x /= s;
	y /= s;
	z /= s;
	return *this;
}