#include "vector.h"
#include "math.h"
#include <sstream>

// Initialize vector as (0, 0, 0)
Vector::Vector() : x(0.f), y(0.f), z(0.f) {
}

// Initialize vector as (x, y, z)
Vector::Vector(float x, float y, float z) : x(x), y(y), z(z) {
}

// Returns the length of the vector
float Vector::Length() const {
	return sqrtf(x*x + y*y + z*z);
}

// Returns the length of the vector squared - faster than Length()
float Vector::LengthSquared() const {
	return x*x + y*y + z*z;
}

bool Vector::operator==(const Vector& v) const {
	return (x == v.x) && (y == v.y) && (z == v.z);
}

Vector Vector::operator-() const {
	return Vector(-x, -y, -z);
}

Vector Vector::operator+(const Vector& v) const {
	return Vector(x + v.x, y + v.y, z + v.z);
}

Vector& Vector::operator+=(const Vector& v) {
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

Vector Vector::operator*(float s) const {
	return Vector(x * s, y * s, z * s);
}

Vector& Vector::operator*=(float s) {
	x *= s;
	y *= s;
	z *= s;
	return *this;
}

Vector Vector::operator/(float s) const {
	return Vector(x / s, y / s, z / s);
}

Vector& Vector::operator/=(float s) {
	x /= s;
	y /= s;
	z /= s;
	return *this;
}

Vector Vector::operator-(const Vector& v) const {
	return Vector(x - v.x, y - v.y, z - v.z);
}

Vector& Vector::operator-=(const Vector& v) {
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

std::string Vector::ToString() const {
	std::stringstream ss;
	ss << "(" << x << ", " << y << ", " << z << ")";
	return ss.str();
}
