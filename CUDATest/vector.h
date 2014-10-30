#ifndef VECTOR_H
#define VECTOR_H

#include <string>

// Represents a vector in 3-dimensional space
class Vector {
public:
	// Components of the vector
	float x, y, z;

	// Initialize vector as (0, 0, 0)
    Vector();
	// Initialize vector as (x, y, z)
    Vector(float x, float y, float z);

	// Returns the length of the vector
    float Length() const;
	// Returns the length of the vector squared - faster than Length()
    float LengthSquared() const;

	// Comparison operator
    bool operator==(const Vector& v) const;

	// Arithmetic operators
    Vector operator-() const;
    Vector operator+(const Vector& v) const;
    Vector& operator+=(const Vector& v);
    Vector operator*(float s) const;
    Vector& operator*=(float s);
    Vector operator/(float s) const;
    Vector& operator/=(float s);
    Vector operator-(const Vector& v) const;
    Vector& operator-=(const Vector& v);

    std::string ToString() const;
};

// Arithmetic operators
inline Vector operator*(float s, const Vector& v) {
	return Vector(s * v.x, s * v.y, s * v.z);
}

#endif
