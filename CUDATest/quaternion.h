#ifndef QUATERNION_H
#define QUATERNION_H

#include "vector.h"

class Quaternion {
public:
    Quaternion() {};
    Quaternion( Vector& n, float a);
	
    float Length() const;
    Quaternion Normalize() const;
    Quaternion Inverted() const;
    Quaternion operator*(const Quaternion& q) const;
    Vector operator*(const Vector& q) const;

	float w;
	Vector v;
};

#endif
