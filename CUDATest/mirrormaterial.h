#pragma once

#include "material.h"

// Represents a perfect mirror material
class MirrorMaterial : public Material {
public:
	Color	c;		// Color
	float	albedo;	// Proportion reflected instead of absorbed

	// Initializes a perfect mirror material
	__device__ MirrorMaterial();
	// Initializes a mirror material reflecting colors in c, according to given albedo
	__device__ MirrorMaterial(const Color& c, float albedo);

	// Returns the color of this material
	__device__ Color	GetColor() const;
	// Bidirectional Reflectance Distribution Function
	__device__ float	GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const;
	// Probability Density Function
	__device__ float	GetPDF(const Vector& in, const Vector& out, const Vector& normal) const;
	// Returns a sample out direction, given an in direction and a normal
	__device__ Vector	GetSample(const Vector& in, const Vector& normal, curandState* rng) const;
	// Returns the factor between incoming and outgoing radiance along given rays
	__device__ float	GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const;

private:
	// Returns the in vector reflected about the normal
	__device__ Vector	Reflect(const Vector& in, const Vector& normal) const;
};