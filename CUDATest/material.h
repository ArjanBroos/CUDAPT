#pragma once

#include "color.h"
#include "geometry.h"
#include "curand_kernel.h"

enum MaterialType {
	MT_START = 0,
	MT_DIFFUSE = MT_START,
	MT_MIRROR,
	MT_GLASS,
	MT_END,
};

// Represents the base material
class Material {
public:
	// Returns the color of this material
	__device__ virtual Color		GetColor() const = 0;
	// Bidirectional Reflectance Distribution Function
	__device__ virtual float		GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const = 0;
	// Probability Density Function
	__device__ virtual float		GetPDF(const Vector& in, const Vector& out, const Vector& normal) const = 0;
	// Returns a sample out direction, given an in direction and a normal
	__device__ virtual Vector		GetSample(const Vector& in, const Vector& normal, curandState* rng) const = 0;
	// Returns the factor between incoming and outgoing radiance along given rays
	__device__ virtual float		GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const = 0;
	// Returns the albedo of this material
	__device__ virtual float		GetAlbedo() const = 0;
	// Returns the type of material
	__device__ virtual MaterialType	GetType() const = 0;
protected:
};