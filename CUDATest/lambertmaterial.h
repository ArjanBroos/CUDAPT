#pragma once

#include "material.h"

// Represents a Lambertian material
class LambertMaterial : public Material {
public:
	float	albedo;	// Proportion of light reflected instead of absorbed
	Color	color;	// Color of the material

	// Initializes a white material with an albedo of 1
	__device__ LambertMaterial();
	// Initializes a material with given color and albedo
	__device__ LambertMaterial(const Color& color, float albedo);

	// Returns the color of the material
	__device__ Color		GetColor() const;
	// Bidirectional Reflectance Distribution Function
	__device__ float		GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const;
	// Probability Density Function for Cosine-weighted distribution sampling
	__device__ float		GetPDF(const Vector& in, const Vector& out, const Vector& normal) const;
	// Returns a sample according to a Cosine-weighted distribution
	__device__ Vector		GetSample(const Vector& in, const Vector& normal, DRNG* rng, unsigned x, unsigned y) const;
	// Returns the factor between incoming and outgoing radiance along given rays
	__device__ float		GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const;
	// Returns the albedo of this material
	__device__ float		GetAlbedo() const;
	// Returns the type of this material
	__device__ MaterialType	GetType() const;
};