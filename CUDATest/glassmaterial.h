#pragma once

#include "material.h"

class GlassMaterial : public Material {
public:
	// reflect + transmit = 1
	float	reflect;	// Proportion of light reflected
	float	transmit;	// Proportion of light transmitted
	float	ior;		// Index of refraction
	Color	color;		// Color of the material

	// Initializes a glass material with 30% reflectance and 70% transmittance
	__device__ GlassMaterial();
	// Initializes a material with given color, reflectance, transmittance and index of refraction
	__device__ GlassMaterial(const Color& color, float reflect, float transmit, float ior);

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
	__device__ float		GetAlbedo() const;
	__device__ MaterialType	GetType() const;

private:
	// Returns the in vector reflected about the normal
	__device__ Vector	Reflect(const Vector& in, const Vector& normal) const;
	// Returns the in vector transmitted according to the normal and the index of refraction
	__device__ Vector	Transmit(const Vector& in, const Vector& normal) const;
};