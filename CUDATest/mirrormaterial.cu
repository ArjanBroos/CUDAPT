#include "mirrormaterial.h"

// Initializes a perfect mirror material
__device__ MirrorMaterial::MirrorMaterial() : c(Color(1.f, 1.f, 1.f)), albedo(1.f) {
}

// Initializes a mirror material reflecting colors in c, according to given albedo
__device__ MirrorMaterial::MirrorMaterial(const Color& c, float albedo) : c(c), albedo(albedo) {
}

// Returns the color of this material
__device__ Color MirrorMaterial::GetColor() const {
	return c;
}

// Bidirectional Reflectance Distribution Function
__device__ float MirrorMaterial::GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const {
	return (Reflect(in, normal) == out) ? albedo : 0.f;
}

// Probability Density Function
__device__ float MirrorMaterial::GetPDF(const Vector& in, const Vector& out, const Vector& normal) const {
	return (Reflect(in, normal) == out) ? 1.f : 1e-30f; // Prevent division by 0
}

// Returns a sample out direction, given an in direction and a normal
__device__ Vector MirrorMaterial::GetSample(const Vector& in, const Vector& normal, curandState* rng) const {
	return Reflect(in, normal);
}

// Returns the factor between incoming and outgoing radiance along given rays
__device__ float MirrorMaterial::GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const {
	return albedo;
}

// Returns the in vector reflected about the normal
__device__ Vector MirrorMaterial::Reflect(const Vector& in, const Vector& normal) const {
	return -2.f * Dot(in, normal) * normal + in;
}

// Returns the albedo of this material
__device__ float MirrorMaterial::GetAlbedo() const {
	return albedo;
}

// Returns this type of material
__device__ MaterialType MirrorMaterial::GetType() const {
	return MT_MIRROR;
}