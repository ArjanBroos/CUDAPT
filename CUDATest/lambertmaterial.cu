#include "lambertmaterial.h"
#include "math.h"

// Initializes a white material with an albedo of 1
__device__ LambertMaterial::LambertMaterial() : color(Color(1.f, 1.f, 1.f)), albedo(1.f){
}

// Initializes a material with given color and albedo
__device__ LambertMaterial::LambertMaterial(const Color& color, float albedo) : color(color), albedo(albedo) {
}

// Returns the color of this material
__device__ Color LambertMaterial::GetColor() const {
	return color;
}

// Bidirectional Reflectance Distribution Function
__device__ float LambertMaterial::GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const {
	return albedo / M_PI;
}

// Probability Density Function for cosine-weighted hemisphere sampling
__device__ float LambertMaterial::GetPDF(const Vector& in, const Vector& out, const Vector& normal) const {
	return Dot(out, normal) / M_PI;
}

// Cosine weighted sampling on the unit hemisphere
__device__ Vector LambertMaterial::GetSample(const Vector& in, const Vector& normal, DRNG* rng, unsigned x, unsigned y) const {
	const Vector u = Normalize(Vector(normal.y, normal.z - normal.x, -normal.y)); // A vector perpendicular to the normal
	const Vector v = Cross(u, normal); // Another vector perpendicular to both u and the normal

	const float u1 = rng->Next(x, y);
	const float u2 = rng->Next(x, y);

	const float r = sqrtf(u1);
	const float phi = u2 * 2.f * (float)M_PI;

	const float vx = r * cosf(phi);
	const float vy = sqrtf(fmaxf(0.f, 1.f - u1));
	const float vz = r * sinf(phi);

	return u * vx + normal * vy + v * vz;
}

// Returns the factor between incoming and outgoing radiance along given rays
__device__ float LambertMaterial::GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const {
	return albedo;
}

// Returns the albedo of this material
__device__ float LambertMaterial::GetAlbedo() const {
	return albedo;
}

// Returns this type of material
__device__ MaterialType LambertMaterial::GetType() const {
	return MT_DIFFUSE;
}