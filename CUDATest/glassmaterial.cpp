#include "glassmaterial.h"
#include "math.h"

// Initializes a white material with an albedo of 1
GlassMaterial::GlassMaterial() : color(Color(1.f, 1.f, 1.f)), reflect(0.3f), transmit(0.7f), ior(1.5f) {
}

// Initializes a material with given color and albedo
GlassMaterial::GlassMaterial(const Color& color, float reflect, float transmit, float ior) :
	color(color), reflect(reflect), transmit(transmit), ior(ior) {
}

// Returns the color of this material
Color GlassMaterial::GetColor() const {
	return color;
}

// Bidirectional Reflectance Distribution Function
float GlassMaterial::GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const {
	return 1.f;
}

// Probability Density Function for cosine-weighted hemisphere sampling
float GlassMaterial::GetPDF(const Vector& in, const Vector& out, const Vector& normal) const {
	return 1.f;
}

// Cosine weighted sampling on the unit hemisphere
Vector GlassMaterial::GetSample(const Vector& in, const Vector& normal) const {
	return Transmit(in, normal);
}

// Returns the factor between incoming and outgoing radiance along given rays
float GlassMaterial::GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const {
	return 1.f;
}

// Returns the in vector reflected about the normal
Vector GlassMaterial::Reflect(const Vector& in, const Vector& normal) const {
	return -2.f * Dot(in, normal) * normal + in;
}

Vector GlassMaterial::Transmit(const Vector& in, const Vector& normal) const {
	bool goingIn = Dot(in, normal) < 0.f;
	if (goingIn) return Normalize(in - normal);
	else return in;
}

float GlassMaterial::GetAlbedo() const {
	return 1.f;
}

MaterialType GlassMaterial::GetType() const {
	return MT_GLASS;
}
