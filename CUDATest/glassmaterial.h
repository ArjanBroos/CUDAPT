#ifndef GLASSMATERIAL_H
#define GLASSMATERIAL_H

#include "material.h"

class GlassMaterial : public Material {
public:
	// reflect + transmit = 1
	float	reflect;	// Proportion of light reflected
	float	transmit;	// Proportion of light transmitted
	float	ior;		// Index of refraction
	Color	color;		// Color of the material

	// Initializes a glass material with 30% reflectance and 70% transmittance
    GlassMaterial();
	// Initializes a material with given color, reflectance, transmittance and index of refraction
    GlassMaterial(const Color& color, float reflect, float transmit, float ior);

	// Returns the color of the material
    Color		GetColor() const;
	// Bidirectional Reflectance Distribution Function
    float		GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const;
	// Probability Density Function for Cosine-weighted distribution sampling
    float		GetPDF(const Vector& in, const Vector& out, const Vector& normal) const;
	// Returns a sample according to a Cosine-weighted distribution
    Vector		GetSample(const Vector& in, const Vector& normal) const;
	// Returns the factor between incoming and outgoing radiance along given rays
    float		GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const;
    float		GetAlbedo() const;
    MaterialType	GetType() const;

private:
	// Returns the in vector reflected about the normal
    Vector	Reflect(const Vector& in, const Vector& normal) const;
	// Returns the in vector transmitted according to the normal and the index of refraction
    Vector	Transmit(const Vector& in, const Vector& normal) const;
};

#endif
