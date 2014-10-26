#ifndef LAMBERTMATERIAL_H
#define LAMBERTMATERIAL_H

#include "material.h"

// Represents a Lambertian material
class LambertMaterial : public Material {
public:
	float	albedo;	// Proportion of light reflected instead of absorbed
	Color	color;	// Color of the material

	// Initializes a white material with an albedo of 1
    LambertMaterial();
	// Initializes a material with given color and albedo
    LambertMaterial(const Color& color, float albedo);

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
	// Returns the albedo of this material
    float		GetAlbedo() const;
	// Returns the type of this material
    MaterialType	GetType() const;
};

#endif
