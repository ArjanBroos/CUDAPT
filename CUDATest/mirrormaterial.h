#ifndef MIRRORMATERIAL_H
#define MIRRORMATERIAL_H

#include "material.h"

// Represents a perfect mirror material
class MirrorMaterial : public Material {
public:
	Color	c;		// Color
	float	albedo;	// Proportion reflected instead of absorbed

	// Initializes a perfect mirror material
    MirrorMaterial();
	// Initializes a mirror material reflecting colors in c, according to given albedo
    MirrorMaterial(const Color& c, float albedo);

	// Returns the color of this material
    Color		GetColor() const;
	// Bidirectional Reflectance Distribution Function
    float		GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const;
	// Probability Density Function
    float		GetPDF(const Vector& in, const Vector& out, const Vector& normal) const;
	// Returns a sample out direction, given an in direction and a normal
    Vector		GetSample(const Vector& in, const Vector& normal) const;
	// Returns the factor between incoming and outgoing radiance along given rays
    float		GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const;
	// Returns the albedo of this material
    float		GetAlbedo() const;
	// Returns this type of material
    MaterialType	GetType() const;

private:
	// Returns the in vector reflected about the normal
    Vector	Reflect(const Vector& in, const Vector& normal) const;
};

#endif
