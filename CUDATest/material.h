#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "geometry.h"
#include "curand_kernel.h"

enum MaterialType {
	MT_START = 0,
	MT_DIFFUSE = MT_START,
	MT_MIRROR,
	MT_GLASS,
	MT_END = MT_GLASS,
};

// Represents the base material
class Material {
public:
	// Returns the color of this material
    virtual Color		GetColor() const = 0;
	// Bidirectional Reflectance Distribution Function
    virtual float		GetBRDF(const Vector& in, const Vector& out, const Vector& normal) const = 0;
	// Probability Density Function
    virtual float		GetPDF(const Vector& in, const Vector& out, const Vector& normal) const = 0;
	// Returns a sample out direction, given an in direction and a normal
    virtual Vector		GetSample(const Vector& in, const Vector& normal) const = 0;
	// Returns the factor between incoming and outgoing radiance along given rays
    virtual float		GetMultiplier(const Vector& in, const Vector& out, const Vector& normal) const = 0;
	// Returns the albedo of this material
    virtual float		GetAlbedo() const = 0;
	// Returns the type of material
    virtual MaterialType	GetType() const = 0;
protected:
};

#endif
