#include "light.h"
#include "shape.h"

// Represents an area light (ie. with a surface/shape)
class AreaLight : public Light {
public:
	Shape*	shape;	// The shape of this area light

	// Initializes area light with given shape (cannot be NULL) and reasonable attenuation factors
	__device__ AreaLight(Shape* shape);
	// Initializes area light with given shape (cannot be NULL) and given attenuation factors
	__device__ AreaLight(Shape* shape, Color color, float intensity);

	// Returns true when this light intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ bool	Intersect(const Ray& ray, float& t) const;
};