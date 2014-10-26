#ifndef CAMERA_H
#define CAMERA_H

#include "geometry.h"
#include "ray.h"
#include "cuda_inc.h"
#include <curand_kernel.h>

// Represents a pinhole camera
class Camera {
public:
	// Initializes a camera at position, looking in direction, with up being the up direction for the camera
	// and a film of filmWidth x filmHeight pixels.
	// The camera will have given Field of View in degrees
	Camera(const Point& position, const Vector& direction, const Vector& up,
		unsigned filmWidth, unsigned filmHeight, float FoV);

    // Returns a ray normal ray throug (x, y)
    Ray			GetNormalRay(unsigned x, unsigned y) const;
    // Returns a ray ray throug (x, y) with aa and dof enabled
    Ray			GetAaDofRay(unsigned x, unsigned y) const;
    // Returns a ray ray throug (x, y) with only aa enabled
    Ray			GetAaRay(unsigned x, unsigned y) const;
    // Returns a ray ray throug (x, y) with only dof enable
    Ray			GetDofRay(unsigned x, unsigned y) const;
	// Returns a ray from the viewpoint through the center of the film
    Ray			GetCenterRay() const;

    void Walk(float x);
    void Strafe(float x);
    void Elevate(float x);
    void Yaw(float r);
    void Pitch(float r);
    void RotateCameraU(float angle);
    void RotateCameraV(float angle);
    void CalcUV();
    void Reposition();

	Point		pos;	// Position
	Vector		dir;	// Direction camera is looking at
    float       fov;    // Field Of View Angle

    // Settings Depth of Field
    bool        dof;
    float       aperture;
    float       fpoint;

    // Settings Anti-Aliasing
    bool        anti;

private:
	Vector		u;		// Up direction of film plane
	Vector		v;		// Right direction of film plane
	Vector		worldUp;
	float		aspectRatio;
	unsigned	width;
    unsigned	height;

	float		xmin;	// Minimum normalized x-coordinate on film plane
	float		ymin;	// Minimum normalized y-coordinate on film plane
	float		dx;		// Difference in normalized x-coordinate of pixels on film plane
    float		dy;		// Difference in normalized y-coordinate of pixels on film plane
};

#endif
