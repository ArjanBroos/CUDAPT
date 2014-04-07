#pragma once

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

	// Returns a ray from the viewpoint through the center of pixel (x, y)
	__device__ Ray			GetRay(unsigned x, unsigned y) const;
	// Returns a ray through pixel (x, y), randomly jittered around its center
	__device__ Ray			GetJitteredRay(unsigned x, unsigned y, curandState* rng) const;
	// Returns a ray from the viewpoint through the center of the film
	__device__ Ray			GetCenterRay() const;

	__host__ __device__ void Walk(float x);
	__host__ __device__ void Strafe(float x);
	__host__ __device__ void Elevate(float x);
	__host__ __device__ void Yaw(float r);
	__host__ __device__ void Pitch(float r);
	__host__ __device__ void RotateCameraU(float angle);
	__host__ __device__ void RotateCameraV(float angle);
	__host__ __device__ void CalcUV();
	__host__ __device__ void Reposition();

	Point		pos;	// Position
	Vector		dir;	// Direction camera is looking at

private:
	Vector		u;		// Up direction of film plane
	Vector		v;		// Right direction of film plane
	Vector		worldUp;
	float		aspectRatio;
	unsigned	width;
	unsigned	height;
	float		halfWidth;

	float		xmin;	// Minimum normalized x-coordinate on film plane
	float		ymin;	// Minimum normalized y-coordinate on film plane
	float		dx;		// Difference in normalized x-coordinate of pixels on film plane
	float		dy;		// Difference in normalized y-coordinate of pixels on film plane
};