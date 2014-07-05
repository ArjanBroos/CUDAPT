#include "camera.h"
#include "math.h"
#include "quaternion.h"
#define AA true
#define DOF true

// Initializes a camera at position, looking in direction, with up being the up direction for the camera
// and a film of filmWidth x filmHeight pixels.
// The camera will have given Field of View in degrees
Camera::Camera(const Point& position, const Vector& direction, const Vector& up,
			   unsigned filmWidth, unsigned filmHeight, float FoV)
               : pos(position), fov(FoV), fpoint(4.f), aperture(10.f), anti(true), dof(true) {
				   // Establish coordinate system with u, v and dir
				   dir = Normalize(direction);
				   v = Normalize(Cross(dir, up));
				   u = Cross(dir, v);
				   worldUp = up;
                   fov *= (PI / 180.f); // Convert FoV to radians
                   float halfWidth = tanf(fov/2.f);
				   aspectRatio = (float)filmWidth / (float)filmHeight;
				   u *= halfWidth; // Make u's length half of the film's width
				   v *= halfWidth * aspectRatio; // Make v's length half of the film's height

				   width = filmWidth;
				   height = filmHeight;

				   dx = 2.f / (float)filmWidth;
				   dy = 2.f / (float)filmHeight;
				   xmin = -1.f + dx / 2.f;
				   ymin = -1.f + dy / 2.f;
}

__device__ Ray Camera::GetNormalRay(unsigned x, unsigned y) const {
    Vector vx = (xmin + x*dx) * v;
    Vector vy = (ymin + y*dy) * u;
    return Ray(pos, Normalize(vx + vy + dir));
}

// Returns a ray through pixel (x, y), randomly jittered around its center
__device__ Ray Camera::GetAaDofRay(unsigned x, unsigned y, curandState* rng) const {
    const float pxmin = -1.f + (float)x * dx;		// Pixel x mininum
    const float pymin = -1.f + (float)y * dy;		// Pixel y minimum

    // Calculate the ray to the focal point with AA
    float apx = pxmin + curand_uniform(rng) * dx;
    float apy = pymin + curand_uniform(rng) * dy;
    float aperx = -aperture/2.0f*dx + pxmin + curand_uniform(rng) * aperture*dx;
    float apery = -aperture/2.0f*dy + pymin + curand_uniform(rng) * aperture*dy;
    Vector vx = apx * v;
    Vector vy = apy * u;
    Vector focalDir = dir+vx+vy + fpoint*Normalize(dir+vx+vy);

    // Calculate the aperDir ray for DoF
    Vector avx = aperx * v;
    Vector avy = apery * u;
    Vector aperDir = dir+avx+avy;

    // Calculate the ray to trace
    Vector renderDir = focalDir - aperDir;
    return Ray(pos+dir+avx+avy, Normalize(renderDir));
}

// Returns a ray through pixel (x, y), randomly jittered around its center
__device__ Ray Camera::GetAaRay(unsigned x, unsigned y, curandState* rng) const {
    const float pxmin = -1.f + (float)x * dx;		// Pixel x mininum
    const float pymin = -1.f + (float)y * dy;		// Pixel y minimum

    // Calculate aa point
    float px = pxmin + curand_uniform(rng) * dx;
    float py = pymin + curand_uniform(rng) * dy;

    // Calculate the ray to the focal point
    Vector vx = px * v;
    Vector vy = py * u;
    return Ray(pos, Normalize(vx + vy + dir));
}

// Returns a ray through pixel (x, y), randomly jittered around its center
__device__ Ray Camera::GetDofRay(unsigned x, unsigned y, curandState* rng) const {
    const float pxmin = -1.f + (float)x * dx;		// Pixel x mininum
    const float pymin = -1.f + (float)y * dy;		// Pixel y minimum

    // Calculate the aperDir ray
    float aperx = -aperture/2.0f*dx + pxmin + curand_uniform(rng) * aperture*dx;
    float apery = -aperture/2.0f*dy + pymin + curand_uniform(rng) * aperture*dy;
    Vector avx = aperx * v;
    Vector avy = apery * u;
    Vector aperDir = dir+avx+avy;

    // Calculate the ray to the focal point
    Vector vx = (xmin + x*dx) * v;
    Vector vy = (ymin + y*dy) * u;
    Vector focalDir = dir+vx+vy + fpoint*Normalize(dir+vx+vy);

    // Calculate the ray to trace
    Vector renderDir = focalDir - aperDir;
    return Ray(pos+dir+avx+avy, Normalize(renderDir));
}

// Returns a ray from the viewpoint through the center of the film
__device__ Ray	Camera::GetCenterRay() const {
    float x = (float) width / 2.f;
    float y = (float) height / 2.f;
    Vector vx = (xmin + x*dx) * v;
    Vector vy = (ymin + y*dy) * u;
    return Ray(pos, Normalize(vx + vy + dir));
}

void Camera::Reposition() {
	pos = Point(0.f, 4.f, 0.f);
	dir = Vector(1.f, -1.f, 1.f);
	CalcUV();
}

void Camera::Walk(float x)
{
	pos.x += x*dir.x;
	pos.z += x*dir.z;
}

void Camera::Strafe(float x)
{
	pos.x += x*v.x;
	pos.z += x*v.z;
}

void Camera::Elevate(float x)
{
	pos.y += x;
}

void Camera::Yaw(float r)
{
	//Rotate each vector around the y-axis
	float x = dir.x*cosf(r) + dir.z*sinf(r);
	float y = dir.y;
	float z = -dir.x*sinf(r) + dir.z*cosf(r);
	dir = Vector(x, y, z);
	x = u.x*cosf(r) + u.z*sinf(r);
	y = u.y;
	z = -u.x*sinf(r) + u.z*cosf(r);
	u = Vector(x, y, z);
	x = v.x*cosf(r) + v.z*sinf(r);
	y = v.y;
	z = -v.x*sinf(r) + v.z*cosf(r);
	v = Vector(x, y, z);
}

void Camera::Pitch(float r)
{
	//Rotate dir above x-axis
	Vector stdRy(1,0,0);
	Vector actRy(dir);
	actRy.y = 0;
	float rx = -acosf(Dot(actRy, stdRy));
	float x = dir.x*cosf(rx) + dir.z*sinf(rx);
	float y = dir.y;
	float z = -dir.x*sinf(rx) + dir.z*cosf(rx);
	dir = Vector(x, y, z);
	//Pitch dir
	x = dir.x*cosf(r) + dir.z*sinf(r);
	y = -dir.x*sinf(r) + dir.y*cosf(r);
	z = dir.z;
	dir = Vector(x, y, z);
	//Rotate dir back
	x = dir.x*cosf(-rx) + dir.z*sinf(-rx);
	y = dir.y;
	z = -dir.x*sinf(-rx) + dir.z*cosf(-rx);
	dir = Vector(x, y, z);

	//Rotate u above x-axis
	stdRy = Vector(1,0,0);
	actRy = Vector(u);
	actRy.y = 0;
	rx = -acosf(Dot(actRy, stdRy));
	x = u.x*cosf(rx) + u.z*sinf(rx);
	y = u.y;
	z = -u.x*sinf(rx) + u.z*cosf(rx);
	u = Vector(x, y, z);
	//Pitch u
	x = u.x*cosf(r) + u.z*sinf(r);
	y = -u.x*sinf(r) + u.y*cosf(r);
	z = u.z;
	u = Vector(x, y, z);
	//Rotate u back
	x = u.x*cosf(-rx) + u.z*sinf(-rx);
	y = u.y;
	z = -u.x*sinf(-rx) + u.z*cosf(-rx);
	u = Vector(x, y, z);
}

void Camera::RotateCameraU(float angle)
{
  Quaternion rQuat;
  rQuat = Quaternion(worldUp, angle);
  dir = rQuat * dir;

  CalcUV();
}

void Camera::RotateCameraV(float angle)
{
  Quaternion rQuat;
  rQuat = Quaternion(v, angle);
  dir = rQuat * dir;

  CalcUV();
}

void Camera::CalcUV() {
	dir = Normalize(dir);
	v = Normalize(Cross(dir, worldUp));
	u = Cross(dir, v);
    float halfWidth = tanf(fov/2.f);
	u *= halfWidth; // Make u's length half of the film's width
	v *= halfWidth * aspectRatio; // Make v's length half of the film's height
}
