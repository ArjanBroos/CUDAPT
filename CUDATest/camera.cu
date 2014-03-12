#include "camera.h"
#include "math.h"

// Initializes a camera at position, looking in direction, with up being the up direction for the camera
// and a film of filmWidth x filmHeight pixels.
// The camera will have given Field of View in degrees
Camera::Camera(const Point& position, const Vector& direction, const Vector& up,
			   unsigned filmWidth, unsigned filmHeight, float FoV)
			   : pos(position) {
				   // Establish coordinate system with u, v and dir
				   dir = Normalize(direction);
				   v = Normalize(Cross(dir, up));
				   u = Cross(dir, v);

				   FoV *= 180.f / PI; // Convert FoV to radians
				   float halfWidth = tanf(FoV/2.f);
				   float aspectRatio = (float)filmWidth / (float)filmHeight;
				   u *= halfWidth; // Make u's length half of the film's width
				   v *= halfWidth * aspectRatio; // Make v's length half of the film's height

				   dx = 2.f / (float)filmWidth;
				   dy = 2.f / (float)filmHeight;
				   xmin = -1.f + dx / 2.f;
				   ymin = -1.f + dy / 2.f;
}

// Returns a ray from the viewpoint through pixel (x, y)
Ray	Camera::GetRay(unsigned x, unsigned y) const {
	Vector vx = (xmin + x*dx) * v;
	Vector vy = (ymin + y*dy) * u;
	return Ray(pos, Normalize(vx + vy + dir));
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