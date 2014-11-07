#ifndef MOVIEMAKER_H
#define MOVIEMAKER_H

#include "kernels.h"
#include "camera.h"
#include "scene.h"
#include "color.h"
#include <string>
#include <vector>

struct MMControlPoint {
	MMControlPoint() : position(Point(0.f, 0.f, 0.f)), direction(Vector(0.f, 0.f, -1.f)) {}
	MMControlPoint(Point p, Vector d) : position(p), direction(d) {}

	Point	position;
	Vector	direction;
};

// Renders movies by specifying a scene and a set of camera points to interpolate between
class MovieMaker {
public:
	// Initialize a movie with given scene (already allocated), frames per second, pixel width, pixel height and samples per pixel
    MovieMaker(Scene* scene, unsigned width, unsigned height, unsigned spp);
	~MovieMaker();

    // Render the movie by saving them as a lot of jpg files
    unsigned char*						RenderFrame();
    void						SetCamera(const MMControlPoint& p);

private:

    Scene*						scene;
    Camera*						camera;
	unsigned					width;
	unsigned					height;
    Color*						result;
    unsigned char*				pixelData;
    unsigned					spp;					// Samples per pixel
};

#endif
