#pragma once

#include "kernels.h"
#include "camera.h"
#include "scene.h"
#include "color.h"
#include <SFML\Graphics.hpp>
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
	MovieMaker(Scene* d_scene, curandState* d_rng, float fps, unsigned width, unsigned height, unsigned spp);
	~MovieMaker();

	// Add a control point, which is a point and direction for the camera in the movie
	void						AddControlPoint(const MMControlPoint& p);
	// Add an interpolation time, time[i] is the time it takes to get from controlPoint[i] to controlPoint[i+1]
	void						AddInterpolationTime(float t);

	// Render the movie by saving them as a lot of jpg files
	void						RenderMovie(const std::string& name);

private:
	Vector						Interpolate(const Vector& v1, const Vector& v2, float mu);
	Point						Interpolate(const Point& p1, const Point& p2, float mu);
	void						CalculateFrames();
	void						SetCamera(const MMControlPoint& p);
	void						RenderFrame(const MMControlPoint& frame, const std::string& name, unsigned index);
	std::string					FileName(const std::string& name, unsigned index);

	Scene*						d_scene;
	Camera*						camera;
	Camera*						d_camera;

	float						fps;					// Frames per second
	std::vector<MMControlPoint>	controlPoints;
	std::vector<float>			interpolationTimes;

	std::vector<MMControlPoint>	frames;					// Eventual frames to be rendered (camera positions+directions)

	unsigned					width;
	unsigned					height;
	Color*						d_result;
	unsigned char*				pixelData;
	unsigned char*				d_pixelData;
	unsigned					spp;					// Samples per pixel
	sf::Image					image;

	curandState*				d_rng;

	static const unsigned		TILE_SIZE = 8;
};