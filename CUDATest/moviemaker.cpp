#include "moviemaker.h"
#include <sstream>
#include <iostream>

MovieMaker::MovieMaker(Scene* d_scene, float fps, unsigned width, unsigned height, unsigned spp) {
    this->scene = d_scene;
	this->fps = fps;
	this->width = width;
	this->height = height;
    this->spp = spp;
    result = new Color[this->width*this->height];
    pixelData = new unsigned char[width*height * 4];
    pixelData2 = new unsigned char[width*height * 3];
	camera = new Camera(Point(0.f, 0.f, 10.f), Vector(0.f, 0.f, -1.f), Vector(0.f, 1.f, 0.f), width, height, 60.f);

    LaunchInitResult(result, width, height);
}

MovieMaker::~MovieMaker() {
	delete[] pixelData;
    delete camera;
}


void MovieMaker::AddControlPoint(const MMControlPoint& p) {
	controlPoints.push_back(p);
}

void MovieMaker::AddInterpolationTime(float t) {
	interpolationTimes.push_back(t);
}

void MovieMaker::RenderMovie(const std::string& name) {
	CalculateFrames();

    std::cout << "\n" << frames.size() << std::endl;
	for (unsigned i = 0; i < frames.size(); i++) {
        float progress = 100.f * (float) i / (float)frames.size();
        RenderFrame(progress, frames[i], name, i);
	}
    std::cout << std::endl;
}

Vector MovieMaker::Interpolate(const Vector& v1, const Vector& v2, float mu) {
	return (1.f - mu) * v1 + mu * v2;
}

Point MovieMaker::Interpolate(const Point& p1, const Point& p2, float mu) {
	return (1.f - mu) * p1 + mu * p2;
}

void MovieMaker::CalculateFrames() {
	frames.clear();

	// Interpolate frames between control points
	for (unsigned i = 0; i < controlPoints.size() - 1; i++) {
		MMControlPoint& current = controlPoints[i];
		MMControlPoint& next = controlPoints[i+1];

		// Calculate number of frames until next control point
		float t = interpolationTimes[i]; // time until next control point
		unsigned nFrames = (unsigned)(t * fps);

		// Interpolate frames until next control point
		for (unsigned j = 0; j < nFrames-1; j++) {
			MMControlPoint cp;
			cp.position = Interpolate(current.position, next.position, (float)j / (float)(nFrames - 2));
			cp.direction = Interpolate(current.direction, next.direction, (float)j / (float)(nFrames - 2));
			frames.push_back(cp);
		}
	}
}

void MovieMaker::SetCamera(const MMControlPoint& p) {
	camera->pos = p.position;
	camera->dir = p.direction;
    camera->CalcUV();
}

void MovieMaker::RenderFrame(float totalProgress, const MMControlPoint& frame, const std::string& name, unsigned index) {
    LaunchInitResult(result, width, height);	// Reset frame colors
	SetCamera(frame);

	// Do path tracing for spp samples per pixel
    for (unsigned i = 0; i < spp; i++) {
        float sampleProgress = 100.f * ((float) i + 1.f) / spp;
        std::cout << "\rTotal progress: " << totalProgress << "% --- Samples progress: "  << sampleProgress << "%              " << std::flush;
        LaunchTraceRays(camera, scene, result, width, height);
    }

    // Convert data into SFML pixel data and retrieve it from GPU
    LaunchConvertRaw(result, pixelData2, spp, width, height);
}

std::string MovieMaker::FileName(const std::string& name, unsigned index) {
	unsigned nrLeadingZeros = 5;
	unsigned i = index;
	while (i > 10) {
		i /= 10;
		nrLeadingZeros--;
	}

	std::stringstream ss;
	ss << name << '-' << std::string(nrLeadingZeros, '0') << index << ".jpg";
	
	return ss.str();
}
