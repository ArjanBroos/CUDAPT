#include "moviemaker.h"
#include <sstream>
#include <iostream>

MovieMaker::MovieMaker(Scene* d_scene, unsigned width, unsigned height, unsigned spp) {
    this->scene = d_scene;
	this->width = width;
	this->height = height;
    this->spp = spp;
    result = new Color[this->width*this->height];
    pixelData = new unsigned char[width*height * 3];
	camera = new Camera(Point(0.f, 0.f, 10.f), Vector(0.f, 0.f, -1.f), Vector(0.f, 1.f, 0.f), width, height, 60.f);

    LaunchInitResult(result, width, height);
}

MovieMaker::~MovieMaker() {
	delete[] pixelData;
    delete camera;
}

void MovieMaker::SetCamera(const MMControlPoint& p) {
	camera->pos = p.position;
	camera->dir = p.direction;
    camera->CalcUV();
}

void MovieMaker::RenderFrame() {
	// Do path tracing for spp samples per pixel
    for (unsigned i = 0; i < spp; i++) {
        float sampleProgress = 100.f * ((float) i + 1.f) / spp;
        std::cout << "\rSamples progress: "  << sampleProgress << "%              " << std::flush;
        LaunchTraceRays(camera, scene, result, width, height);
    }
    std::cout << std::endl;

    // Convert data into SFML pixel data and retrieve it from GPU
    LaunchConvertRaw(result, pixelData, spp, width, height);
}
