#include "application.h"
#include <ctime>
#include <sstream>
#include <iostream>
#include <ctime>
#include <algorithm>

Application::Application(const std::string& title, unsigned width, unsigned height) : frozen(false)
{
    this->width = width;
    this->height = height;
    const unsigned NR_PIXELS = width * height;
    iteration = 1;

    // Set up camera
    cam = new Camera(Point(-2.f, 4.f, -2.f), Normalize(Vector(2.f, -4.f, 2.f)), Vector(0.f, 1.f, 0.f), width, height, 70.f);
    //cudaMalloc(&d_cam, sizeof(Camera));
    //cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);

    // Set up memory for results
    result = new Color[NR_PIXELS];
    //cudaMalloc(&d_result, NR_PIXELS * sizeof(Color));
    //LaunchInitResult(d_result, width, height, tileSize);

    // Set up memory for pixel data
    pixelData = new unsigned char[NR_PIXELS * 4];

    // Allocate scene on device and make sure we can pass on this pointer via host
    scene = new Scene();
    LaunchInitScene(scene);

    // Allocate builder on device and make sure we can pass on this pointer via host
    builder = new Builder();

    // Set up window
//    window.setVerticalSyncEnabled(false);
//    window.setMouseCursorVisible(false);
//    midScreen.x = window.getPosition().x + window.getSize().x / 2;
//    midScreen.y = window.getPosition().y + window.getSize().y / 2;
//    sf::Mouse::setPosition(midScreen);
}

Application::~Application() {
    //cudaFree(*builder);
    //cudaFree(d_builder);

    //cudaFree(*scene);
    //cudaFree(d_scene);

    //cudaFree(d_rng);

    //cudaFree(d_pixelData);
    delete[] pixelData;

    //cudaFree(d_result);
    delete[] result;

    //cudaFree(d_cam);
    delete cam;
}

void Application::Render() {
    LaunchTraceRays(cam, scene, result, width, height);
    LaunchConvert(result, pixelData, iteration, width, height);

    iteration += 1;
}

void Application::Reset() {
    iteration = 1;
    LaunchInitResult(result, width, height);
}

bool Application::HandleEvents() {
}

void Application::HandleKeyboard() {
}

void Application::HandleMouse() {
}

void Application::UpdateTitle() {
    std::stringstream ss;
    ss << "CUDA Path Tracer - " << iteration << " - P: " << cam->pos.ToString() << " D: " << cam->dir.ToString();
}

void Application::UpdateDeviceCamera() {
    LaunchSetFocalPoint(cam, scene);
}
