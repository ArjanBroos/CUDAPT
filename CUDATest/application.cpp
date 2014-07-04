#include "application.h"
#include <ctime>
#include <sstream>
#include <iostream>
#include <ctime>
#include <algorithm>

Application::Application(const std::string& title, unsigned width, unsigned height, unsigned tileSize) :
	window(sf::VideoMode(width, height), title),
	interface(width, height), frozen(false)
{
	this->width = width;
	this->height = height;
	this->tileSize = tileSize;
	const unsigned NR_PIXELS = width * height;
	iteration = 1;

	// Set up camera
    cam = new Camera(Point(-2.f, 4.f, -2.f), Normalize(Vector(2.f, -4.f, 2.f)), Vector(0.f, 1.f, 0.f), width, height, 70.f);
	cudaMalloc(&d_cam, sizeof(Camera));
	cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);

	// Set up memory for results
	result = new Color[NR_PIXELS];
	cudaMalloc(&d_result, NR_PIXELS * sizeof(Color));
	LaunchInitResult(d_result, width, height, tileSize);

	// Set up memory for pixel data
	pixelData = new unsigned char[NR_PIXELS * 4];
	cudaMalloc(&d_pixelData, NR_PIXELS * 4 * sizeof(unsigned char));

	// Set up RNG states
	cudaMalloc(&d_rng, NR_PIXELS * sizeof(curandState));
	LaunchInitRNG(d_rng, (unsigned long)time(NULL), width, height, tileSize);

	// Allocate scene on device and make sure we can pass on this pointer via host
	scene = new Scene*;
	cudaMalloc(&d_scene, sizeof(Scene*));
	LaunchInitScene(d_scene, d_rng);
	cudaMemcpy(scene, d_scene, sizeof(Scene*), cudaMemcpyDeviceToHost);

	// Allocate builder on device and make sure we can pass on this pointer via host
	builder = new Builder*;
	cudaMalloc(&d_builder, sizeof(Builder*));
	LaunchInitBuilder(d_builder);
	cudaMemcpy(builder, d_builder, sizeof(Builder*), cudaMemcpyDeviceToHost);

	// Set up window
	window.setVerticalSyncEnabled(false);
	window.setMouseCursorVisible(false);
	midScreen.x = window.getPosition().x + window.getSize().x / 2;
	midScreen.y = window.getPosition().y + window.getSize().y / 2;
	sf::Mouse::setPosition(midScreen);
}

Application::~Application() {
	cudaFree(*builder);
	cudaFree(d_builder);

	cudaFree(*scene);
	cudaFree(d_scene);

	cudaFree(d_rng);

	cudaFree(d_pixelData);
	delete[] pixelData;

	cudaFree(d_result);
	delete[] result;

	cudaFree(d_cam);
	delete cam;
}

void Application::Render() {
	LaunchTraceRays(d_cam, *scene, d_result, d_rng, width, height, tileSize);
	LaunchConvert(d_result, d_pixelData, iteration, width, height, tileSize);
	cudaMemcpy(pixelData, d_pixelData, width*height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	image.create(width, height, pixelData);
	texture.loadFromImage(image);
	sprite.setTexture(texture);

	window.draw(sprite);
	window.draw(interface.GetBuildIcon());
	window.draw(interface.GetCrosshair());
	window.display();

	UpdateTitle();
	iteration += 1;
}

void Application::Reset() {
	iteration = 1;
	LaunchInitResult(d_result, width, height, tileSize);
}

bool Application::HandleEvents() {
	sf::Event event;
	while (window.pollEvent(event)) {
		if (event.type == sf::Event::Closed) return false;

		if (event.type == sf::Event::KeyPressed) {
			if (event.key.code == sf::Keyboard::Escape) return false;
			if (event.key.code == sf::Keyboard::L) {
				LaunchBuilderNextBuildType(*builder);
				interface.NextBuildType();
			}
			if (event.key.code == sf::Keyboard::M) {
				LaunchBuilderNextMaterialType(*builder);
				interface.NextMaterialType();
			}
			if (event.key.code == sf::Keyboard::N) {
				LaunchBuilderNextShapeType(*builder);
				interface.NextShapeType();
			}
			if (event.key.code == sf::Keyboard::Num1) {
				LaunchBuilderSetPresetColor(*builder, 0);
				interface.SetPresetColor(0);
			}
			if (event.key.code == sf::Keyboard::Num2) {
				LaunchBuilderSetPresetColor(*builder, 1);
				interface.SetPresetColor(1);
			}
			if (event.key.code == sf::Keyboard::Num3) {
				LaunchBuilderSetPresetColor(*builder, 2);
				interface.SetPresetColor(2);
			}
			if (event.key.code == sf::Keyboard::Num4) {
				LaunchBuilderSetPresetColor(*builder, 3);
				interface.SetPresetColor(3);
			}
			if (event.key.code == sf::Keyboard::Num5) {
				LaunchBuilderSetPresetColor(*builder, 4);
				interface.SetPresetColor(4);
			}
			if (event.key.code == sf::Keyboard::Num6) {
				LaunchBuilderSetPresetColor(*builder, 5);
				interface.SetPresetColor(5);
			}
			if (event.key.code == sf::Keyboard::Num7) {
				LaunchBuilderSetPresetColor(*builder, 6);
				interface.SetPresetColor(6);
			}
			if (event.key.code == sf::Keyboard::Num8) {
				LaunchBuilderSetPresetColor(*builder, 7);
				interface.SetPresetColor(7);
			}
			if (event.key.code == sf::Keyboard::F) {
				frozen = !frozen;
				window.setMouseCursorVisible(frozen);
			}
			if (event.key.code == sf::Keyboard::R) {
				cam->Reposition();
				UpdateDeviceCamera();
				Reset();
			}
			if (event.key.code == sf::Keyboard::X) LaunchBuilderIncrease(*builder);
			if (event.key.code == sf::Keyboard::Z) LaunchBuilderDecrease(*builder);

			if (event.key.code == sf::Keyboard::Add) {
				LaunchIncreaseDayLight(*scene);
				Reset();
			}
			if (event.key.code == sf::Keyboard::Subtract) {
				LaunchDecreaseDayLight(*scene);
				Reset();
			}
			if (event.key.code == sf::Keyboard::F1) {
				LaunchSaveBlocks(*scene);
			}
			if (event.key.code == sf::Keyboard::F2) {
				LaunchLoadBlocks(*scene);
				Reset();
			}
			if (event.key.code == sf::Keyboard::F5) {
				std::cout << LaunchCountObjects(*scene) << std::endl;
			}
            if (event.key.code == sf::Keyboard::P) {
                time_t now = time(0);
                std::stringstream ss;
                ss << now;
                std::string timeString = ss.str();
                ss.clear();
                ss.str("");
                ss << iteration;
                std::string iterationString = ss.str();
                image.saveToFile("PrintScreen" + timeString + "_" + iterationString + ".jpg");
            }
		}
		if(!frozen) {
			if (event.type == sf::Event::MouseButtonPressed) {
				if (event.mouseButton.button == sf::Mouse::Left) {
					LaunchAddBlock(d_cam, *scene, *builder);			
					Reset();
				}
				if (event.mouseButton.button == sf::Mouse::Right) {
					LaunchRemoveBlock(d_cam, *scene);
					Reset();
				}
			}
            if (event.type == sf::Event::MouseWheelMoved) {
                cam->fov = std::max(0.1f*PI, std::min(0.9f*PI, cam->fov + 2.0f*PI*(float)(-event.mouseWheel.delta) / 180.f));
                cam->CalcUV();
                UpdateDeviceCamera();
                Reset();
            }
		}
		if(event.type == sf::Event::GainedFocus) {
			midScreen.x = window.getPosition().x + window.getSize().x / 2;
			midScreen.y = window.getPosition().y + window.getSize().y / 2;
		}
	}

	return true;
}

void Application::HandleKeyboard() {
	if (frozen) return;

	const float step  = 0.4f;
	const float rstep = 0.05f;
	bool keyWasPressed = false;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {			cam->Walk(step); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {	 		cam->Walk(-step); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {			cam->Strafe(-step); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {			cam->Strafe(step); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {		cam->Elevate(step); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) {	cam->Elevate(-step); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {		cam->RotateCameraU(rstep); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {		cam->RotateCameraU(-rstep); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {			cam->RotateCameraV(-rstep); keyWasPressed = true; }
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {		cam->RotateCameraV(rstep); keyWasPressed = true; }
	if (keyWasPressed) {
		UpdateDeviceCamera();
		Reset();
	}
}

void Application::HandleMouse() {
	if (frozen) return;

	const float rstep = 400.f;
	sf::Vector2i newMousePos = sf::Mouse::getPosition();
	int dx = newMousePos.x - midScreen.x;
	int dy = newMousePos.y - midScreen.y;

	// Rotate camera according to mouse movement
	if (dx != 0 || dy != 0) {
		cam->RotateCameraU((float)dx / -rstep);
		cam->RotateCameraV((float)dy / -rstep);
		UpdateDeviceCamera();
		Reset();
		sf::Mouse::setPosition(midScreen);
	}
}

void Application::UpdateTitle() {
	std::stringstream ss;
	ss << "CUDA Path Tracer - " << iteration << " - P: " << cam->pos.ToString() << " D: " << cam->dir.ToString();
	window.setTitle(ss.str());
}

void Application::UpdateDeviceCamera() {
	cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
}
