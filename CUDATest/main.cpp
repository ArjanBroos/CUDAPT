#include "cuda_inc.h"
#include <iostream>
#include <fstream>
#include <SFML/Graphics.hpp>
#include <sstream>
#include <ctime>

#include "ray.h"
#include "kernels.h"
#include "color.h"
#include "camera.h"
#include "geometry.h"
#include "sphere.h"
#include "interface.h"
#include "moviemaker.h"

void SetTitle(sf::RenderWindow& window, Camera* cam, unsigned iteration);
bool HandleEvents(Interface& interface, Builder* d_builder, Scene* pScene, sf::RenderWindow& window, Camera* cam, Camera* d_cam, unsigned& iteration, Color* d_result, unsigned width, unsigned height, unsigned tileSize);
void pollKeyboard(Camera* cam, Camera* d_cam, unsigned& iteration, Color* d_result, unsigned width, unsigned height, unsigned tileSize);
void resetCamera(unsigned& iteration, Color* d_result, unsigned width, unsigned height, unsigned tileSize);

bool freeze = true;
sf::Vector2i midScreen;
clock_t begin, end;

int main() {
	const unsigned		WIDTH = 1280;
	const unsigned		HEIGHT = 720;
	const unsigned		TILE_SIZE = 8;
	const unsigned		SPP = 150;
	const unsigned		NR_PIXELS = WIDTH * HEIGHT;
	const float			FPS = 30.f;
	const std::string	name = "testmovie";

	std::cout << "Initializing..." << std::endl;

	// Initialize random number generator
	curandState* d_rng;
	cudaMalloc(&d_rng, NR_PIXELS * sizeof(curandState));
	LaunchInitRNG(d_rng, (unsigned long)time(NULL), WIDTH, HEIGHT, TILE_SIZE);

	// Initialize scene
	Scene** pScene = new Scene*;
	Scene** d_pScene; cudaMalloc(&d_pScene, sizeof(Scene*));
	LaunchInitScene(d_pScene, d_rng);
	cudaMemcpy(pScene, d_pScene, sizeof(Scene*), cudaMemcpyDeviceToHost);

	// Load world
	LaunchLoadBlocks(*pScene);

	MovieMaker movie(*pScene, d_rng, FPS, WIDTH, HEIGHT, SPP);
	// Set up camera path
	movie.AddControlPoint(MMControlPoint(Point(43.f, 23.f, 11.f), Normalize(Vector(-0.70f, -0.63f, 0.33f))));
	movie.AddInterpolationTime(1.f);
	movie.AddControlPoint(MMControlPoint(Point(37.f, 10.f, 23.f), Normalize(Vector(-0.86f, -0.49f, 0.02f))));
	movie.AddInterpolationTime(1.f);
	movie.AddControlPoint(MMControlPoint(Point(26.f, 7.f, 18.f), Normalize(Vector(-0.55f, -0.32f, -0.77f))));
	movie.AddInterpolationTime(1.f);
	movie.AddControlPoint(MMControlPoint(Point(18.f, 7.f, 18.f), Normalize(Vector(0.58f, -0.35f, -0.74f))));
	movie.AddInterpolationTime(1.f);
	movie.AddControlPoint(MMControlPoint(Point(15.f, 7.f, 17.f), Normalize(Vector(-0.66f, -0.49f, 0.57f))));
	movie.AddInterpolationTime(1.f);
	movie.AddControlPoint(MMControlPoint(Point(25.f, 1.f, 22.f), Normalize(Vector(0.62f, -0.37f, 0.69f))));
	movie.AddInterpolationTime(1.f);
	movie.AddControlPoint(MMControlPoint(Point(30.f, 6.f, 24.f), Normalize(Vector(-0.39f, -0.75f, 0.53f))));
	movie.AddInterpolationTime(1.f);
	movie.AddControlPoint(MMControlPoint(Point(49.f, 20.f, 46.f), Normalize(Vector(-0.70f, -0.31f, -0.63f))));

	std::cout << "Rendering movie \"" << name << "\"..." << std::endl;
	movie.RenderMovie(name);

	return 0;
}

int main2() {
	std::cout << "CUDA path tracing tests" << std::endl;

	const unsigned WIDTH = 640;
	const unsigned HEIGHT = 480;
	const unsigned NR_PIXELS = WIDTH * HEIGHT;
	const unsigned TILE_SIZE = 8;

	std::cout << "Allocating memory..." << std::endl;

	Camera* cam = new Camera(Point(50.f, 50.f, -10.f), Normalize(Vector(0.f, -1.f, 1.f)), Vector(0.f, 1.f, 0.f), WIDTH, HEIGHT, 60.f);
	Color* result = new Color[NR_PIXELS];
	unsigned char* pixelData = new unsigned char[NR_PIXELS * 4];
	Camera* d_cam; cudaMalloc(&d_cam, sizeof(Camera));
	Color* d_result; cudaMalloc(&d_result, NR_PIXELS * sizeof(Color));
	unsigned char* d_pixelData; cudaMalloc(&d_pixelData, NR_PIXELS * 4 * sizeof(unsigned char));
	curandState* d_rng; cudaMalloc(&d_rng, NR_PIXELS * sizeof(curandState));

	LaunchInitRNG(d_rng, (unsigned long)time(NULL), WIDTH, HEIGHT, TILE_SIZE);
	LaunchInitResult(d_result, WIDTH, HEIGHT, TILE_SIZE);

	// This will be a pointer on the host, that points to a device pointer to a scene
	Scene** pScene = new Scene*;
	// Allocate memory for device pointer to device pointer to scene
	Scene** d_pScene; cudaMalloc(&d_pScene, sizeof(Scene*));
	// This will allocate the scene on the device, d_pScene will now point to the device pointer we need
	LaunchInitScene(d_pScene, d_rng);
	// Copy the device pointer from d_pScene to our host pointer
	cudaMemcpy(pScene, d_pScene, sizeof(Scene*), cudaMemcpyDeviceToHost);
	// We can now use *pScene on the host as a pointer to the Scene allocated on the device

	Builder** pBuilder = new Builder*;
	Builder** d_pBuilder; cudaMalloc(&d_pBuilder, sizeof(Builder*));
	LaunchInitBuilder(d_pBuilder);
	cudaMemcpy(pBuilder, d_pBuilder, sizeof(Builder*), cudaMemcpyDeviceToHost);


	cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);

	std::cout << "Done" << std::endl;

	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "CUDA Path tracer");
	window.setVerticalSyncEnabled(false);

	window.setMouseCursorVisible(false);
	midScreen.x = window.getPosition().x + window.getSize().x / 2;
	midScreen.y = window.getPosition().y + window.getSize().y / 2;
	sf::Mouse::setPosition(midScreen);

	Interface interface(WIDTH, HEIGHT);

	std::cout << "Calculating rays" << std::endl << std::flush;

	// Render
	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;
	bool running = true;
	unsigned iteration = 1;
	clock_t begin = clock();
	while (HandleEvents(interface, *pBuilder, *pScene, window, cam, d_cam, iteration, d_result, WIDTH, HEIGHT, TILE_SIZE)) {
		if(iteration == 301) {
			end = clock();
			double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			std::cout << "Time for 300 iterations was: " << elapsed_secs << "s" << std::endl;
		}
		if(!freeze) {
			pollKeyboard(cam, d_cam, iteration, d_result, WIDTH, HEIGHT, TILE_SIZE);
			sf::Vector2i newMousePos = sf::Mouse::getPosition();
			int dx = newMousePos.x - midScreen.x;
			int dy = newMousePos.y - midScreen.y;

			if (dx != 0 || dy != 0) {
				cam->RotateCameraU( (float)dx / -400.f );
				cam->RotateCameraV( (float)dy / -400.f );
				cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
				resetCamera(iteration, d_result, WIDTH, HEIGHT, TILE_SIZE);
				sf::Mouse::setPosition(midScreen);
			}
		}
		midScreen.x = window.getPosition().x + window.getSize().x / 2;
		midScreen.y = window.getPosition().y + window.getSize().y / 2;

		LaunchTraceRays(d_cam, *pScene, d_result, d_rng, WIDTH, HEIGHT, TILE_SIZE);
		LaunchConvert(d_result, d_pixelData, iteration, WIDTH, HEIGHT, TILE_SIZE);
		cudaMemcpy(pixelData, d_pixelData, NR_PIXELS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		image.create(WIDTH, HEIGHT, pixelData);
		texture.loadFromImage(image);
		sprite.setTexture(texture);
		window.draw(sprite);
		window.draw(interface.GetBuildIcon());
		window.draw(interface.GetCrosshair());
		window.display();

		SetTitle(window, cam, iteration);
		iteration += 1;
	}

	std::cout << "Freeing memory..." << std::endl;

	LaunchDestroyScene(*pScene);
	cudaFree(d_cam);
	cudaFree(d_result);
	cudaFree(d_pixelData);
	cudaFree(d_pScene);
	delete pScene;
	delete cam;
	delete[] result;
	delete[] pixelData;

	return 0;
}

void SetTitle(sf::RenderWindow& window, Camera* cam, unsigned iteration) {
	std::stringstream ss;
	ss << "CUDA Path Tracer - " << iteration << " - P: " << cam->pos.ToString() << " D: " << cam->dir.ToString();
	window.setTitle(ss.str());
}

void resetCamera(unsigned& iteration, Color* d_result, unsigned width, unsigned height, unsigned tileSize) {
	iteration = 1;
	begin = clock();
	LaunchInitResult(d_result, width, height, tileSize);
}

bool HandleEvents(Interface& interface, Builder* d_builder, Scene* scene, sf::RenderWindow& window, Camera* cam, Camera* d_cam, unsigned& iteration, Color* d_result, unsigned width, unsigned height, unsigned tileSize) {
	sf::Event event;
	while (window.pollEvent(event)) {
		if (event.type == sf::Event::Closed) return false;
		if (event.type == sf::Event::KeyPressed) {
			if (event.key.code == sf::Keyboard::Escape) return false;
			if (event.key.code == sf::Keyboard::L) {
				LaunchBuilderNextBuildType(d_builder);
				interface.NextBuildType();
			}
			if (event.key.code == sf::Keyboard::M) {
				LaunchBuilderNextMaterialType(d_builder);
				interface.NextMaterialType();
			}
			if (event.key.code == sf::Keyboard::N) {
				LaunchBuilderNextShapeType(d_builder);
				interface.NextShapeType();
			}
			if (event.key.code == sf::Keyboard::Num1) {
				LaunchBuilderSetPresetColor(d_builder, 0);
				interface.SetPresetColor(0);
			}
			if (event.key.code == sf::Keyboard::Num2) {
				LaunchBuilderSetPresetColor(d_builder, 1);
				interface.SetPresetColor(1);
			}
			if (event.key.code == sf::Keyboard::Num3) {
				LaunchBuilderSetPresetColor(d_builder, 2);
				interface.SetPresetColor(2);
			}
			if (event.key.code == sf::Keyboard::Num4) {
				LaunchBuilderSetPresetColor(d_builder, 3);
				interface.SetPresetColor(3);
			}
			if (event.key.code == sf::Keyboard::Num5) {
				LaunchBuilderSetPresetColor(d_builder, 4);
				interface.SetPresetColor(4);
			}
			if (event.key.code == sf::Keyboard::Num6) {
				LaunchBuilderSetPresetColor(d_builder, 5);
				interface.SetPresetColor(5);
			}
			if (event.key.code == sf::Keyboard::Num7) {
				LaunchBuilderSetPresetColor(d_builder, 6);
				interface.SetPresetColor(6);
			}
			if (event.key.code == sf::Keyboard::Num8) {
				LaunchBuilderSetPresetColor(d_builder, 7);
				interface.SetPresetColor(7);
			}
			if (event.key.code == sf::Keyboard::F) {
				freeze = !freeze;
				window.setMouseCursorVisible(freeze);
			}
			if (event.key.code == sf::Keyboard::R) {
				cam->Reposition();
				cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
				resetCamera(iteration, d_result, width, height, tileSize);
			}
			if (event.key.code == sf::Keyboard::X) LaunchBuilderIncrease(d_builder);
			if (event.key.code == sf::Keyboard::Z) LaunchBuilderDecrease(d_builder);

			if (event.key.code == sf::Keyboard::Add) {
				LaunchIncreaseDayLight(scene);
				resetCamera(iteration, d_result, width, height, tileSize);
			}
			if (event.key.code == sf::Keyboard::Subtract) {
				LaunchDecreaseDayLight(scene);
				resetCamera(iteration, d_result, width, height, tileSize);
			}
			if (event.key.code == sf::Keyboard::F1) {
				LaunchSaveBlocks(scene);
			}
			if (event.key.code == sf::Keyboard::F2) {
				LaunchLoadBlocks(scene);
				resetCamera(iteration, d_result, width, height, tileSize);
			}
			if (event.key.code == sf::Keyboard::F5) {
				std::cout << LaunchCountObjects(scene) << std::endl;
			}
		}
		if(!freeze)
			if (event.type == sf::Event::MouseButtonPressed) {
				if (event.key.code == sf::Mouse::Left) {
					LaunchAddBlock(d_cam, scene, d_builder);
					resetCamera(iteration, d_result, width, height, tileSize);
				}
				if (event.key.code == sf::Mouse::Right) {
					LaunchRemoveBlock(d_cam, scene);
					resetCamera(iteration, d_result, width, height, tileSize);
				}
			}
	}
	return true;
}

// Handles continuous pressed keys
void pollKeyboard(Camera* cam, Camera* d_cam, unsigned& iteration, Color* d_result, unsigned width, unsigned height, unsigned tileSize) 
{
	const float step = 0.4f;
	const float rstep = 0.05f;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
		cam->Walk(step);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
		cam->Walk(-step);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
		cam->Strafe(-step);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
		cam->Strafe(step);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
		cam->Elevate(step);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) {
		cam->Elevate(-step);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
		cam->RotateCameraU(rstep);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
		cam->RotateCameraU(-rstep);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
		cam->RotateCameraV(-rstep);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
		cam->RotateCameraV(rstep);
		cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
		resetCamera(iteration, d_result, width, height, tileSize);
	}
}