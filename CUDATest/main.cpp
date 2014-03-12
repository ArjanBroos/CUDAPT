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

void SetTitle(sf::RenderWindow& window, unsigned iteration) {
	std::stringstream ss;
	ss << "CUDA Path Tracer - Iteration: " << iteration;
	window.setTitle(ss.str());
}

int main() {
	std::cout << "CUDA path tracing tests" << std::endl;

	const unsigned WIDTH = 1024;
	const unsigned HEIGHT = 768;
	const unsigned NR_PIXELS = WIDTH * HEIGHT;
	const unsigned TILE_SIZE = 8;

	std::cout << "Allocating memory..." << std::endl;

	Camera* cam = new Camera(Point(0.f, 4.f, 10.f), Normalize(Vector(0.f, -0.4f, -1.f)), Vector(0.f, 1.f, 0.f), WIDTH, HEIGHT, 60.f);
	Color* result = new Color[NR_PIXELS];
	unsigned char* pixelData = new unsigned char[NR_PIXELS * 4];
	Camera* d_cam; cudaMalloc(&d_cam, sizeof(Camera));
	Color* d_result; cudaMalloc(&d_result, NR_PIXELS * sizeof(Color));
	unsigned char* d_pixelData; cudaMalloc(&d_pixelData, NR_PIXELS * 4 * sizeof(unsigned char));
	curandState* d_rng; cudaMalloc(&d_rng, sizeof(curandState));

	LaunchInitRNG(d_rng, (unsigned long)time(NULL));

	// This will be a pointer on the host, that points to a device pointer to a scene
	Scene** pScene = new Scene*;
	// Allocate memory for device pointer to device pointer to scene
	Scene** d_pScene; cudaMalloc(&d_pScene, sizeof(Scene*));
	// This will allocate the scene on the device, d_pScene will now point to the device pointer we need
	LaunchInitScene(d_pScene);
	// Copy the device pointer from d_pScene to our host pointer
	cudaMemcpy(pScene, d_pScene, sizeof(Scene*), cudaMemcpyDeviceToHost);
	// We can now use *pScene on the host as a pointer to the Scene allocated on the device

	cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);

	std::cout << "Done" << std::endl;

	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "CUDA Path tracer");
	window.setVerticalSyncEnabled(false);

	bool running = true;
	unsigned iteration = 0;
	while (running) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) running = false;
			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) running = false;
		}
		window.clear();

		LaunchTraceRays(d_cam, *pScene, d_result, d_rng, WIDTH, HEIGHT, TILE_SIZE);
		LaunchConvert(d_result, d_pixelData, WIDTH, HEIGHT, TILE_SIZE);
		cudaMemcpy(pixelData, d_pixelData, NR_PIXELS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		sf::Image image;
		sf::Texture texture;
		sf::Sprite sprite;
		image.create(WIDTH, HEIGHT, pixelData);
		texture.loadFromImage(image);
		sprite.setTexture(texture);
		window.draw(sprite);
		window.display();

		SetTitle(window, iteration++);
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
}