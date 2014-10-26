//#include "cuda_inc.h"
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
#include "application.h"

int main() {
    Application application("Cloud Path Tracer", 600, 400);
    int i = 0;
	while (application.HandleEvents()) {        
		application.HandleKeyboard();
		application.HandleMouse();

		application.Render();
	}

	return 0;
}

int main2() {
    const unsigned		WIDTH = 600;
    const unsigned		HEIGHT = 400;
	const unsigned		TILE_SIZE = 8;
	const unsigned		SPP = 200;
	const unsigned		NR_PIXELS = WIDTH * HEIGHT;
	const float			FPS = 30.f;
	const std::string	name = "testmovie";

	std::cout << "Initializing..." << std::endl;

	// Initialize scene
    Scene* pScene = new Scene();
    LaunchInitScene(pScene);

	// Load world
    LaunchLoadBlocks(pScene);

    MovieMaker movie(pScene, FPS, WIDTH, HEIGHT, SPP);
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
	movie.AddControlPoint(MMControlPoint(Point(16.f, 3.f, 24.f), Normalize(Vector(-0.89f, -0.28f, -0.36f))));
	movie.AddInterpolationTime(2.f);
	movie.AddControlPoint(MMControlPoint(Point(25.f, 1.f, 22.f), Normalize(Vector(0.62f, -0.37f, 0.69f))));
	movie.AddInterpolationTime(1.f);
	movie.AddControlPoint(MMControlPoint(Point(25.6f, 1.2f, 24.f), Normalize(Vector(0.74f, -0.47f, -0.46f))));
	movie.AddInterpolationTime(1.5f);
	movie.AddControlPoint(MMControlPoint(Point(30.f, 6.f, 24.f), Normalize(Vector(-0.39f, -0.75f, 0.53f))));
	movie.AddInterpolationTime(2.f);
	movie.AddControlPoint(MMControlPoint(Point(49.f, 20.f, 46.f), Normalize(Vector(-0.70f, -0.31f, -0.63f))));

	std::cout << "Rendering movie \"" << name << "\"..." << std::endl;
	movie.RenderMovie(name);

	return 0;
}
