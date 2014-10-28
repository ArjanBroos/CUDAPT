#include <iostream>
#include <fstream>
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

bool movieMaker = true;

int runRealTime() {
    Application application("Cloud Path Tracer", 600, 400);
    int i = 0;
	while (application.HandleEvents()) {        
		application.HandleKeyboard();
		application.HandleMouse();

		application.Render();
	}
	return 0;
}

int runMovieMaker() {
    const unsigned		WIDTH = 1920;
    const unsigned		HEIGHT = 1080;
    const unsigned		SPP = 300;
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
    movie.AddControlPoint(MMControlPoint(Point(-5.f, 1.2f, 3.f), Normalize(Vector(0.03f, -0.36f, -0.9f))));
    movie.AddInterpolationTime(10.f);
    movie.AddControlPoint(MMControlPoint(Point(21.f, 1.2f, 3.f), Normalize(Vector(0.03f, -0.36f, -0.9f))));


    std::cout << "Rendering movie \"" << name << "\"..." << std::endl;
	movie.RenderMovie(name);

	return 0;
}

int main() {
    if(movieMaker) runMovieMaker();
    else runRealTime();
    return 0;
}
