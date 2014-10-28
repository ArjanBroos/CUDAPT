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
#include "task.h"

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

void getTask(Task* task) {
    // TODO: WAIT ON SOCKET AND RETRIEVE VARIABLES
    int jobId = 0;
    int width = 640;
    int height = 480;
    int frame = 0;
    int nSamples = 0;
    Point cameraPosition(10.f, 1.2f, 3.f);
    Vector cameraOrientation(0.03f, -0.36f, -0.9f);
    std::string worldToRender = "movieWorld";

    // define the task
    task->setJobId(jobId);
    task->setWidth(width);
    task->setHeight(height);
    task->setFrame(frame);
    task->setNSamples(nSamples);
    task->setCameraPosition(cameraPosition);
    task->setCameraOrientation(cameraOrientation);
    task->setWorldToRender(worldToRender);
}

int runMovieMaker() {
    bool waitForNewTask = true;
    while(waitForNewTask) {
        Task* newTask = new Task();
        getTask(newTask);

        std::stringstream nameSs;
        nameSs << newTask->getJobId() << "_" << newTask->getFrame();
        const std::string	name = nameSs.str();

        std::cout << "Initializing..." << std::endl;

        // Initialize scene
        Scene* pScene = new Scene();
        LaunchInitScene(pScene);

        // Load world
        LaunchLoadBlocks(pScene, newTask->getWorldToRender());

        MovieMaker movie(pScene, newTask->getWidth(), newTask->getHeight(), newTask->getNSamples());
        // Set up camera
        MMControlPoint camera = MMControlPoint(newTask->getCameraPosition(), Normalize(newTask->getCameraOrientation()));
        movie.SetCamera(camera);



        std::cout << "Rendering movie \"" << name << "\"..." << std::endl;
        movie.RenderFrame();
        //movie.RenderMovie(name);

        return 0;
    }
}

int main() {
    runMovieMaker();
    return 0;
}
