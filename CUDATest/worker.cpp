#include "worker.h"
#include "moviemaker.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>

Worker::Worker() {
}

// Connect to master node
bool Worker::Connect(const std::string &address, const std::string &port) {
    return client.Connect(address, port);
}

// Receive a task from the master node
Task Worker::ReceiveTask() {
    // First receive the invariably sized part
    RcvData rd;
    client.Receive(rd);

    if (rd.size != 5 * sizeof(int) + sizeof(Point) + sizeof(Vector)) {
        std::cout << "Error, expected different size in ReceiveTask" << std::endl;
    }

    // Start with pointer at start of data
    //     move pointer/iterator along and switch types where necessary
    Task task;
    int *intIter = (int*)rd.data;
    task.setJobId(*intIter); intIter++;
    task.setWidth(*intIter); intIter++;
    task.setHeight(*intIter); intIter++;
    task.setFrame(*intIter); intIter++;
    task.setNSamples(*intIter); intIter++;

    Point* pointIter = (Point*)intIter;
    task.setCameraPosition(*pointIter);
    pointIter++;

    Vector* vectorIter = (Vector*)pointIter;
    task.setCameraOrientation(*vectorIter);

    // Free old data and receive world name
    delete[] rd.data;
    client.Receive(rd);

    std::string world(rd.data, rd.size);
    task.setWorldToRender(world);

    delete[] rd.data;

    return task;
}

// Performs the given task (rendering an image with path tracer)
byte *Worker::PerformTask(const Task &task) {
    time_t timer = time(0);
    std::cout << "::: Task started on " << GetTimeStamp() << std::endl;
    std::cout << "::: Performing task. Job ID: " << task.getJobId() << ", Frame: " << task.getFrame() << std::endl;
    std::cout << "        World name: " << task.getWorldToRender() << std::endl;

    std::stringstream nameSs;
    nameSs << task.getJobId() << "_" << task.getFrame();
    const std::string	name = nameSs.str();

    std::cout << "Initializing..." << std::endl;

    // Initialize scene
    Scene* pScene = new Scene();
    LaunchInitScene(pScene);

    // Load world
    LaunchLoadBlocks(pScene, task.getWorldToRender());

    MovieMaker movie(pScene, task.getWidth(), task.getHeight(), task.getNSamples());
    // Set up camera
    MMControlPoint camera = MMControlPoint(task.getCameraPosition(), Normalize(task.getCameraOrientation()));
    movie.SetCamera(camera);

    std::cout << "Rendering image \"" << name << "\"..." << std::endl;
    unsigned char* frame = movie.RenderFrame();
    result.clear();
    result.resize(task.getWidth() * task.getHeight() * 3);
    for(int i = 0; i < task.getWidth() * task.getHeight() * 3; i++)
        result[i] = frame[i];
    delete pScene;

    double seconds = difftime(timer, time(0));
    secondsWorked += seconds;
    std::cout << "::: Task finished on " << GetTimeStamp() << std::endl;
    std::cout << "::: This worker has worked " << secondsWorked << " seconds, so far" << std::endl;

    return NULL;
}

// Sends the result back to the master node
void Worker::SendResults(const Task &task)
{
    byte arrayResult[task.getWidth() * task.getHeight() * 3];

    for(int i = 0; i < task.getWidth() * task.getHeight() * 3; i++)
        arrayResult[i] = result[i];

    // First send the invariably sized part
    client.Send( (byte*) &arrayResult, task.getWidth() * task.getHeight() * 3 * sizeof(char));

    /*
    std::string fileName("Test.ppm");
    std::stringstream resultS( std::stringstream::out | std::stringstream::binary);

    resultS << "P6\n" << task.getWidth() << " " << task.getHeight() << "\n255\n";
    for(int i = 0; i < task.getWidth() * task.getHeight() * 3; i++)
        resultS << result[i];

    std::fstream file(fileName);
    if(!file.is_open())
        return;
    else {
        file << resultS.str();
        file.close();
    }*/
    return;
}

std::string Worker::GetTimeStamp() {
    time_t rawTime;
    tm* timeInfo;

    time(&rawTime);
    timeInfo = localtime(&rawTime);
    return std::string(asctime(timeInfo));
}
