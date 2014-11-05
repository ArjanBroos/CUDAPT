#include "worker.h"
#include <iostream>

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
    std::cout << "::: Performing task. Job ID: " << task.getJobId() << ", Frame: " << task.getFrame() << std::endl;
    std::cout << "        World name: " << task.getWorldToRender() << std::endl;
    return NULL;
}
