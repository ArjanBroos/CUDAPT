#include "master.h"
#include <iostream>

Master::Master() {
}

// Start listening for messages from worker nodes
// Returns true if no errors occurred
bool Master::StartListening(const std::string &port) {
    if (!server.StartListening(port))
        return false;

    server.StartAcceptingConnections();

    return true;
}

// Assigns tasks to all workers
void Master::AssignTasks() {
    // Retrieve all the workers
    std::vector<int> workers = server.GetConnections();
    if (workers.empty())
        return;

    // Create a dummy task
    Task task;
    task.setJobId(42);
    task.setWidth(800);
    task.setHeight(600);
    task.setFrame(3);
    task.setNSamples(32);
    task.setCameraPosition(Point(0.f, 0.f, 0.f));
    task.setCameraOrientation(Vector(0.f, 0.f, -1.f));
    task.setWorldToRender("funnyguy");

    // Assign tasks to the workers
    std::cout << "::: Assigning tasks to " << workers.size() << " workers" << std::endl;
    for (std::vector<int>::iterator i = workers.begin(); i != workers.end(); i++) {
        SendTask(*i, task);
    }
}

// Sends task to worker with given file descriptor
void Master::SendTask(int fd, Task &task) {
    //std::cout << "::: Sending task to worker node" << std::endl;

    // First send the invariably sized part
    server.Send(fd, (byte*)&task, 5 * sizeof(int) + sizeof(Point) + sizeof(Vector));

    // Then send the world name
    server.Send(fd, task.getWorldToRender().c_str(), task.getWorldToRender().size());
}

// Handles results that workers have finished and sent in
void Master::HandleResults() {
    while (server.HasData()) {
        RcvData rd = server.PopData();

        std::cout << "::: Received results" << std::endl;
        // TODO: Do stuff with the results
    }
}
