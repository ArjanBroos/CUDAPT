#ifndef WORKER_H
#define WORKER_H

#include "task.h"
#include "client.h"
#include <string>

// Accepts tasks from the master node, processes them and sends back results
class Worker {
public:
    Worker();

    // Connect to master node
    bool Connect(const std::string &address, const std::string &port);

    // Recieve a task from the master node
    Task ReceiveTask();

    // Performs the given task (rendering an image with path tracer)
    byte *PerformTask(const Task &task);

private:
    Client client;
};

#endif // WORKER_H