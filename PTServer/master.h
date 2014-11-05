#ifndef MASTER_H
#define MASTER_H

#include "server.h"
#include "task.h"
#include <string>

// Splits jobs into tasks and sends those off to worker nodes
// Receives the results from the worker nodes and saves them to disk
class Master {
public:
    Master();

    // Start listening for messages from worker nodes
    bool StartListening(const std::string &port);

    // Assigns tasks to all workers
    void AssignTasks();

    // Handles results that workers have finished and sent in
    void HandleResults();

private:
    // Sends task to worker with given file descriptor
    void SendTask(int fd, Task &task);

    Server server;
};

#endif // MASTER_H
