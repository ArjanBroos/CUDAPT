#ifndef MASTER_H
#define MASTER_H

#include "server.h"
#include "task.h"
#include <string>
#include <utility>
#include <map>

// Typedefs for a job and a list of jobs
typedef std::vector<Task> taskList;
typedef std::map<int, taskList> jobList;
typedef std::pair<int, taskList> jobRecord;
typedef std::map<std::pair<int, int>, std::string> statusList;
typedef std::pair<std::pair<int, int>, std::string> statusRecord;

// Splits jobs into tasks and sends those off to worker nodes
// Receives the results from the worker nodes and saves them to disk
class Master {
public:
    Master();

    // Start listening for messages from worker nodes
    bool StartListening(const std::string &port);

    // Receive Jobs from external service
    void ReceiveJob();

    // Assigns tasks to all workers
    void AssignTasks();

    // Handles results that workers have finished and sent in
    void HandleResults();

private:
    // Sends task to worker with given file descriptor
    void SendTask(int fd, Task &task);

    // Updates the worker mappings and restarts abandonned tasks
    void updateMappingsAndRespawnTasks();

    statusList status;
    int nextJobId;
    jobList jobs;
    std::vector<int> jobQueue;
    jobList::iterator jobIt;
    std::vector<int>::iterator jobQueueIt;

    std::map<int, std::string> idleList;
    std::map<int, Task> workerTaskMap;
    std::map<int, int> workersPerJob;

    Server server;
};

#endif // MASTER_H
