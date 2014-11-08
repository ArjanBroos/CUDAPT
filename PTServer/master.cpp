#include "master.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdio.h>

Master::Master()
    : nextJobId(0)
{
    jobIt = jobs.begin();
}

// Start listening for messages from worker nodes
// Returns true if no errors occurred
bool Master::StartListening(const std::string &port) {
    if (!server.StartListening(port))
        return false;

    server.StartAcceptingConnections();

    return true;
}

// Receive Jobs from external service
void Master::ReceiveJob()
{   
    std::string jobFileName("newJob.job");
    std::ifstream jobDescriptionFile(jobFileName);

    if( jobDescriptionFile.good() ) {
        jobDescriptionFile.close();
    } else {
        jobDescriptionFile.close();
        return;
    }

    int jobId = nextJobId++;

    // Create a dummy task
    Task task;
    task.setJobId(jobId);
    task.setWidth(640);
    task.setHeight(400);
    task.setNSamples(10);
    task.setCameraPosition(Point(10.f, 1.2f, 3.f));
    task.setCameraOrientation(Vector(0.03f, -0.36f, -0.9f));
    task.setWorldToRender("movieWorld");

    taskList job;
    for(int i = 0; i < 6; i++) {
        task.setFrame(i);
        job.push_back(task);
        statusRecord record(std::pair<int, int>(jobId,i) , std::string("pending"));
        status.insert(record);
    }
    jobRecord record(jobId, job);
    workersPerJob.insert(std::pair<int, int>(jobId, 0));
    jobs.insert(record);
    jobQueue.push_back(jobId);

    if( remove(jobFileName.c_str()) != 0 )
        std::cout << "\n\t!!! Couldn't remove job file !!!\n" << std::endl;
    else
        std::cout << "\n\t!!! Added new job: Job " << task.getJobId() << " !!!\n" << std::endl;
}

// Assigns tasks to all workers
void Master::AssignTasks() {
    // Set mappings of new workers and reset abandonned tasks
    updateMappingsAndRespawnTasks();

    // Retrieve all the workers
    std::vector<int> workers = server.GetConnections();

    // If no jobs are pending, just return
    if(jobs.size() == 0)
        return;

    // Assign tasks to the workers
    for (auto i = workers.begin(); i != workers.end(); i++) {
        // Check if *i is idling, jump to next worker otherwise
        auto idleIt = idleList.find(*i);
        if(idleIt != idleList.end())
            if(idleIt->second != "idle")
                continue;

        // point jobIt to the first job in the queue
        jobQueueIt = jobQueue.begin();
        jobIt = jobs.find(*jobQueueIt);

        // Do not let a worker start on a job with already the max number of workers
        while(workersPerJob[jobIt->first] >= 2) {
            jobQueueIt++;
            if(jobQueueIt == jobQueue.end())
                return;
            jobIt = jobs.find(*jobQueueIt);
        }

        // Find a pending task in the job
        for(auto taskIt = jobIt->second.begin(); taskIt != jobIt->second.end(); taskIt++) {
            // Check if the task is not already claimed
            auto statusIt = status.find( std::pair<int,int>(taskIt->getJobId(), taskIt->getFrame() ));
            if(statusIt != status.end()) {
                if( statusIt->second == "pending") {
                    // Update the mapping, *i is going to work on *taskIt
                    auto workerTaskMapIt = workerTaskMap.find(*i);
                    if( workerTaskMapIt != workerTaskMap.end() )
                        workerTaskMapIt->second = *taskIt;

                    // TODO QUEUE for JOBS
                    workersPerJob[taskIt->getJobId()]++;

                    std::cout << "::: Assigning Task " << taskIt->getJobId() << "_" << taskIt->getFrame() << " to worker " << *i << " :::\n" << std::endl;
                    SendTask(*i, *taskIt);
                    statusIt->second = "claimed";
                    idleIt->second = "working";

                    // Move job to the end of the queue
                    auto queueRemoveIt = *jobQueueIt;
                    jobQueue.erase(jobQueueIt);
                    jobQueue.push_back(queueRemoveIt);
                    break;
                }
            }
        }
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

// Updates the worker mappings and restarts abandonned tasks
void Master::updateMappingsAndRespawnTasks()
{
    // Reset abandonned tasks
    std::vector<int> &recentlyDisconnected = server.GetRecentlyDisconnectedClients();
    // For each disconnected client
    for(auto client = recentlyDisconnected.begin(); client != recentlyDisconnected.end(); client++) {
        std::cout << "=== Client " << *client << " disconnected ===\n";
        // Find the task that the client was working on
        auto taskIt = workerTaskMap.find(*client);
        if(taskIt == workerTaskMap.end()) break;

        // Find the status of that task and reset it to pending so it can start again
        auto statusIt = status.find( std::pair<int,int>(taskIt->second.getJobId(), taskIt->second.getFrame() ));
        if(statusIt == status.end()) break;

        // Set task to pending, decrease workers of the job and notify system
        statusIt->second = "pending";
        workersPerJob[taskIt->second.getJobId()]--;
        std::cout << "=== Restarted task " << taskIt->second.getJobId() << "_" << taskIt->second.getFrame() << " ===" << std::endl;
    }
    server.ClearRecentlyDisconnected();

    // Fix mappings voor newly connected clients
    std::vector<int> &recentlyConnected = server.GetRecentlyConnectedClients();
    for(auto client = recentlyConnected.begin(); client != recentlyConnected.end(); client++) {
        std::cout << "=== Client " << *client << " connected ===" << std::endl;
        auto idleIt = idleList.find(*client);
        if(idleIt == idleList.end())
            idleList.insert( std::pair<int, std::string>(*client, "idle"));
        else
            idleIt->second = "idle";

        auto workerIt = workerTaskMap.find(*client);
        if( workerIt == workerTaskMap.end())
            workerTaskMap.insert( std::pair<int, Task>(*client, Task()));
        else
            workerIt->second = Task();
    }
    server.ClearRecentlyConnected();
}

// Handles results that workers have finished and sent in
void Master::HandleResults() {
    while (server.HasData()) {
        RcvData rd = server.PopData();

        // Retrieve the task of worker rd.fd
        auto workerTaskMapIt = workerTaskMap.find(rd.fd);
        if( workerTaskMapIt != workerTaskMap.end() ) {
        }
        Task &task = workerTaskMapIt->second;

        // Retrieve width and height (resolution) for the size of the pixel data
        int width = task.getWidth();
        int height = task.getHeight();

        // If the size is correct, retrieve the data and store it in a dedicated array
        if (rd.size != width*height*3*sizeof(char)) {
            std::cout << "Error, expected different size in ReceiveTask" << std::endl;
        }
        byte result[width*height*3];
        byte *intIter = (byte*)rd.data;
        for(int i = 0; i < width*height*3; i++) {
            result[i] = *intIter;
            intIter++;
        }

        // Notify the system that the task results are received
        std::cout << "::: Received results of task " << task.getJobId() << "_" << task.getFrame() << " from worker " << rd.fd  << " :::" << std::endl;

        // Worker is done with the task, decrease the number of workers on the job belonging to the task
        workersPerJob[task.getJobId()]--;

        // Set *i to idling
        auto idleIt = idleList.find(rd.fd);
        if(idleIt != idleList.end())
            idleIt->second = "idle";

        // Write received data to .ppm file
        std::stringstream fileName;
        fileName << task.getJobId() << "_" << task.getFrame() << ".ppm";
        std::stringstream resultS( std::stringstream::out | std::stringstream::binary);

        resultS << "P6\n" << width << " " << height << "\n255\n";
        for(int i = 0; i < width * height * 3; i++)
            resultS << result[i];

        std::ofstream file(fileName.str(), std::ios::out);
        if(file.is_open()) {
            file << resultS.str();
            file.close();
            std::cout << "\t::: Results stored in .ppm file :::" << std::endl;
        } else {
            std::cout << "\t!!! Could not write to .ppm file !!!" << std::endl;
            return;
        }

        // Remove the task
        auto statusIt = status.find( std::pair<int,int>(task.getJobId(), task.getFrame() ));
        if( statusIt != status.end())
            status.erase(statusIt);

        // Check if all tasks of the job are finished, if so, remove the job
        auto jobIt2 = jobs.find(task.getJobId());
        bool jobFinished = true;
        // Check for each task of the job the status, if job is empty (finished) => remove the job from the job list
        for(auto taskIt = jobIt2->second.begin(); taskIt != jobIt2->second.end(); taskIt++) {
            auto statusIt = status.find( std::pair<int,int>(jobIt2->first, taskIt->getFrame() ));
            if(statusIt != status.end()) {
                jobFinished = false;
                break;
            }
        }
        // If job is finished, notify the system and remove the job from the system
        if(jobFinished) {
            std::cout << "\n\t### Job " << jobIt2->first << " done ###\n" << std::endl;
            jobs.erase(jobIt2);
            auto queueRemoveIt = std::find(jobQueue.begin(), jobQueue.end(), jobIt2->first);
            if(queueRemoveIt != jobQueue.end())
                jobQueue.erase(queueRemoveIt);
        }
    }
}
