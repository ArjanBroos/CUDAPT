#include "master.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

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
    for(int jobI = 0; jobI < 6; jobI++) {
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
    }
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
    for (std::vector<int>::iterator i = workers.begin(); i != workers.end(); i++) {
        // Check if *i is idling, jump to next worker otherwise
        std::map<int,std::string>::iterator idleIt = idleList.find(*i);
        if(idleIt != idleList.end())
            if(idleIt->second != "idle")
                continue;

        // Round Robin
        // If jobIt is at end of list, just point it at begin again
        if(jobIt == jobs.end())
            jobIt = jobs.begin();

        // Do not let a worker start on a job with already the max number of workers
        if(workersPerJob[jobIt->first] >= 2)
            break;

        // Find a pending task in the job
        for(taskList::iterator taskIt = jobIt->second.begin(); taskIt != jobIt->second.end(); taskIt++) {
            // Check if the task is not already claimed
            statusList::iterator statusIt = status.find( std::pair<int,int>(jobIt->first, taskIt->getFrame() ));
            if(statusIt != status.end()) {
                if( statusIt->second == "pending") {
                    // Update the mapping, *i is going to work on *taskIt
                    std::map<int, Task>::iterator workerTaskMapIt = workerTaskMap.find(*i);
                    if( workerTaskMapIt != workerTaskMap.end() )
                        workerTaskMapIt->second = *taskIt;

                    // TODO QUEUE for JOBS
                    workersPerJob[jobIt->first]++;

                    std::cout << "::: Assigning Task " << (*taskIt).getJobId() << "_" << (*taskIt).getFrame() << " to worker " << *i << " :::\n" << std::endl;
                    SendTask(*i, *taskIt);
                    statusIt->second = "claimed";
                    idleIt->second = "working";
                    break;
                }
            }
        }
        jobIt++;
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
    for(std::vector<int>::iterator client = recentlyDisconnected.begin(); client != recentlyDisconnected.end(); client++) {
        // Find the task that the client was working on
        std::map<int,Task>::iterator taskIt = workerTaskMap.find(*client);
        if(taskIt != workerTaskMap.end()) {
            // Find the status of that task and reset it to pending so it can start again
            statusList::iterator statusIt = status.find( std::pair<int,int>(taskIt->second.getJobId(), taskIt->second.getFrame() ));
            if(statusIt != status.end()) {
                std::cout << "=== Client " << *client << " disconnected ===\n";

                if(statusIt->second != "finished") {
                    statusIt->second = "pending";
                    workersPerJob[taskIt->second.getJobId()]--;
                    std::cout << "=== Restarted task " << taskIt->second.getJobId() << "_" << taskIt->second.getFrame() << " ===" << std::endl;
                }
            }
        }
    }
    server.ClearRecentlyDisconnected();

    // Fix mappings voor newly connected clients
    std::vector<int> &recentlyConnected = server.GetRecentlyConnectedClients();
    for(std::vector<int>::iterator client = recentlyConnected.begin(); client != recentlyConnected.end(); client++) {
        std::cout << "=== Client " << *client << " connected ===" << std::endl;
        std::map<int, std::string>::iterator idleIt = idleList.find(*client);
        if(idleIt == idleList.end())
            idleList.insert( std::pair<int, std::string>(*client, "idle"));
        else
            idleIt->second = "idle";

        std::map<int, Task>::iterator workerIt = workerTaskMap.find(*client);
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


        std::map<int, Task>::iterator workerTaskMapIt = workerTaskMap.find(rd.fd);
        if( workerTaskMapIt != workerTaskMap.end() ) {
        }
        Task &task = workerTaskMapIt->second;

        int width = task.getWidth();
        int height = task.getHeight();

        if (rd.size != width*height*3*sizeof(char)) {
            std::cout << "Error, expected different size in ReceiveTask" << std::endl;
        }
        byte result[width*height*3];
        byte *intIter = (byte*)rd.data;
        for(int i = 0; i < width*height*3; i++) {
            result[i] = *intIter;
            intIter++;
        }
        // Write file
        std::stringstream fileNameS;
        fileNameS << task.getJobId() << "_" << task.getFrame() << ".ppm";
        std::string fileName(fileNameS.str());
        std::stringstream resultS( std::stringstream::out | std::stringstream::binary);

        resultS << "P6\n" << width << " " << height << "\n255\n";
        for(int i = 0; i < width * height * 3; i++)
            resultS << result[i];

        std::ofstream file(fileName, std::ios::out);
        if(!file.is_open()) {
            return;
        }
        else {
            file << resultS.str();
            file.close();
        }

        workersPerJob[task.getJobId()]--;

        // Set *i to idling
        std::map<int,std::string>::iterator idleIt = idleList.find(rd.fd);
        if(idleIt != idleList.end())
            idleIt->second = "idle";
        std::cout << "::: Received results of task " << task.getJobId() << "_" << task.getFrame() << " from worker " << rd.fd  << " :::" << std::endl;

        // Check if the task is not already claimed
        statusList::iterator statusIt = status.find( std::pair<int,int>(task.getJobId(), task.getFrame() ));
        if( statusIt != status.end())
            statusIt->second = "finished";

        // Check if the job is already finished, if so, remove the job
        jobList::iterator jobIt2 = jobs.find(task.getJobId());
        bool allTasksFinished = true;
        // Find a pending task in the job
        for(taskList::iterator taskIt = jobIt2->second.begin(); taskIt != jobIt2->second.end(); taskIt++) {
            // Check if the task is not already claimed
            statusList::iterator statusIt = status.find( std::pair<int,int>(jobIt2->first, taskIt->getFrame() ));
            if(statusIt != status.end()) {
                if( statusIt->second != "finished")
                    allTasksFinished = false;
            }
        }
        // Check if all tasks are finished, if so, notify the system and remove the job from the joblist
        if(allTasksFinished) {
            std::cout << "::: Job " << jobIt2->first << " done :::" << std::endl;
            jobs.erase(jobIt2);
        }
    }
}
