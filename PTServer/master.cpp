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
        jobs.insert(record);
    }
}

// Assigns tasks to all workers
void Master::AssignTasks() {
    // Retrieve all the workers
    std::vector<int> workers = server.GetConnections();
    if (workers.empty())
        return;

    // If no jobs are pending, return
    // TODO

    // Assign tasks to the workers
    for (std::vector<int>::iterator i = workers.begin(); i != workers.end(); i++) {
        // Check if *i is a new worker
        std::map<int,std::string>::iterator idleIt = idleList.find(*i);
        if(idleIt == idleList.end()) {
            idleList.insert( std::pair<int, std::string>(*i, "idle"));
            //workerTaskMap.insert( std::pair< int, std::pair<int,int>>(*i, std::pair<int, int>(0,0)  ));
        }

        // Check if *i is idling
        idleIt = idleList.find(*i);
        if(idleIt != idleList.end()) {
            if(idleIt->second == "idle") {
                if(jobIt == jobs.end()) {
                    jobIt = jobs.begin();
                }

                taskList::iterator taskIt;
                for(taskIt = jobIt->second.begin(); taskIt != jobIt->second.end(); taskIt++) {
                    statusList::iterator statusIt = status.find( std::pair<int,int>(jobIt->first, taskIt->getFrame() ));
                    if(statusIt != status.end()) {
                        if( statusIt->second == "pending") {

                            std::map<int, Task>::iterator workerTaskMapIt = workerTaskMap.find(*i);
                            if( workerTaskMapIt != workerTaskMap.end() ) {
                                workerTaskMapIt->second = *taskIt;
                            } else {
                                workerTaskMap.insert(std::pair<int, Task>(*i, *taskIt));
                            }

                            // TODO QUEUE for JOBS
                            /*std::map<int, Task>::iterator workerTaskMapIt = workerTaskMap.find(*i);
                            if( workerTaskMapIt != workerTaskMap.end() ) {
                                workerTaskMapIt->second = *taskIt;
                            } else {
                                workerTaskMap.insert(std::pair<int, Task>(*i, *taskIt));
                            }*/

                            std::cout << "::: Assigning Task " << (*taskIt).getJobId() << "_" << (*taskIt).getFrame() << " to worker " << *i << " :::\n" << std::endl;
                            SendTask(*i, *taskIt);
                            statusIt->second = "claimed";

                            break;
                        }
                    }
                }


                idleIt->second = "working";
                jobIt++;
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
            std::cout << "GAAT FOUT" << std::endl;
            return;
        }
        else {
            file << resultS.str();
            file.close();
        }



        std::cout << "::: Received results of task " << task.getJobId() << "_" << task.getFrame() << " from worker " << rd.fd  << " :::"<< std::endl;

        // Check if *i is idling
        std::map<int,std::string>::iterator idleIt = idleList.find(rd.fd);
        if(idleIt != idleList.end())
            idleIt->second = "idle";
        // TODO: Do stuff with the results
    }
}
