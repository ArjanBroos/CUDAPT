#include "master.h"
#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    std::cout << "!! Welcome to the path tracer master node !!" << std::endl;

    // Take port from cli argument, 12345 as default
    std::string port = "12345";
    if (argc >= 2) {
        port = argv[1];
    }

    // Start the master node, listening for workers to make a connection
    Master master;
    master.StartListening(port);

    while (true) {
        // Retrieve new Jobs from external service
        master.ReceiveJob();

        // Assign tasks to the workers
        master.AssignTasks();

        // Handle results that workers have sent in
        master.HandleResults();

        sleep(1);
    }
}
