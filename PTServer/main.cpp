#include "server.h"
#include <iostream>
#include <unistd.h>
#include <string>

int main(int argc, char **argv) {
    std::cout << "!! Welcome to the path tracer master node !!" << std::endl;

    // Take port from cli argument, 12345 as default
    std::string port = "12345";
    if (argc >= 2) {
        port = argv[1];
    }

    // Start accepting connections
    Server server(port);
    if (!server.StartListening())
        return -1;
    server.StartAcceptingConnections();

    // Keep checking whether we received data
    while (true) {
        if (server.HasData()) {
            RcvData rd = server.GetData();

            // Do stuff with data
            std::cout << "Received data from client " << rd.fd << ": " << rd.data << std::endl;

            delete[] rd.data; // Really need to not forget this

            usleep(100000); // Sleep 100ms (100000 microseconds)
        }
    }

    server.StopAcceptingConnections();
}
