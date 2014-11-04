#include "server.h"
#include <iostream>
#include <unistd.h>

int main() {
    Server server;
    if (!server.StartListening())
        return -1;
    server.StartAcceptingConnections();

    while (true) {
        if (server.HasData()) {
            RcvData rd = server.GetData();

            // Do stuff with data
            std::cout << "Received data: " << rd.data << std::endl;

            delete[] rd.data; // Really need to not forget this

            usleep(20000); // Sleep 20ms (20000 microseconds)
        }
    }

    server.StopAcceptingConnections();
}
