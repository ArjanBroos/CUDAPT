#ifndef CLIENT_H
#define CLIENT_H

#include <string>

// Client part of the communication between workers and master nodes
class Client {
public:
    Client();
    ~Client();

    // Connects the client to server at address:port (creates a socket first)
    bool Connect(const std::string& address, const std::string& port);

    // Disconnects the client from the server
    void Disconnect();

    // Send data of given size to server
    bool Send(void* data, int size);

    // Receive data from server and process it
    // Note: This function is blocking
    // Returns false when server disconnects
    bool Receive();

private:
    int fd;     // This client's socket's file descriptor
    int mps;    // Maximum packet size
};

#endif // CLIENT_H
