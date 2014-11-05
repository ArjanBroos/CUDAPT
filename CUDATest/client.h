#ifndef CLIENT_H
#define CLIENT_H

#include <string>

typedef char byte;

// Used to bundle received data and its metadata
struct RcvData {
    int fd;
    byte *data;
    int size;
};

// Client part of the communication between workers and master nodes
class Client {
public:
    Client();
    ~Client();

    // Connects the client to server at address:port (creates a socket first)
    bool Connect(const std::string &address, const std::string &port);

    // Returns true if the client is connected to the server
    bool IsConnected() const;

    // Disconnects the client from the server
    void Disconnect();

    // Send data of given size to server
    bool Send(const byte *data, int size);

    // Receive data from server and store it in 'receivedData'
    // Note: This function is blocking
    // Returns false when server disconnects
    bool Receive(RcvData &receivedData);

private:
    // Returns false if server disconnected or an error occured
    bool HandleDisconnectOrError(int ret);

    int fd;     // This client's socket's file descriptor
    int chunkSize; // Size of the chunks data is divided into before sending/receiving
};

#endif // CLIENT_H
