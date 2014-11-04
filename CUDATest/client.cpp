#include "client.h"

#include <sys/socket.h>
#include <netdb.h>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <arpa/inet.h>
#include <errno.h>

Client::Client() :
    fd(-1),
    mps(1024) {
}

Client::~Client() {
    Disconnect();
}

// Connects the client to server at address:port (creates a socket first)
bool Client::Connect(const std::string& address, const std::string& port) {
    addrinfo hostInfo;
    addrinfo *result;
    memset(&hostInfo, 0, sizeof(hostInfo));
    hostInfo.ai_family = AF_UNSPEC;
    hostInfo.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(address.c_str(), port.c_str(), &hostInfo, &result) != 0) {
        std::cerr << "Client: Error retrieving address info for " << address << ":" << port << std::endl;
        std::cerr << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (fd == -1) {
        std::cerr << "Client: Error creating socket: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    std::cout << "Client: Connecting to the server..." << std::endl;
    if (connect(fd, result->ai_addr, result->ai_addrlen) == -1) {
        std::cerr << "Error connecting to server: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    freeaddrinfo(result);

    std::cout << "Client: Connected to server" << std::endl;

    return true;
}

// Disconnects the client from the server
void Client::Disconnect() {
    if (fd != -1)
        close(fd);
}

// Send data of given size to server
bool Client::Send(const char* data, int size) {
    // First let the server know how many bytes to expect
    if (send(fd, (void*)&size, sizeof(int), 0) == -1) {
        std::cerr << "Client: Failed to send data to server: " <<strerror(errno) << std::endl;
        return false;
    }

    // Divide data into chunks and send these chunks
    const int chunkSize = 8;
    int bytesSent = 0;
    while (bytesSent < size) {
        // Send chunkSize bytes, unless we can send less
        int bytesToSend = std::min(size - bytesSent, chunkSize);

        int sent = send(fd, data + bytesSent, bytesToSend, 0);
        if (sent == -1) {
            std::cerr << "Client: Failed to send data to server: " <<strerror(errno) << std::endl;
            return false;
        } else {
            bytesSent += sent;
        }
    }

    return true;
}

// Receive data from server and process it
// Note: This function is blocking
// Returns false when server disconnects
bool Client::Receive() {
    char buffer[mps];
    int bufferLength = read(fd, buffer, sizeof(buffer));

    if (bufferLength <= 0) {
        std::cout << "Client: Server disconnected" << std::endl;
        return false;
    }

    std::cout << "Client: Received data: " << buffer << std::endl;

    // TODO: Actually do stuff with received data

    return true;
}
