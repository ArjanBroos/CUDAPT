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
    chunkSize(32 * 1024) {
}

Client::~Client() {
    Disconnect();
}

// Connects the client to server at address:port (creates a socket first)
bool Client::Connect(const std::string &address, const std::string &port) {
    addrinfo hostInfo;
    addrinfo *result;
    memset(&hostInfo, 0, sizeof(hostInfo));
    hostInfo.ai_family = AF_UNSPEC;
    hostInfo.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(address.c_str(), port.c_str(), &hostInfo, &result) != 0) {
        std::cerr << "Error retrieving address info for " << address << ":" << port << std::endl;
        std::cerr << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (fd == -1) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    std::cout << "Connecting to " << address << ":" << port << "..." << std::endl;
    if (connect(fd, result->ai_addr, result->ai_addrlen) == -1) {
        std::cerr << "Error connecting to server: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    freeaddrinfo(result);

    //std::cout << "Connected to server" << std::endl;

    return true;
}

// Returns true if the client is connected to the server
bool Client::IsConnected() const {
    return fd != -1;
}

// Disconnects the client from the server
void Client::Disconnect() {
    if (fd != -1) {
        close(fd);
        fd = -1;
        std::cout << "Disconnected" << std::endl;
    }
}

// Send data of given size to server
bool Client::Send(const byte *data, int size) {
    // First let the server know how many bytes to expect
    if (send(fd, (void*)&size, sizeof(int), 0) == -1) {
        std::cerr << "Failed to send data to server: " <<strerror(errno) << std::endl;
        return false;
    }

    // Divide data into chunks and send these chunks
    int bytesSent = 0;
    while (bytesSent < size) {
        // Send chunkSize bytes, unless we can send less
        int bytesToSend = std::min(size - bytesSent, chunkSize);

        int sent = send(fd, data + bytesSent, bytesToSend, 0);
        if (sent == -1) {
            std::cerr << "Failed to send data to server: " <<strerror(errno) << std::endl;
            return false;
        } else {
            bytesSent += sent;
        }
    }

    return true;
}

// Receive data from server and store it in 'receivedData'
// Note: This function is blocking
// Returns false when server disconnects
bool Client::Receive(RcvData &receivedData) {
    // First receive an integer, telling us the size of the data in bytes
    int size = -1;
    int ret = recv(fd, (void*)&size, sizeof(int), 0);
    if (!HandleDisconnectOrError(ret))
        return false;

    // Receive the data in chunks
    byte *data = new byte[size];
    int bytesReceived = 0;
    while (bytesReceived < size) {
        // Receive chunkSize bytes, unless we can receive less
        int bytesToReceive = std::min(size - bytesReceived, chunkSize);
        int received = recv(fd, data + bytesReceived, bytesToReceive, 0);
        if (!HandleDisconnectOrError(received))
            return false;
        bytesReceived += received;
    }

    receivedData.fd = fd; receivedData.data = data; receivedData.size = size;
    //std::cout << "Received data: " << data << std::endl;

    return true;
}

// Returns false if server disconnected or an error occured
bool Client::HandleDisconnectOrError(int ret) {
    if (ret == 0) {
        std::cout << "Server disconnected." << std::endl;
        Disconnect();
        return false;
    } else if (ret == -1) {
        std::cerr << "Error receiving data from server: " << strerror(errno) << std::endl;
        return false;
    } else {
        return true;
    }
}
