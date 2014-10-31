#include "server.h"

#include <sys/socket.h>
#include <netdb.h>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>

const int Server::mps = 1024;

Server::Server() :
    port("12345"),
    fd(-1),
    backlog(24),
    connectionThread(nullptr) {
}

// Returns true when the server has some unprocessed received data stored
bool Server::HasData() const {
    return !receivedData.empty();
}

// Returns received data
// In FIFO order
// !! BE SURE TO CALL DELETE ON THE ACTUAL DATA WHEN DONE !!
RcvData Server::GetData() {
    RcvData rd = receivedData.front();
    receivedData.pop_front();
    return rd;
}

// Start listening for incoming connections
// Returns true when succesfully started listening
bool Server::StartListening() {
    // Set up address info
    addrinfo hostinfo, *result;
    memset(&hostinfo, 0, sizeof(hostinfo));
    hostinfo.ai_family = AF_INET;       // IPv4
    hostinfo.ai_socktype = SOCK_STREAM; // UDP?
    hostinfo.ai_flags = AI_PASSIVE;     // Fill in my IP for me
    getaddrinfo(NULL, port, &hostinfo, &result);

    // Create socket
    fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (fd < 0) {
        std::cerr << "Error when setting up server socket" << std::endl;
        return false;
    }

    // Check if socket is free
    int yes = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) < 0) {
        std::cerr << "Error, address for server socket already in use" << std::endl;
        return false;
    }

    // Bind to this socket
    if (bind(fd, result->ai_addr, result->ai_addrlen) != 0) {
        std::cerr << "Error, " << strerror(errno) << std::endl;
        return false;
    }

    // Actually start listening
    if (listen(fd, backlog) < 0) {
        std::cerr << "Error when server is trying to listen on socket" << std::endl;
        return false;
    }

    std::cout << "Started listening" << std::endl;

    return true;
}

// Spawns a thread to start accepting connections
void Server::StartAcceptingConnections() {
    RcvParam rp; rp.fd = fd; rp.context = this;
    if (connectionThread == nullptr)
        pthread_create(&connectionThread, NULL, KeepEstablishingConnections, &rp);

    std::cout << "Started accepting connections" << std::endl;
}

// Stops accepting connections
void Server::StopAcceptingConnections() {
    if (connectionThread == nullptr)
        return;
    pthread_join(connectionThread, NULL);
    connectionThread = nullptr;

    std::cout << "Stopped accepting connections" << std::endl;
}

// Will constantly try to establish connections
void* Server::KeepEstablishingConnections(void* rp) {
    RcvParam *rcvParam = (RcvParam*)rp;
    while (true) {
        rcvParam->context->EstablishConnection();
    }
}

// When a client tries to connect, we can accept the connection with this function
void Server::EstablishConnection() {
    // Accept the connection
    sockaddr_storage remoteInfo;
    socklen_t addrSize = sizeof(socklen_t);
    int newFD = accept(fd, (sockaddr*)&remoteInfo, &addrSize);
    if (newFD < 0) {
        std::cerr << "Error, accepting connection failed" << std::endl;
        return;
    }

    // Retrieve ip address and port of client to show them
    sockaddr_in *s = (sockaddr_in*)&remoteInfo;
    int port = ntohs(s->sin_port);
    char ipStr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ipStr, sizeof(ipStr));
    std::cout << "Connection accepted from " << ipStr << " using port " << port << std::endl;

    // Create a thread to receive data from this client
    pthread_mutex_lock(&fdMutex);
    RcvParam rp; rp.fd = newFD; rp.context = this;
    pthread_t thread;
    pthread_create(&thread, NULL, Receive, &rp);

    // Save file descriptor for socket of new client and thread handle
    clientFDs.insert(std::pair<int, pthread_t>(newFD, thread));
    pthread_mutex_unlock(&fdMutex);
}

// Send data to worker node with given file descriptor
bool Send(int fd, void* data, int size) {
    if (send(fd, data, size, 0) != size) {
        std::cerr << "Failed to send data to " << fd << std::endl;
        return false;
    }
    return true;
}

// Receive data from worker node with given file descriptor
void Server::ActualReceive(int fd) {
    char *buffer = new char[mps];   // Buffer to hold received data

    while (true) {
        int bufferLength = read(fd, buffer, sizeof(buffer));

        if (bufferLength <= 0) {
            std::cout << "Client disconnected. Removing file descriptor." << std::endl;
            pthread_mutex_lock(&fdMutex);
            pthread_join(clientFDs.at(fd), NULL);   // Shut down the thread for receiving from this client as well
            clientFDs.erase(fd);
            pthread_mutex_unlock(&fdMutex);
            close(fd);
            pthread_exit(NULL);
        }

        // Store received data in list
        RcvData rd;
        rd.fd = fd;
        rd.data = buffer;
        rd.size = bufferLength;
        pthread_mutex_lock(&dataMutex);
        receivedData.push_back(rd);
        pthread_mutex_unlock(&dataMutex);
    }
}

// Wrapper around ActualReceive() function, compatible with pthread
void* Server::Receive(void *rcvParam) {
    RcvParam* rp = (RcvParam*)rcvParam;
    rp->context->ActualReceive(rp->fd);
}
