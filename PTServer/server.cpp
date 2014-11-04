#include "server.h"

#include <sys/socket.h>
#include <netdb.h>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>

Server::Server() :
    port("12345"),
    fd(-1),
    backlog(24),
    connectionThread(-1),
    fdMutex(PTHREAD_MUTEX_INITIALIZER),
    dataMutex(PTHREAD_MUTEX_INITIALIZER) {
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
        std::cerr << "Server: Error when setting up server socket: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    // Check if socket is free
    int yes = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) < 0) {
        std::cerr << "Server: Error, address for server socket already in use" << std::endl;
        freeaddrinfo(result);
        return false;
    }

    // Bind to this socket
    if (bind(fd, result->ai_addr, result->ai_addrlen) != 0) {
        std::cerr << "Error binding socket: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    // Actually start listening
    if (listen(fd, backlog) < 0) {
        std::cerr << "Server: Error when server is trying to listen on socket: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    freeaddrinfo(result);

    std::cout << "Server: Started listening" << std::endl;

    return true;
}

// Spawns a thread to start accepting connections
void Server::StartAcceptingConnections() {
    if (connectionThread == -1) {
        establishingConnections = true;
        pthread_create(&connectionThread, NULL, KeepEstablishingConnections, this);
        std::cout << "Server: Started accepting connections" << std::endl;
    } else {
        std::cout << "Server: Was already accepting connections" << std::endl;
    }
}

// Stops accepting connections
void Server::StopAcceptingConnections() {
    if (connectionThread != -1) {
        establishingConnections = false;
        pthread_join(connectionThread, NULL);
        connectionThread = -1;

        std::cout << "Server: Stopped accepting connections" << std::endl;
    } else {
        std::cout << "Server: Was not accepting any connections to stop" << std::endl;
    }
}

// Will constantly try to establish connections
void* Server::KeepEstablishingConnections(void* context) {
    Server* server = (Server*)context;
    while (server->establishingConnections) {
        server->EstablishConnection();
    }
}

// When a client tries to connect, we can accept the connection with this function
void Server::EstablishConnection() {
    // Accept the connection
    sockaddr_storage clientInfo;
    memset(&clientInfo, 0, sizeof(clientInfo));
    socklen_t size = sizeof(clientInfo);
    int newFD = accept(fd, (sockaddr*)&clientInfo, &size);
    if (newFD < 0) {
        std::cerr << "Server: Error, accepting connection failed: " << strerror(errno) << std::endl;
        return;
    }

    // Retrieve ip address and port of client to show them
    sockaddr_in *s = (sockaddr_in*)&clientInfo;
    int port = ntohs(s->sin_port);
    char ipStr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ipStr, sizeof(ipStr));
    std::cout << "Server: Connection accepted from " << ipStr << " using port " << port << std::endl;

    // Create a thread to receive data from this client
    pthread_mutex_lock(&fdMutex);
    RcvParam* rp = new RcvParam; rp->fd = newFD; rp->context = this;
    pthread_t thread;
    pthread_create(&thread, NULL, Receive, rp);
    pthread_detach(thread);

    // Save file descriptor for socket of new client and thread handle
    clientFDs.insert(std::pair<int, pthread_t>(newFD, thread));
    pthread_mutex_unlock(&fdMutex);
}

// Send data to worker node with given file descriptor
bool Send(int fd, void* data, int size) {
    if (send(fd, data, size, 0) != size) {
        std::cerr << "Server: Failed to send data to " << fd << std::endl;
        return false;
    }
    return true;
}

// Receive data from worker node with given file descriptor
void Server::ActualReceive(RcvParam* rp) {
    // First receive an integer, telling us the size of the data in bytes
    int size = -1;
    int ret = recv(rp->fd, (void*)&size, sizeof(int), 0);
    HandleDisconnect(rp, ret);
    HandleReceiveErrors(rp, ret);

    // Receive the data in chunks
    const int chunkSize = 8;
    char* data = new char[size];
    int bytesReceived = 0;
    while (bytesReceived < size) {
        // Receive chunkSize bytes, unless we can receive less
        int bytesToReceive = std::min(size - bytesReceived, chunkSize);
        int received = recv(rp->fd, data + bytesReceived, bytesToReceive, 0);
        HandleReceiveErrors(rp, received);
        bytesReceived += received;
    }

    // Store received data in list
    RcvData rd; rd.fd = fd; rd.data = data; rd.size = size;
    //std::cout << "Server: Received data: " << data << std::endl;

    pthread_mutex_lock(&dataMutex);
    receivedData.push_back(rd);
    pthread_mutex_unlock(&dataMutex);
}

// Wrapper around ActualReceive() function, compatible with pthread
void* Server::Receive(void *rcvParam) {
    RcvParam* rp = (RcvParam*)rcvParam;
    rp->context->ActualReceive(rp);
}

void Server::CleanUpThread(RcvParam* rp) {
    pthread_mutex_lock(&fdMutex);
    clientFDs.erase(rp->fd);
    pthread_mutex_unlock(&fdMutex);

    close(rp->fd);
    delete rp;
    pthread_exit(NULL);
}

void Server::HandleDisconnect(RcvParam* rp, int ret) {
    if (ret == 0) {
        std::cout << "Server: Client disconnected. Removing file descriptor." << std::endl;
        CleanUpThread(rp);
    }
}

// Handles disconnection by client or error when trying to receive data
void Server::HandleReceiveErrors(RcvParam* rp, int ret) {
    if (ret < 0) {
        std::cout << "Server: Error receiving: " << strerror(errno) << std::endl;
        CleanUpThread(rp);
    }
}
