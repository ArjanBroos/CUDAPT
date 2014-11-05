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
    port(""),
    fd(-1),
    backlog(24),
    connectionThread(-1),
    chunkSize(32 * 1024),
    fdListMutex(PTHREAD_MUTEX_INITIALIZER),
    dataMutex(PTHREAD_MUTEX_INITIALIZER) {
}

// Returns true when the server has some unprocessed received data stored
bool Server::HasData() const {
    return !receivedData.empty();
}

// Returns earliest received data and removes it from the list
// !! BE SURE TO CALL DELETE ON THE ACTUAL DATA WHEN DONE !!
RcvData Server::PopData() {
    RcvData rd = receivedData.front();
    receivedData.pop_front();
    return rd;
}

// Start listening for incoming connections
// Returns true when succesfully started listening
bool Server::StartListening(const std::string &port) {
    this->port = port;

    // Set up address info
    addrinfo hostinfo, *result;
    memset(&hostinfo, 0, sizeof(hostinfo));
    hostinfo.ai_family = AF_INET;       // IPv4
    hostinfo.ai_socktype = SOCK_STREAM; // UDP?
    hostinfo.ai_flags = AI_PASSIVE;     // Fill in my IP for me
    getaddrinfo(NULL, port.c_str(), &hostinfo, &result);

    // Create socket
    fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (fd < 0) {
        std::cerr << "Error when setting up server socket: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    // Check if socket is free
    int yes = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) < 0) {
        std::cerr << "Error, address for server socket already in use" << std::endl;
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
        std::cerr << "Error when server is trying to listen on socket: " << strerror(errno) << std::endl;
        freeaddrinfo(result);
        return false;
    }

    freeaddrinfo(result);

    //std::cout << "Started listening" << std::endl;

    return true;
}

// Spawns a thread to start accepting connections
void Server::StartAcceptingConnections() {
    if (connectionThread == -1) {
        establishingConnections = true;
        pthread_create(&connectionThread, NULL, KeepEstablishingConnections, this);
        //std::cout << "Started accepting connections" << std::endl;
    } else {
        //std::cout << "Was already accepting connections" << std::endl;
    }
}

// Stops accepting connections
void Server::StopAcceptingConnections() {
    if (connectionThread != -1) {
        establishingConnections = false;
        pthread_join(connectionThread, NULL);
        connectionThread = -1;

        //std::cout << "Stopped accepting connections" << std::endl;
    } else {
        //std::cout << "Was not accepting any connections to stop" << std::endl;
    }
}

// Returns true if the server has open connections with clients
bool Server::HasConnections() const {
    return !clientFDs.empty();
}

// Retrieves file descriptors for all connections
std::vector<int> Server::GetConnections() {
    std::vector<int> fds;
    pthread_mutex_lock(&fdListMutex);
    for (std::map<int, pthread_t>::iterator i = clientFDs.begin(); i != clientFDs.end(); i++) {
        fds.push_back(i->first);
    }
    pthread_mutex_unlock(&fdListMutex);
    return fds;
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
        std::cerr << "Error, accepting connection failed: " << strerror(errno) << std::endl;
        return;
    }

    // Retrieve ip address and port of client to show them
    /*sockaddr_in *s = (sockaddr_in*)&clientInfo;
    int port = ntohs(s->sin_port);
    char ipStr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ipStr, sizeof(ipStr));
    std::cout << "Connection accepted from " << ipStr << " using port " << port << std::endl;
    std::cout << "    New client id is " << newFD << std::endl;*/

    // Create a thread to receive data from this client
    RcvParam* rp = new RcvParam; rp->fd = newFD; rp->context = this;
    pthread_t thread;
    pthread_create(&thread, NULL, Receive, rp);
    pthread_detach(thread);

    // Save file descriptor for socket of new client and thread handle
    pthread_mutex_lock(&fdListMutex);
    clientFDs.insert(std::pair<int, pthread_t>(newFD, thread));
    pthread_mutex_unlock(&fdListMutex);
}

// Send data to client node with given file descriptor
bool Server::Send(int fd, const byte *data, int size) {
    // First let the client know how many bytes to expect
    if (send(fd, (void*)&size, sizeof(int), 0) == -1) {
        std::cerr << "Failed to send data to client: " <<strerror(errno) << std::endl;
        return false;
    }

    // Divide data into chunks and send these chunks
    int bytesSent = 0;
    while (bytesSent < size) {
        // Send chunkSize bytes, unless we can send less
        int bytesToSend = std::min(size - bytesSent, chunkSize);

        int sent = send(fd, data + bytesSent, bytesToSend, 0);
        if (sent == -1) {
            std::cerr << "Failed to send data to client: " <<strerror(errno) << std::endl;
            return false;
        } else {
            bytesSent += sent;
        }
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
    byte *data = new byte[size];
    int bytesReceived = 0;
    while (bytesReceived < size) {
        // Receive chunkSize bytes, unless we can receive less
        int bytesToReceive = std::min(size - bytesReceived, chunkSize);
        int received = recv(rp->fd, data + bytesReceived, bytesToReceive, 0);
        HandleReceiveErrors(rp, received);
        HandleDisconnect(rp, ret);
        bytesReceived += received;
    }

    RcvData rd; rd.fd = rp->fd; rd.data = data; rd.size = size;
    //std::cout << "Received data from client << " << rp->fd << ": " << data << std::endl;

    // Store received data in list
    pthread_mutex_lock(&dataMutex);
    receivedData.push_back(rd);
    pthread_mutex_unlock(&dataMutex);
}

// Wrapper around ActualReceive() function, compatible with pthread
void* Server::Receive(void *rcvParam) {
    while (true) {
        RcvParam* rp = (RcvParam*)rcvParam;
        rp->context->ActualReceive(rp);
    }
}

void Server::CleanUpThread(RcvParam* rp) {
    pthread_mutex_lock(&fdListMutex);
    clientFDs.erase(rp->fd);
    pthread_mutex_unlock(&fdListMutex);

    close(rp->fd);
    delete rp;
    pthread_exit(NULL);
}

void Server::HandleDisconnect(RcvParam* rp, int ret) {
    if (ret == 0) {
        //std::cout << "Client " << rp->fd << " disconnected. Removing file descriptor." << std::endl;
        CleanUpThread(rp);
    }
}

// Handles disconnection by client or error when trying to receive data
void Server::HandleReceiveErrors(RcvParam* rp, int ret) {
    if (ret < 0) {
        //std::cerr << "Error receiving from client " << rp->fd << ": " << strerror(errno) << std::endl;
        CleanUpThread(rp);
    }
}
