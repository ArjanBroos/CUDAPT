#ifndef SERVER_H
#define SERVER_H

#include <list>
#include <pthread.h>
#include <map>

// Used to make a list of data received from a certain node
struct RcvData {
    int fd;
    char* data;
    int size;
};

// Used to make a pthread compatible version of Receive()
class Server;
struct RcvParam {
    int fd;
    Server* context;
};

// Server class which can handle multiple incoming connections
// Used to communicate and send data between master and worker nodes
// Run on the master node, worker nodes connect to this one
class Server {
public:
    Server();

    // Start listening for incoming connections
    // Returns true when succesfully started listening
    bool StartListening();

    // Spawns a thread to start accepting connections
    void StartAcceptingConnections();
    // Stops the thread that accepts connections
    void StopAcceptingConnections();

    // Returns true when the server has some unprocessed received data stored
    bool HasData() const;
    // Returns received data
    // In FIFO order
    // !! BE SURE TO CALL DELETE ON THE ACTUAL DATA WHEN DONE !!
    RcvData GetData();

    // Send data to worker node with given file descriptor
    bool Send(int fd, void* data, int size);

private:
    // Constantly tries to establish connections
    static void* KeepEstablishingConnections(void* context);
    // When a client tries to connect, we can accept the connection with this function
    void EstablishConnection();

    // Receive data from worker node with given file descriptor
    void ActualReceive(int fd);
    // Wrapper around ActualReceive() function, compatible with pthread
    static void* Receive(void* rcvParam);

    const char *port;   // Port to listen on
    int fd;             // Socket's file descriptor

    int backlog;        // Number of connections allowed to be queued on listen()
    static const int mps;      // Maximum packet size

    pthread_t connectionThread; // Thread handle for the establishing of connections

    pthread_mutex_t fdMutex;    // Mutex to prevent race conditions on list of client fds
    pthread_mutex_t dataMutex;  // Mutex to prevent race conditions on list of received data
    std::map<int, pthread_t> clientFDs;   // Mapping from client socket file descriptor to thread handle
    std::list<RcvData> receivedData; // List of all data received from clients
};

#endif // SERVER_H
