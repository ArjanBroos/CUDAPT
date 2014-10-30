#ifndef TASK_H
#define TASK_H

#include "point.h"
#include "vector.h"

class Task
{
public:
    Task();

    int getFrame() const;
    void setFrame(int value);

    int getJobId() const;
    void setJobId(int value);

    int getWidth() const;
    void setWidth(int value);

    int getHeight() const;
    void setHeight(int value);

    Point getCameraPosition() const;
    void setCameraPosition(const Point &value);

    Vector getCameraOrientation() const;
    void setCameraOrientation(const Vector &value);

    int getNSamples() const;
    void setNSamples(int value);

    std::string getWorldToRender() const;
    void setWorldToRender(const std::string &value);

private:
    int jobId;
    int width;
    int height;
    int frame;
    int nSamples;
    Point cameraPosition;
    Vector cameraOrientation;
    std::string worldToRender;
};

#endif // TASK_H
