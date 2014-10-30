#include "task.h"

Task::Task()
{
}

int Task::getFrame() const
{
    return frame;
}

void Task::setFrame(int value)
{
    frame = value;
}
int Task::getJobId() const
{
    return jobId;
}

void Task::setJobId(int value)
{
    jobId = value;
}
int Task::getWidth() const
{
    return width;
}

void Task::setWidth(int value)
{
    width = value;
}
int Task::getHeight() const
{
    return height;
}

void Task::setHeight(int value)
{
    height = value;
}
Point Task::getCameraPosition() const
{
    return cameraPosition;
}

void Task::setCameraPosition(const Point &value)
{
    cameraPosition = value;
}
Vector Task::getCameraOrientation() const
{
    return cameraOrientation;
}

void Task::setCameraOrientation(const Vector &value)
{
    cameraOrientation = value;
}
int Task::getNSamples() const
{
    return nSamples;
}

void Task::setNSamples(int value)
{
    nSamples = value;
}
std::string Task::getWorldToRender() const
{
    return worldToRender;
}

void Task::setWorldToRender(const std::string &value)
{
    worldToRender = value;
}








