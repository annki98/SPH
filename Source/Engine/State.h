//
// Created by maxbr on 26.05.2020.
//

#pragma  once

#include "Defs.h"
#include "Camera.h"

class State
{
public:
    State();

    State(size_t width, size_t height);

    ~State();

    std::shared_ptr<Camera> getCamera() const;

    void setCamera(std::shared_ptr<Camera> mCamera);

    size_t getWidth() const;

    size_t getHeight() const;

    double getDeltaTime() const;

    void setDeltaTime(double mDeltaTime);

    double getTime() const;

    void setTime(double mTime);

private:
    std::shared_ptr<Camera> m_camera;
    size_t m_width, m_height;
    double m_deltaTime;
    double m_time;

};