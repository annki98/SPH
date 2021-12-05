//
// Created by maxbr on 26.05.2020.
//

#include "State.h"

State::State()
{

}

State::State(size_t width, size_t height)
{
    m_width = width;
    m_height = height;

    m_camera = std::make_shared<Camera>(m_width, m_height);
}

State::~State()
{

}

std::shared_ptr<Camera> State::getCamera() const
{
    return m_camera;
}

void State::setCamera(std::shared_ptr<Camera> mCamera)
{
    m_camera = mCamera;
}

size_t State::getWidth() const
{
    return m_width;
}

size_t State::getHeight() const
{
    return m_height;
}

double State::getDeltaTime() const
{
    return m_deltaTime;
}

void State::setDeltaTime(double mDeltaTime)
{
    m_deltaTime = mDeltaTime;
}

double State::getTime() const
{
    return m_time;
}

void State::setTime(double mTime)
{
    m_time = mTime;
}
