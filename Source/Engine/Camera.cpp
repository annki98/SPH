//
// Created by maxbr on 14.05.2020.
//

#include "Camera.h"

Camera::Camera(float width, float height)
{
    m_fov = 45.5f;
    m_projectionMatrix = glm::perspective(m_fov, width / height, 0.1f, 100.0f);

    m_cameraPos = glm::vec3(0.0f, 0.0f, 0.0f);
    m_center = glm::vec3(0.0f, 0.0f, 0.0f);
    m_up = glm::vec3(0.0f, 1.0f, 0.0f);
    m_cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);

    m_width = width;
    m_height = height;

    m_speed = 1.0f;
    m_sensitivity = 0.1;

    m_viewmatrix = glm::lookAt(m_center + m_cameraPos, m_center, m_up);

    m_oldX = width / 2.f;
    m_oldY = height / 2.f;

    m_yaw = -90.0f;
    m_pitch = 0.0f;
    m_initialMouse = true;

    m_nearClippingPlane = 0.1f;
    m_farClippingPlane = 100.0f;
}

Camera::Camera(float width, float height, float fov, float nearClippingPlane, float farClippingPlane)
{
    m_fov = fov;
    m_projectionMatrix = glm::perspective(fov, width / height, nearClippingPlane, farClippingPlane);

    m_cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
    m_center = glm::vec3(0.0f, 0.0f, 0.0f);
    m_up = glm::vec3(0.0f, 1.0f, 0.0f);
    m_cameraFront = glm::vec3(0.0f, 0.0f, -5.0f);

    m_width = width;
    m_height = height;

    m_speed = 10.0f;
    m_sensitivity = 0.5f;

    m_viewmatrix = glm::lookAt(m_center + m_cameraPos, m_center, m_up);

    m_oldX = width / 2.f;
    m_oldY = height / 2.f;

    m_yaw = -90.0f;
    m_pitch = 0.0f;
    m_initialMouse = true;

    m_nearClippingPlane = nearClippingPlane;
    m_farClippingPlane = farClippingPlane;
}


void Camera::drawGui()
{
    ImGui::Begin("Camera");
    ImGui::SliderFloat("Camera Speed", &m_speed, 1.0f, 100.0f);
    ImGui::End();
}

void Camera::update(GLFWwindow *window)
{
    // update Gui
    drawGui();

    m_deltaTime = glfwGetTime() - m_LastDeltaTime;
    m_LastDeltaTime = glfwGetTime();

    double x, y;

    glfwGetCursorPos(window, &x, &y);
    if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        m_mouseLastPressed = true;

        if(m_initialMouse)
        {
            m_oldX = x;
            m_oldY = y;
            m_initialMouse = false;
        }

        float changeX = ((float) x - m_oldX) * m_sensitivity;
        float changeY = ((float) y - m_oldY) * m_sensitivity;

        m_oldX = (float) x;
        m_oldY = (float) y;

        m_yaw += changeX;
        m_pitch -= changeY;

        // reset pitch so camera can be rotated indefinitely
        if(m_pitch >= 360.0f || m_pitch <= -360.0f)
            m_pitch = 0.0f;

        glm::vec3 direction;
        direction.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
        direction.y = sin(glm::radians(m_pitch));
        direction.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));

        m_cameraFront = glm::normalize(direction);
    } else if(m_mouseLastPressed)
    {
        m_mouseLastPressed = false;
        m_initialMouse = true;
    }

    // calculate real up vector based on cross product of the cameras front facing direction and the y axis
    glm::vec3 up;

    if(m_pitch >= 90.0f && m_pitch <= 270.0f || m_pitch <= -90.0f && m_pitch >= -270.0f)
        up = glm::vec3(0.0f, -1.0f, 0.0f);
    else
        up = glm::vec3(0.0f, 1.0f, 0.0f);


    glm::vec3 cameraRight = glm::normalize(glm::cross(up, m_cameraFront));
    m_up = glm::cross(m_cameraFront, cameraRight);

    // speed up camera movement by pressing shift
    float speedUp = 1.0f;
    if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        speedUp = 100.0f;

    speedUp *= m_speed;

    // handle camera movement
    if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        m_cameraPos += m_speed * speedUp * m_cameraFront * m_deltaTime;
    if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        m_cameraPos -= m_speed * speedUp * m_cameraFront * m_deltaTime;
    if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        m_cameraPos -= glm::normalize(glm::cross(m_cameraFront, m_up)) * m_speed * 2.0f * speedUp * m_deltaTime;
    if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        m_cameraPos += glm::normalize(glm::cross(m_cameraFront, m_up)) * m_speed * 2.0f * speedUp * m_deltaTime;
    if(glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        m_cameraPos += m_speed * speedUp * m_deltaTime * m_up;
    if(glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        m_cameraPos -= m_speed * speedUp * m_deltaTime * m_up;

    m_viewmatrix = glm::lookAt(m_cameraPos, m_cameraPos + m_cameraFront, m_up);
}


Camera::~Camera()
{}

glm::mat4 *Camera::getViewMatrix()
{ return &m_viewmatrix; }

void Camera::getViewMatrix(glm::vec3 *x, glm::vec3 *y, glm::vec3 *z,
                           glm::vec3 *pos)
{
    *x = glm::vec3(glm::row(m_viewmatrix, 0));
    *y = glm::vec3(glm::row(m_viewmatrix, 1));
    *z = glm::vec3(glm::row(m_viewmatrix, 2));
    *pos = glm::vec3(glm::column(m_viewmatrix, 3));
    glm::mat3 mat_inv = glm::inverse(glm::mat3(m_viewmatrix));
    *pos = -mat_inv * *pos;
}

glm::mat4 *Camera::getProjectionMatrix()
{ return &m_projectionMatrix; }

void Camera::setViewMatrix(glm::mat4 *view)
{ m_viewmatrix = *view; }

void Camera::setWidthHeight(int width, int height)
{
    m_width = width;
    m_height = height;

    m_projectionMatrix = glm::perspective(m_fov, static_cast<float>(m_width) / m_height, 0.1f, 100.0f);
}

void Camera::getWidthHeight(int *width, int *height)
{
    *width = m_width;
    *height = m_height;
}

void Camera::lookAt(glm::vec3 position, glm::vec3 center, glm::vec3 up)
{
    m_viewmatrix = glm::lookAt(position, center, up);
}

void Camera::setCenter(glm::vec3 *center)
{
    m_center = *center;
}

void Camera::setUpvector(glm::vec3 *up)
{
    m_up = *up;
}

glm::vec3 *Camera::getCameraPosition()
{
    return &m_cameraPos;
}
