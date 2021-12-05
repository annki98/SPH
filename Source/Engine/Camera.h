//
// Created by maxbr on 14.04.2020.
//

#pragma once

#include "Defs.h"
#include <glm/gtc/matrix_transform.hpp>
#include "imgui/imgui.h"

#include <vector>

class Camera
{
public:
    Camera(float width, float height, float fov, float nearClippingPlane, float farClippingPlane);

    Camera(float width, float height);

    ~Camera();

    void update(GLFWwindow *window);

    void setCenter(glm::vec3 *center);

    void setUpvector(glm::vec3 *up);

    glm::mat4 *getViewMatrix();

    glm::mat4 *getProjectionMatrix();

    void getViewMatrix(glm::vec3 *x, glm::vec3 *y, glm::vec3 *z, glm::vec3 *pos);

    void setViewMatrix(glm::mat4 *view);

    void setWidthHeight(int width, int height);

    void getWidthHeight(int *width, int *height);

    glm::vec3 *getCameraPosition();

    void lookAt(glm::vec3 position, glm::vec3 center, glm::vec3 up);

protected:
    int m_width, m_height;
    glm::mat4 m_viewmatrix;
    glm::mat4 m_projectionMatrix;
private:
    void drawGui();

    glm::vec3 m_cameraPos, m_center, m_up, m_cameraFront;

    float m_fov;
    float m_deltaTime, m_LastDeltaTime;
    float m_oldX, m_oldY;
    float m_sensitivity, m_speed;
    float m_pitch, m_yaw;
    float m_farClippingPlane, m_nearClippingPlane;
    bool m_initialMouse;
    bool m_mouseLastPressed;
};