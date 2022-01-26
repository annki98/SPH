#include <iostream>
#include <filesystem>
#include <random>

#include "Engine/Defs.h"
#include "Objects/Triangle.h"
#include "Engine/Camera.h"
#include "Engine/Shader.h"
#include "Engine/ShaderProgram.h"
#include "Engine/State.h"
#include "Engine/FBO.h"
#include "Objects/ScreenFillingQuad.h"
#include "Objects/SPHMesh.cuh"

#include "CudaGrid/particlegrid.cuh"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

const int WIDTH =  1280;
const int HEIGHT = 768;

const bool FULLSCREEN = false;



// context creation callback
void errorCallback(int error, const char *description)
{
    LOG_ERROR(description);
}

void sizeCallback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// open gl error callback
void GLAPIENTRY
MessageCallback(GLenum source,
                GLenum type,
                GLuint id,
                GLenum severity,
                GLsizei length,
                const GLchar *message,
                const void *userParam)
{
    if(severity > GL_DEBUG_SEVERITY_LOW)
        fprintf(stderr, "GL Error: %s type = 0x%x, severity = 0x%x, message = %s\n",
                type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "",
                type, severity, message);
}

void updateImGuiTheme()
{
    ImGui::GetStyle().FrameRounding = 4.0f;
    ImGui::GetStyle().GrabRounding = 4.0f;

    ImVec4 *colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_Text] = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.36f, 0.42f, 0.47f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.11f, 0.15f, 0.17f, 0.85f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.15f, 0.18f, 0.22f, 0.70f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
    colors[ImGuiCol_Border] = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.25f, 0.39f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.20f, 0.28f, 1.00f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.09f, 0.12f, 0.14f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.09f, 0.12f, 0.14f, 0.65f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.39f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.18f, 0.22f, 0.25f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.09f, 0.21f, 0.31f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.37f, 0.61f, 1.00f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.29f, 0.55f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_Separator] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.10f, 0.40f, 0.75f, 0.78f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.10f, 0.40f, 0.75f, 1.00f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    colors[ImGuiCol_Tab] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
}


// CUDA Test
void __global__ copyArray(const float* in, float* out, int num)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = id; i < num; i += stride) {
        out[i] = in[i];
    }
}

template<typename itT>
void genRandomData(itT begin, itT end) {
    std::random_device seed;
    std::default_random_engine rng(seed());
    std::uniform_real_distribution<float> dist(0, 100);
    for (auto it = begin; it != end; it++) {
        *it = dist(rng);
    }
}


int main()
{

    glfwSetErrorCallback(errorCallback);

    if(!glfwInit())
    {
        printf("failed to initialize OpenGL\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window;
    if(FULLSCREEN)
        window = glfwCreateWindow(WIDTH, HEIGHT, "SPH", glfwGetPrimaryMonitor(), nullptr);
    else
        window = glfwCreateWindow(WIDTH, HEIGHT, "SPH", NULL, nullptr);

    if(!window)
    {
        LOG_ERROR("Failed to create window (Glfw)");
        glfwTerminate();
        return -1;
    }

    glfwSetWindowSizeCallback(window, sizeCallback);
    glfwMakeContextCurrent(window);

    // disable vsync for performance meassuring
    glfwSwapInterval(0);

    if(!gladLoadGL())
    {
        LOG_ERROR("Failed to create window (Glad)");
        exit(-1);
    }

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(MessageCallback, 0);

    std::cout << "OpenGL " << glGetString(GL_VERSION) << ", GLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION)
              << ", Rendering on " << glGetString(GL_RENDERER) << std::endl;

    // setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    // setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
    // setup Dear ImGui style
    ImGui::StyleColorsDark();
    updateImGuiTheme();
    // enable docking
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigDockingWithShift = false;
    
    // Init state system
    std::shared_ptr<State> state = std::make_shared<State>(WIDTH, HEIGHT);

    std::unique_ptr<SPHMesh> sphmesh = std::make_unique<SPHMesh>(state);

    // screen filling quad
    std::unique_ptr<ScreenFillingQuad> sfq = std::make_unique<ScreenFillingQuad>(state);

    double startTime = glfwGetTime();

    // FBO for rendering the scene
    std::unique_ptr<FBO> sceneFBO = std::make_unique<FBO>(WIDTH, HEIGHT, 2);
    
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneFBO->getColorAttachment(0), 0);

    // set larger point size so points are properly visible on higher res screens
    glPointSize(10.0f);

    // move camera to a position where mesh is visible
    state->getCamera()->setCameraPosition(glm::vec3(25.0f, 10.0f, 50.0f));

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if(glfwGetKey(window, GLFW_KEY_ESCAPE))
            exit(0);

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        float deltaTime = glfwGetTime() - startTime;
        startTime = glfwGetTime();

        state->getCamera()->update(window);
        state->setTime(glfwGetTime());
        state->setDeltaTime(deltaTime);

        // First pass for water
        glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO->getID());
        glClearColor(0.8f, 0.8f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        sphmesh->updateParticles(deltaTime);
        sphmesh->draw();


        // Second pass with SFQ
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_DEPTH_TEST);

        sfq->getShaderProgram()->use();
        sfq->getShaderProgram()->setSampler2D("fbo", sceneFBO->getColorAttachment(0), 0);
        sfq->draw(sceneFBO->getColorAttachment(0));

        ImGui::Begin("Performance");
        ImGui::Text("%s", std::string("Frame Time: " + std::to_string(deltaTime * 1000.0f) + "ms").c_str());
        ImGui::Text("%s", std::string("Frames per Second: " + std::to_string(1.0f / deltaTime)).c_str());
        ImGui::End();


        // Render dear imgui into screen
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    sphmesh.reset(); //call destructor of psystem before opengl context gets deleted
    glfwTerminate();
    return 0;
}
