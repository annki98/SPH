cmake_minimum_required(VERSION 3.3)


project(SPH CUDA CXX C)
cmake_policy(SET CMP0074 NEW)

# CUDA

include(CMakePrintHelpers)

cmake_print_variables(CMAKE_CUDA_COMPILER_VERSION)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)


# IMPORTANT: edit arch and code according to the graphics card you have
# see for example Here: https://gt3rs.medium.com/compile-with-nvcc-3566fbdfdbf
set(COMPUTE_CAPABILITY -gencode arch=compute_75,code=sm_75)
set(CUDA_OPTIONS -lineinfo ${COMPUTE_CAPABILITY})

message(${CUDA_OPTIONS})

add_subdirectory(Source/CudaTest)


# OPENGL/GLFW/GLM

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")


set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIB_DIR Include)

# use PACKAGE_ROOT variables
# cmake_policy(SET CMP0074 NEW)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)


add_subdirectory(Dependencies/glfw)

# target_link_libraries(${PROJECT_NAME} glfw)

# find_package(GLFW3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLM REQUIRED)

cmake_policy(SET CMP0104 OLD)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME}
        Include/imgui/imconfig.h
        Include/imgui/imgui.cpp
        Include/imgui/imgui.h
        Include/imgui/imgui_demo.cpp
        Include/imgui/imgui_draw.cpp
        Include/imgui/imgui_impl_glfw.cpp
        Include/imgui/imgui_impl_glfw.h
        Include/imgui/imgui_impl_opengl3.cpp
        Include/imgui/imgui_impl_opengl3.h
        Include/imgui/imgui_internal.h
        Include/imgui/imgui_widgets.cpp
        Include/imgui/imgui_impl_opengl3_loader.h
        Include/imgui/imstb_rectpack.h
        Include/imgui/imgui_tables.cpp
        Include/imgui/imstb_textedit.h
        Include/imgui/imstb_truetype.h
        Source/Engine/Defs.h
        Source/main.cu
        ${SRC_LIST}
        Source/Objects/Triangle.cpp
        Source/Objects/Triangle.h
        Source/Engine/Drawable.cpp
        Source/Engine/Drawable.h
        Source/Engine/Camera.cpp
        Source/Engine/Camera.h
        Source/Engine/ShaderProgram.cpp
        Source/Engine/ShaderProgram.h
        Source/Engine/Shader.cpp
        Source/Engine/Shader.h
        Include/stb_image.h
        Source/Engine/Texture.cpp
        Source/Engine/FBO.cpp
        Source/Engine/FBO.h
        Source/Objects/ScreenFillingQuad.cpp
        Source/Objects/ScreenFillingQuad.h
        Source/Objects/Quad.cpp
        Source/Objects/Quad.h
        Source/Engine/State.cpp
        Source/sph_naive.cu
        )

# target_compile_options(CudaTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
# set_target_properties(CudaTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIR} ${GLFW3_INCLUDE_DIR} ${GLM_INCLUDE_PATH} Include)

# glad
set(GLAD_DIR "${LIB_DIR}/glad")
add_library(glad "${GLAD_DIR}/src/glad.c")
target_include_directories(glad PRIVATE "${GLAD_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${GLAD_DIR}/include")
# target_link_libraries(${PROJECT_NAME} glad "${CMAKE_DL_LIBS}")

# add_subdirectory(Source/SPH_Naive)

target_link_libraries(
        ${PROJECT_NAME}
        ${OpenGL3_LIBRARIES}
       #  GLFW3::glfw3
        glfw
        GLM::glm
        opengl32
        glad
        imm32)

# prevent console from opening
#list(APPEND CMAKE_CXX_FLAGS "-mwindows")

add_compile_definitions(${PROJECT_NAME} PRIVATE  "SHADERPATH=\"${CMAKE_CURRENT_LIST_DIR}/Source/Shader\"")
