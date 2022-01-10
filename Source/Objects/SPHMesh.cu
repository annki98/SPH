//
// Created by maxbr on 10.05.2020.
//

#include "SPHMesh.cuh"

template<typename itT>
void genRandomData(itT arr, int maxSize) {
    std::random_device seed;
    std::default_random_engine rng(seed());
    std::uniform_real_distribution<float> dist(0, maxSize);
    for (auto it = 0; it < maxSize; it++) {
        arr[it] = glm::vec3(dist(rng),dist(rng),dist(rng));
    }
}

SPHMesh::SPHMesh()
{
    
    glm::vec3* ptrNative = static_cast<glm::vec3 *>(malloc(sizeof(glm::vec3) * 200));
    //cudaMalloc((void**)&ptrNative, 200 * sizeof(ptrNative[0]));
    genRandomData(&ptrNative[0], 50);

    for(auto it = 0; it < 200; it++) {
        m_vertices.push_back(glm::vec4(ptrNative[it], 1.0f));
    }

    createBuffers();
}

void SPHMesh::draw() 
{
    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_vertices.size());
}