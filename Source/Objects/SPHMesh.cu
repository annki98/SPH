#include "SPHMesh.cuh"

// void genRandomData(glm::vec3* arr, int maxSize) {
//     std::random_device seed;
//     std::default_random_engine rng(seed());
//     std::uniform_real_distribution<float> dist(0, maxSize);
//     for (auto it = arr; it != arr + maxSize; it++) {
//         *it = glm::vec3(dist(rng),dist(rng),dist(rng));
//     }
// }

constexpr int numElements = int(1e5);

SPHMesh::SPHMesh()
{
    glm::vec3 *ptrNative, *ptrCuda;
    ptrNative = static_cast<glm::vec3 *>(malloc(sizeof(glm::vec3) * numElements));

    cudaMalloc(&ptrCuda, sizeof(glm::vec3) * numElements);
    cudaMemcpy(ptrCuda, ptrNative, sizeof(glm::vec3) * numElements, cudaMemcpyHostToDevice);

    // genRandomData(&ptrCuda[0], 50);

    cudaMemcpy(ptrNative, ptrCuda, sizeof(glm::vec3) * numElements, cudaMemcpyDeviceToHost);


    // setup particle system
    float3 hostWorldOrigin = make_float3(0.f,0.f,0.f);
    float h = 1.f;
    uint3  hostGridSize = make_uint3(64,64,64); // must be power of 2

    ParticleSystem* psystem = new ParticleSystem(numElements, hostWorldOrigin, hostGridSize, h);

    psystem->update();

    //for testing purposes
    psystem->checkNeighbors(5);


    for(auto i = 0; i < numElements; i++) {
        float3 pos = psystem->getParticleArray()[i].position;
        glm::vec4 glPos = glm::vec4(pos.x, pos.y, pos.z ,1.0f);
        m_vertices.push_back(glPos);
    }

    createBuffers();
}

void SPHMesh::draw() 
{
    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_vertices.size());
}