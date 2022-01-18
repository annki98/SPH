#include "SPHMesh.cuh"

// void genRandomData(glm::vec3* arr, int maxSize) {
//     std::random_device seed;
//     std::default_random_engine rng(seed());
//     std::uniform_real_distribution<float> dist(0, maxSize);
//     for (auto it = arr; it != arr + maxSize; it++) {
//         *it = glm::vec3(dist(rng),dist(rng),dist(rng));
//     }
// }

constexpr int numElements = int(1e4);

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
    uint3  hostGridSize = make_uint3(16,16,16); // must be power of 2

    m_psystem = new ParticleSystem(numElements, hostWorldOrigin, hostGridSize, h);

    // m_psystem->update(0.01f);
    //for testing purposes
    // psystem->checkNeighbors(5, numElements);
    // for(auto i = 0; i < numElements; i++) {
    //     float3 pos = m_psystem->getParticleArray()[i].position;
    //     glm::vec4 glPos = glm::vec4(pos.x, pos.y, pos.z ,1.0f);
    //     m_vertices.push_back(glPos);
    // }
    // createBuffers();


    time = 0.f;
}

SPHMesh::~SPHMesh(){
    delete m_psystem;
}

void SPHMesh::createBuffers()
{
    m_numberOfPoints = m_vertices.size();
    m_numberOfIndices = m_index.size();

    // create the buffers and bind the data
    m_vertexbuffer = m_psystem->getVBO(); // get vertex buffer from particleSystem (ALREADY FILLED)
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexbuffer);

    if(m_normalbuffer == 0 && m_normals.size() > 0)
    {
        glGenBuffers(1, &m_normalbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, m_normalbuffer);
        glBufferData(GL_ARRAY_BUFFER, m_numberOfPoints * sizeof(glm::vec3), &m_normals[0], GL_STATIC_DRAW);
    }

    if(m_uvbuffer == 0 && m_uvs.size() > 0)
    {
        glGenBuffers(1, &m_uvbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, m_uvbuffer);
        glBufferData(GL_ARRAY_BUFFER, m_numberOfPoints * sizeof(glm::vec2), &m_uvs[0], GL_STATIC_DRAW);
    }

    if(m_tangentbuffer == 0 && !m_tangents.empty())
    {
        if(m_tangents.empty())
        {
            //
        }
        glGenBuffers(1, &m_tangentbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, m_tangentbuffer);
        glBufferData(GL_ARRAY_BUFFER, m_tangents.size() * sizeof(glm::vec3), &m_tangents[0], GL_STATIC_DRAW);
    }

    // Generate a buffer for the indices as well
    if(m_indexlist == 0 && !m_index.empty())
    {
        glGenBuffers(1, &m_indexlist);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexlist);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_numberOfIndices * sizeof(unsigned int), &m_index[0], GL_STATIC_DRAW);
    }

    if(m_vao == 0)
        glGenVertexArrays(1, &m_vao);

    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertexbuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_normalbuffer);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_uvbuffer);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_tangentbuffer);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexlist);

    glBindVertexArray(m_vao);

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexbuffer);
}


void SPHMesh::updateParticles(float deltaTime){

    m_psystem->update(deltaTime);
    time += deltaTime;

    if(time > 3.f){ //Debug info every 3 sec
        time = 0.f;
        gpuErrchk( cudaDeviceSynchronize());  
        m_psystem->dumpParticleInfo(45,48);
    }
    
    // NOTE: m_vertices is not used anymore: filled vbo is returned from psystem
    createBuffers();
}

void SPHMesh::draw() 
{
    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, numElements);
}