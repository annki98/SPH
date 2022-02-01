#include "SPHMesh.cuh"

constexpr int numElements = int(2e3);
constexpr int sphereSections = int(12);

SPHMesh::SPHMesh(std::shared_ptr<State> state)
{
    m_state = state;

    // setup particle system
    float3 hostWorldOrigin = make_float3(0.f,0.f,0.f);
    float h = 1.f/4.f;
    uint3  hostGridSize = make_uint3(16,16,16); // must be power of 2

    m_psystem = std::make_unique<ParticleSystem>(numElements, hostWorldOrigin, hostGridSize, h);

    // m_psystem->update(0.01f);
    // gpuErrchk( cudaDeviceSynchronize()); 
    // m_psystem->dumpParticleInfo(0,1);
    // //for testing purposes
    // // psystem->checkNeighbors(5, numElements);
    // // for(auto i = 0; i < numElements; i++) {
    // //     float3 pos = m_psystem->getParticleArray()[i].position;
    // //     glm::vec4 glPos = glm::vec4(pos.x, pos.y, pos.z ,1.0f);
    // //     m_vertices.push_back(glPos);
    // // }
    createBuffers();

    time = 0.f;

    m_renderingMode = "Real Spheres";
    m_sphereRadius = 3.0f;
    m_renderBoundaries = true;

    // Setup shaders for sphere rendering
    m_vertexSphereShader = std::make_shared<Shader>(std::string(SHADERPATH) + "/Sphere.vert");
    m_fragmentSphereShader = std::make_shared<Shader>(std::string(SHADERPATH) + "/Sphere.frag");

    m_shpereShaderProgram = std::make_unique<ShaderProgram>("Sphere");
    m_shpereShaderProgram->addShader(m_vertexSphereShader);
    m_shpereShaderProgram->addShader(m_fragmentSphereShader);

    m_shpereShaderProgram->link();
    m_shpereShaderProgram->use();


    // Setup basic shader (red)
    m_vertexBasicShader = std::make_shared<Shader>(std::string(SHADERPATH) + "/Basic.vert");
    m_fragmentBasicShader = std::make_shared<Shader>(std::string(SHADERPATH) + "/Basic.frag");

    m_basicShaderProgram = std::make_unique<ShaderProgram>("Basic");
    m_basicShaderProgram->addShader(m_vertexBasicShader);
    m_basicShaderProgram->addShader(m_fragmentBasicShader);

    m_basicShaderProgram->link();
    m_basicShaderProgram->use();

    // Setup shader with model matrix
    m_vertexBasicWithModelShader = std::make_shared<Shader>(std::string(SHADERPATH) + "/BasicWithModel.vert");
    m_fragmentBasicWithModelShader = std::make_shared<Shader>(std::string(SHADERPATH) + "/BasicWithModel.frag");

    m_basicWithModelShaderProgram = std::make_unique<ShaderProgram>("Basic");
    m_basicWithModelShaderProgram->addShader(m_vertexBasicWithModelShader);
    m_basicWithModelShaderProgram->addShader(m_fragmentBasicWithModelShader);

    m_basicWithModelShaderProgram->link();
    m_basicWithModelShaderProgram->use();

    // Setup sphere vertices and normals for rendering
    setupSphere(glm::vec3(0), m_sphereRadius / 100.0f, sphereSections);
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

    glBindVertexArray(0);

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexbuffer);
}


void SPHMesh::setupSphere(glm::vec3 center, float radius, int resolution)
{
	// iniatialize the variable we are going to use
	float u, v;
	float phi, theta;
	float x, y, z;
	int offset = 0, i, j;

	// create points
	for ( j = 0; j <= resolution; j++)  //theta
		for ( i = 0; i <= resolution; i++) //phi
		{
			u = i /(float)resolution;		phi   = 2* glm::pi<float>() * u;
			v = j /(float)resolution;		theta =    glm::pi<float>() * v;

			x = center.x + radius * sin(theta) * sin(phi);
			y = center.y + radius * cos(theta);
			z = center.z + radius * sin(theta) * cos(phi);

			m_sphereVertices.push_back(glm::vec4( x, y, z, 1.0f));
			m_sphereNormals.push_back(glm::vec3( x, y, z) / radius);
		}

	// create index list
	for ( j = 0; j < resolution; j++)
	{
		for ( i = 0; i < resolution; i++)
		{
			// 1. Triangle
			m_sphereIndices.push_back( offset + i );
			m_sphereIndices.push_back( offset + i + resolution+1);
			m_sphereIndices.push_back( offset + i + resolution+1 + 1);

			// 2. Triangle
			m_sphereIndices.push_back( offset + i + resolution+1 + 1);
			m_sphereIndices.push_back( offset + i + 1);
			m_sphereIndices.push_back( offset + i );
		}
		offset += resolution+1;
	}

    // setup buffers for real sphere rendering
    glGenBuffers(1, &m_vboSphere);
    glBindBuffer(GL_ARRAY_BUFFER, m_vboSphere);
    glBufferData(GL_ARRAY_BUFFER, m_sphereVertices.size() * sizeof(glm::vec4), &m_sphereVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &m_normalbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_normalbuffer);
    glBufferData(GL_ARRAY_BUFFER, m_sphereVertices.size() * sizeof(glm::vec3), &m_sphereNormals[0], GL_STATIC_DRAW);

    glGenBuffers(1, &m_indexBufferSphere);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBufferSphere);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_sphereIndices.size() * sizeof(unsigned int), &m_sphereIndices[0], GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_vaoSphere);
    glBindVertexArray(m_vaoSphere);

    glBindBuffer(GL_ARRAY_BUFFER, m_vboSphere);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_normalbuffer);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBufferSphere);

    glBindVertexArray(0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboSphere);
}


void SPHMesh::setDepthtexture(GLuint depthTexture)
{
    m_depthTexture = depthTexture;
}


void SPHMesh::updateParticles(float deltaTime){

    m_psystem->update(deltaTime);
    // time += deltaTime;
    //printf("dt %f.",deltaTime);
    // if(time > 3.f){ //Debug info every 3 sec
    //     time = 0.f;
    //     m_psystem->dumpParticleInfo(0,1);
    //     //m_psystem->checkNeighbors(0, numElements);
    // }
    // gpuErrchk( cudaDeviceSynchronize());  
    // m_psystem->dumpParticleInfo(0,1);
    //m_psystem->checkNeighbors(0, m_psystem->numParticles());
}

void SPHMesh::drawGUI() 
{
    ImGui::Begin("SPH Mesh");
 
    // Combo Box to select either point rendering or sphere particle representation
    const char* items[] = { "Points", "Flat Spheres", "Real Spheres" };
    
    if (ImGui::BeginCombo("##combo", m_renderingMode))
    {
        for (int n = 0; n < IM_ARRAYSIZE(items); n++)
        {
            bool is_selected = (m_renderingMode == items[n]);
            if (ImGui::Selectable(items[n], is_selected))
                m_renderingMode = items[n];
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::Text("Rendering mode");

    // Sphere radius for rendering
    ImGui::SliderFloat("Sphere radius", &m_sphereRadius, 1.0f, 100.0f);

    // Toggle whether to render boundary walls
    ImGui::Checkbox("Draw Boundaries", &m_renderBoundaries);

    // Reset particle simulation
    if(ImGui::Button("Reset particles"))
        m_psystem->resetParticles(numElements);
    ImGui::End();
}

void SPHMesh::draw() 
{
    drawGUI();

    if (strcmp(m_renderingMode, "Points") == 0)
    {
        glBindVertexArray(m_vao);
        glPointSize(10.0f);
        m_basicShaderProgram->use();
        m_basicShaderProgram->setMat4("projectionMatrix", *m_state->getCamera()->getProjectionMatrix());
        m_basicShaderProgram->setMat4("viewMatrix", *m_state->getCamera()->getViewMatrix());

        if (m_renderBoundaries)
            glDrawArrays(GL_POINTS, 0, m_psystem->numParticles()); // use this to draw ALL particles (including Boundary particles)
        else
            glDrawArrays(GL_POINTS, 0, numElements); // use this to draw only fluid particles
    }
    else if (strcmp(m_renderingMode, "Flat Spheres") == 0)
    {
        glBindVertexArray(m_vao);
        glPointSize(m_sphereRadius * 100.0f); // offset for sphere rendering in shader
        m_shpereShaderProgram->use();
        m_shpereShaderProgram->setMat4("projectionMatrix", *m_state->getCamera()->getProjectionMatrix());
        m_shpereShaderProgram->setMat4("viewMatrix", *m_state->getCamera()->getViewMatrix());
        m_shpereShaderProgram->setFloat("sphereRadius", m_sphereRadius);
        m_shpereShaderProgram->setVec2("resolution", glm::vec2(m_state->getWidth(), m_state->getHeight()));
        if (m_renderBoundaries)
            glDrawArrays(GL_POINTS, 0, m_psystem->numParticles()); // use this to draw ALL particles (including Boundary particles)
        else
            glDrawArrays(GL_POINTS, 0, numElements); // use this to draw only fluid particles
    }
    else
    {
        gpuErrchk(cudaDeviceSynchronize());
        glBindVertexArray(m_vaoSphere);
        m_basicWithModelShaderProgram->use();
        m_basicWithModelShaderProgram->setMat4("projectionMatrix", *m_state->getCamera()->getProjectionMatrix());
        m_basicWithModelShaderProgram->setMat4("viewMatrix", *m_state->getCamera()->getViewMatrix());
        for (auto i = 0; i < numElements; i++)
        {
            float3 pos = m_psystem->getParticleArray()[i].position;
            glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, pos.y, pos.z));
            m_basicWithModelShaderProgram->setMat4("modelMatrix", modelMatrix);
            glDrawElements(GL_TRIANGLES, m_sphereIndices.size(), GL_UNSIGNED_INT, 0);
        }

        // render boundaries cheaply

        glBindVertexArray(m_vao);
        glPointSize(10.0f);
        m_basicShaderProgram->use();
        m_basicShaderProgram->setMat4("projectionMatrix", *m_state->getCamera()->getProjectionMatrix());
        m_basicShaderProgram->setMat4("viewMatrix", *m_state->getCamera()->getViewMatrix());

        if (m_renderBoundaries)
            glDrawArrays(GL_POINTS, numElements, m_psystem->numParticles());
    }
}