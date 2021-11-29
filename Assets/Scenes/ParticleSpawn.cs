using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ParticleSpawn : MonoBehaviour
{

    public struct Particle{
        public Vector3 position;
        public Vector3 velocity;
    };
    public ComputeShader particleShader;

    public GameObject ball;
    private ComputeBuffer buffer;
    private Particle[] particles;
    private GameObject[] objects;
    public int num;
    public void InitializeParticles(){
        for(int i = 0; i < this.particles.Length; i++){
            this.particles[i].position = new Vector3(Random.Range(-2.0f,2.0f),Random.Range(-2.0f,2.0f),Random.Range(-2.0f,2.0f));
            this.particles[i].velocity = new Vector3(Random.Range(-2.0f,2.0f),Random.Range(-2.0f,2.0f),Random.Range(-2.0f,2.0f));

            this.objects[i] = GameObject.Instantiate(ball);
            this.objects[i].GetComponent<MeshRenderer>().material.SetColor("_Color", Random.ColorHSV());
            this.objects[i].SetActive(true);
        }
    }
    // private int time = 0;
    // Start is called before the first frame update
    void Start()
    {
        particles = new Particle[num];
        objects = new GameObject[num];

        InitializeParticles();
        int totalSize = (sizeof(float)*3)*2; 
        buffer = new ComputeBuffer(num, totalSize);
        buffer.SetData(particles);

        particleShader.SetBuffer(particleShader.FindKernel("ParticleMain"), "particles", buffer);

    }

    private void printParticles(){
        for(int i = 0; i < particles.Length; i++){
            print("Position ("+  i+"): "+ particles[i].position);
            print("Velocity ("+  i+"): "+ particles[i].velocity);
        }
    }
    // Update is called once per frame
    void Update()
    {
        buffer.SetData(particles);
        // time++;
        particleShader.SetFloat("deltat", Time.deltaTime);
        particleShader.Dispatch(particleShader.FindKernel("ParticleMain"), particles.Length/10,1,1);

        buffer.GetData(particles);
        // if(time % 60 == 0){
        //     printParticles();
        //     time = 0;
        // }
        
        for(int i = 0; i < particles.Length; i++){
            this.objects[i].transform.position = particles[i].position;
        }
    }
    
    void OnApplicationQuit() 
    {
        buffer.Dispose();
    }
}
