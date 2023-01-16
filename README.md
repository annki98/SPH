# SPH mit CUDA

<img src="/images/Center_Splash.png" alt="splash" height="250"/> <img src="/images/Driveby.png" alt="driveby" height="250"/> <img src="/images/Overflow.png" alt="overflow" height="250"/>


# Setup

1. Im Ordner Dependencies jeweils cmake mit `cmake .` konfigurieren und anschießend mit `cmake --build` im gleichen Ordner bauen. Das ganze für glm und glfw. Optional lässt sich das auch in VS Code durchführen.
2. SPH Project in VS Code öffnen und gucken ob es klappt,  OpenGL sollte automatisch gefunden werden.
3. Umgebungsvariablen `GLFW3_ROOT` und `GLM_ROOT` hinzufügen, die auf die bei cmake angegebenen build-, bzw. root-Ordner der eben gebauten dependencies zeigen.
4. Bitte nur bei Erfolg berichten :|
