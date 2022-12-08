CC=mpicxx -std=c++11 -O3 -Xpreprocessor -fopenmp -lomp
#CC=g++ -pg -std=c++11 -O3 -fopenmp

all: pic_2d_openmp

pic_2d_openmp: Names.o Exception.o PoissonSolver.o Matrix.o Pusher.o Particles.o Grid.o Interpolation.o ParticleEmission.o ParticleLeave.o Collision.o NullCollisions.o NeutralGas.o Logger.o ParticlesLogger.o PoissonSolverCircle.o EnergyCrossSection.o Helpers.o GyroKineticParticles.o GyroKineticPusher.o main.o
	$(CC) -o LaunchSimulation/pic_2d_openmp Names.o Exception.o PoissonSolver.o Matrix.o Particles.o Pusher.o Grid.o Interpolation.o ParticleEmission.o ParticleLeave.o Collision.o NullCollisions.o NeutralGas.o Logger.o ParticlesLogger.o PoissonSolverCircle.o EnergyCrossSection.o Helpers.o GyroKineticParticles.o GyroKineticPusher.o main.o
	rm -rf *.o

main.o: main.cpp
	$(CC) -c main.cpp

Names.o: Tools/Names.cpp
	$(CC) -c Tools/Names.cpp

Exception.o: Tools/Exception.cpp
	$(CC) -c Tools/Exception.cpp	

PoissonSolver.o: Field/PoissonSolver.cpp
	$(CC) -c Field/PoissonSolver.cpp

PoissonSolverCircle.o: Field/PoissonSolverCircle.cpp
	$(CC) -c Field/PoissonSolverCircle.cpp

Matrix.o: Tools/Matrix.cpp
	$(CC) -c Tools/Matrix.cpp

Pusher.o: Particles/Pusher.cpp
	$(CC) -c Particles/Pusher.cpp

Particles.o: Particles/Particles.cpp
	$(CC) -c Particles/Particles.cpp

Grid.o: Tools/Grid.cpp
	$(CC) -c Tools/Grid.cpp

Interpolation.o: Interpolation/Interpolation.cpp
	$(CC) -c Interpolation/Interpolation.cpp

ParticleEmission.o: ParticleLeaveEmission/ParticleEmission.cpp
	$(CC) -c ParticleLeaveEmission/ParticleEmission.cpp

ParticleLeave.o: ParticleLeaveEmission/ParticleLeave.cpp
	$(CC) -c ParticleLeaveEmission/ParticleLeave.cpp

Collision.o: Collisions/Collision.cpp
	$(CC) -c Collisions/Collision.cpp

NullCollisions.o: Collisions/NullCollisions.cpp
	$(CC) -c Collisions/NullCollisions.cpp

NeutralGas.o: Collisions/NeutralGas.cpp
	$(CC) -c Collisions/NeutralGas.cpp

Logger.o: Tools/Logger.cpp
	$(CC) -c Tools/Logger.cpp

ParticlesLogger.o: Tools/ParticlesLogger.cpp
	$(CC) -c Tools/ParticlesLogger.cpp

Helpers.o: Tools/Helpers.cpp
	$(CC) -c Tools/Helpers.cpp

EnergyCrossSection.o: Collisions/EnergyCrossSection.cpp
	$(CC) -c Collisions/EnergyCrossSection.cpp

GyroKineticPusher.o: Particles/GyroKineticPusher.cpp
	$(CC) -c Particles/GyroKineticPusher.cpp

GyroKineticParticles.o: Particles/GyroKineticParticles.cpp
	$(CC) -c Particles/GyroKineticParticles.cpp

clean:
	rm -rf *.o
