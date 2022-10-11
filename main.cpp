#include <iostream>
//#include "Tests/Simulation.h"
//#include "Tests/Test_PoissonSolverCircle.h"
//#include "Tests/SimulationCircle.h"
//#include "Tests/SimulationCircleGyro.h"
#include "Tests/SimulationCircleGyroNEW.h"
//#include "Tests/SimulationCircleNEW.h"

int main() {
    //simulation();
    //test_PoissonSolverCircle();
    //test_simulation_circle();
    //test_simulation_circle_gyro();
    MPI_Init(NULL, NULL);
    int commSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t0 = omp_get_wtime();
    test_simulation_circle_gyro_new();
    double t = omp_get_wtime();

    if (rank == 0)
    {
        ofstream fout("timeWholeCalculation.txt", ios::app);
        fout << commSize << ' ' << t - t0 << endl;
    }
    MPI_Finalize();
    //test_simulation_circle_new();

}
