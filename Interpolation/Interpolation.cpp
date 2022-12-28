#include "Interpolation.h"
#include <cmath>
#include <omp.h>
#include <mpi.h>

void __LinearFieldInterpolation(vector<scalar> &Ex, vector<scalar> &Ey, const vector<scalar> &x, const vector<scalar> &y, const vector<scalar> &Ex_grid,
                                const vector<scalar> &Ey_grid, const Grid &grid, int Ntot) 
{
    int cell_x, cell_y, Ny=grid.Ny;
    scalar hx, hy;

    auto *settings = new SettingNames();
    int numThreads = settings->GetNumberOfThreadsPerCore(); 

    #pragma omp parallel for private(hx, hy, cell_x, cell_y)
    for (int i = 0; i < Ntot; i++) {
        cell_x = floor(x[i]/grid.dx);
        cell_y = floor(y[i]/grid.dy);
        hx = (x[i] - cell_x*grid.dx) / grid.dx;
        hy = (y[i] - cell_y*grid.dy) / grid.dy;

        Ex[i] = Ex_grid[cell_x*Ny + cell_y] * (1 - hx) * (1 - hy);
        Ex[i] += Ex_grid[(cell_x+1)*Ny + cell_y] * hx * (1 - hy);
        Ex[i] += Ex_grid[(cell_x+1)*Ny + cell_y+1] * hx * hy;
        Ex[i] += Ex_grid[cell_x*Ny + cell_y+1] * (1 - hx) * hy;

        Ey[i] = Ey_grid[cell_x*Ny + cell_y] * (1 - hx) * (1 - hy);
        Ey[i] += Ey_grid[(cell_x+1)*Ny + cell_y] * hx * (1 - hy);
        Ey[i] += Ey_grid[(cell_x+1)*Ny + cell_y+1] * hx * hy;
        Ey[i] += Ey_grid[cell_x*Ny + cell_y+1] * (1 - hx) * hy;
    }
}

void __LinearFieldInterpolationMPI(vector<scalar> &efx, vector<scalar> &efy, const vector<scalar> &x, const vector<scalar> &y,
                              const vector<scalar> &Ex, const vector<scalar> &Ey, const Grid& grid, const size_t Ntot)
{
    int cell_x, cell_y, Ny=grid.Ny;
    scalar hx, hy;

    auto *settings = new SettingNames();
    int numThreads = settings->GetNumberOfThreadsPerCore(); 

    #pragma omp parallel for private(hx, hy, cell_x, cell_y)
    for (int i = 0; i < Ntot; i++) {
        cell_x = floor(x[i]/grid.dx);
        cell_y = floor(y[i]/grid.dy);
        hx = (x[i] - cell_x*grid.dx) / grid.dx;
        hy = (y[i] - cell_y*grid.dy) / grid.dy;

        efx[i] = Ex[cell_x*Ny + cell_y] * (1 - hx) * (1 - hy);
        efx[i] += Ex[(cell_x+1)*Ny + cell_y] * hx * (1 - hy);
        efx[i] += Ex[(cell_x+1)*Ny + cell_y+1] * hx * hy;
        efx[i] += Ex[cell_x*Ny + cell_y+1] * (1 - hx) * hy;

        efy[i] = Ey[cell_x*Ny + cell_y] * (1 - hx) * (1 - hy);
        efy[i] += Ey[(cell_x+1)*Ny + cell_y] * hx * (1 - hy);
        efy[i] += Ey[(cell_x+1)*Ny + cell_y+1] * hx * hy;
        efy[i] += Ey[cell_x*Ny + cell_y+1] * (1 - hx) * hy;
    }
}

void __LinearChargeInterpolation(vector<scalar> &rho, const vector<scalar> &x, const vector<scalar> &y, const Grid &grid, scalar charge, int Ntot) {
    int cell_x, cell_y, Ny = grid.Ny;
    scalar hx, hy;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < grid.Nx * grid.Ny; ++i)
        rho[i] = 0;

    auto *settings = new SettingNames();
    int numThreads = settings->GetNumberOfThreadsPerCore(); 

    #pragma omp parallel for private(hx, hy, cell_x, cell_y)
    for (int i = 0; i < Ntot; i++) {
        cell_x = floor(x[i] / grid.dx);
        cell_y = floor(y[i] / grid.dy);

        hx = (x[i] - cell_x * grid.dx) / grid.dx;
        hy = (y[i] - cell_y * grid.dy) / grid.dy;
        #pragma omp atomic
        rho[cell_x * Ny + cell_y] += (charge * (1 - hx) * (1 - hy) / (grid.dx*grid.dy));
        #pragma omp atomic
        rho[(cell_x + 1) * Ny + cell_y] += (charge * hx * (1 - hy) / (grid.dx*grid.dy));
        #pragma omp atomic
        rho[(cell_x + 1) * Ny + cell_y + 1] += (charge * hx * hy / (grid.dx*grid.dy));
        #pragma omp atomic
        rho[cell_x * Ny + cell_y + 1] += (charge * (1 - hx) * hy / (grid.dx*grid.dy));
    }
}

void LinearChargeInterpolation(Matrix &rho, const Particles &ptcl, const Grid &grid) {
    __LinearChargeInterpolation(rho.data, ptcl.x, ptcl.y, grid,
                                ptcl.get_charge()*ptcl.get_ptcls_per_macro(),
                                ptcl.get_Ntot());
}

void LinearChargeInterpolationMPI(Matrix& rhoMatrix, Particles& particles, const Grid& grid)
{
    int rank, commSize;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int Nx = grid.Nx, Ny = grid.Ny;
    int NtotPerProc = particles.get_Ntot() / commSize;
    int NtotPerZeroProc = NtotPerProc + particles.get_Ntot() % commSize;

    Matrix rho(Nx, Ny);

    MPI_Bcast(&NtotPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NtotPerZeroProc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int numOfPtclsToCalculate  = (rank == 0) ? NtotPerZeroProc : NtotPerProc;

    particles.Resize(numOfPtclsToCalculate);

    int counts[commSize], displs[commSize];
    counts[0] = NtotPerZeroProc;
    displs[0] = 0;
    for (int i = 1; i < commSize; ++i)
    {
        counts[i] = NtotPerProc;
        displs[i] = NtotPerZeroProc + (i - 1) * NtotPerProc;
    }

    MPI_Scatterv(&particles.x[0], counts, displs, MPI_DOUBLE, &particles.x_[0], numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&particles.y[0], counts, displs, MPI_DOUBLE, &particles.y_[0], numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    __LinearChargeInterpolation(rho.data, particles.x_, particles.y_, grid, particles.get_charge() * particles.get_ptcls_per_macro(), numOfPtclsToCalculate);

    MPI_Reduce(&rho.data[0], &rhoMatrix.data[0], Nx * Ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void LinearFieldInterpolationMPI(Particles &particles, Matrix &Ex, Matrix &Ey, const Grid& grid, int iteration)
{
    int rank, commSize;
    MPI_Status status;
    MPI_Comm comm_cart;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    
    int Nx = grid.Nx, Ny = grid.Ny;
    int NtotPerProc = particles.get_Ntot() / commSize;
    int NtotPerZeroProc = NtotPerProc + particles.get_Ntot() % commSize;

    MPI_Bcast(&NtotPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NtotPerZeroProc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ex.data[0], Nx * Ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ey.data[0], Nx * Ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int numOfPtclsToCalculate  = (rank == 0) ? NtotPerZeroProc : NtotPerProc;

    int counts[commSize], displs[commSize], countsMatrix[commSize], displsMatrix[commSize];
    counts[0] = NtotPerZeroProc;
    displs[0] = 0;
    countsMatrix[0] = Nx * Ny;
    displsMatrix[0] = 0;
    for (int i = 1; i < commSize; ++i)
    {
        counts[i] = NtotPerProc;
        displs[i] = NtotPerZeroProc + (i - 1) * NtotPerProc;
        countsMatrix[i] = Nx * Ny;
        displsMatrix[i] = 0;
    }

    MPI_Scatterv(&particles.x[0], counts, displs, MPI_DOUBLE, particles.x_.data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&particles.y[0], counts, displs, MPI_DOUBLE, particles.y_.data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&particles.Ex[0], counts, displs, MPI_DOUBLE, particles.Ex_.data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&particles.Ey[0], counts, displs, MPI_DOUBLE, particles.Ey_.data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    __LinearFieldInterpolation(particles.Ex_, particles.Ey_, particles.x_, particles.y_, Ex.data, Ey.data, grid, numOfPtclsToCalculate);
    
    MPI_Gatherv(particles.Ex_.data(), numOfPtclsToCalculate, MPI_DOUBLE, &particles.Ex[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(particles.Ey_.data(), numOfPtclsToCalculate, MPI_DOUBLE, &particles.Ey[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}