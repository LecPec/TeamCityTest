//
// Created by Vladimir Smirnov on 03.10.2021.
//

#include "Interpolation.h"
#include <cmath>
#include <omp.h>
#include <mpi.h>

void __LinearFieldInterpolation(scalar *Ex, scalar *Ey, const scalar *x, const scalar *y, const scalar *Ex_grid,
                                const scalar *Ey_grid, const Grid &grid, int Ntot) 
{
    int cell_x, cell_y, Ny=grid.Ny;
    scalar hx, hy;
    #pragma omp parallel for private(hx, hy, cell_x, cell_y) num_threads(NUM_THREADS)
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

void __LinearFieldInterpolationMPI(scalar efx[], scalar efy[], const scalar x[], const scalar y[],
                              const scalar *Ex, const scalar *Ey, const Grid& grid, const size_t Ntot)
{
    int cell_x, cell_y, Ny=grid.Ny;
    scalar hx, hy;
    #pragma omp parallel for private(hx, hy, cell_x, cell_y) num_threads(NUM_THREADS)
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

void __LinearChargeInterpolation(scalar *rho, const scalar *x, const scalar *y, const Grid &grid, scalar charge, int Ntot) {
    int cell_x, cell_y, Ny = grid.Ny;
    scalar hx, hy;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < grid.Nx * grid.Ny; ++i)
        rho[i] = 0;

    #pragma omp parallel for private(hx, hy, cell_x, cell_y) num_threads(NUM_THREADS)
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

void LinearFieldInterpolation(Particles &ptcl, const Matrix &Ex_grid, const Matrix &Ey_grid, const Grid &grid) {
    __LinearFieldInterpolation(ptcl.Ex.data(), ptcl.Ey.data(), ptcl.x.data(), ptcl.y.data(), Ex_grid.data_const_ptr(),
                               Ey_grid.data_const_ptr(), grid, ptcl.get_Ntot());
}

void LinearChargeInterpolation(Matrix &rho, const Particles &ptcl, const Grid &grid) {
    __LinearChargeInterpolation(rho.data_ptr(), ptcl.x.data(), ptcl.y.data(), grid,
                                ptcl.get_charge()*ptcl.get_ptcls_per_macro(),
                                ptcl.get_Ntot());
}

void LinearChargeInterpolationMPI(Matrix& rhoMatrix, const Particles& particles, const Grid& grid)
{
    int rank, commSize;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int Nx = grid.Nx, Ny = grid.Ny;

    int numOfPhysValues = 2;

    Matrix rhoProc(Ny, Nx);

    if (rank == 0){
        int Ntot = particles.get_Ntot();
        int NtotPerZeroProc = Ntot / commSize + Ntot % commSize;
        int NtotPerProc = Ntot / commSize;

        MPI_Datatype ParticlesDataType;
        int blockLength[numOfPhysValues];
        int displacements[numOfPhysValues];
        for (int i = 0; i < numOfPhysValues; ++i)
        {
            blockLength[i] = NtotPerProc;
            displacements[i] = i * NtotPerProc;
        }
        MPI_Type_indexed(numOfPhysValues, blockLength, displacements, MPI_DOUBLE, &ParticlesDataType);
        MPI_Type_commit(&ParticlesDataType);

        vector<scalar> particlesData;
        int start = 0;
        for (int proc = 1; proc < commSize; ++proc)
        {
            MPI_Send(&NtotPerProc, 1, MPI_INT, proc, 1991, MPI_COMM_WORLD);

            start = NtotPerZeroProc + (proc - 1) * NtotPerProc;
            particlesData.insert(particlesData.end(), particles.x.begin() + start, particles.x.begin() + start + NtotPerProc);
            particlesData.insert(particlesData.end(), particles.y.begin() + start, particles.y.begin() + start + NtotPerProc);

            MPI_Send(&particlesData[0], 1, ParticlesDataType, proc, 1221, MPI_COMM_WORLD);

            particlesData.clear();
        }

        __LinearChargeInterpolation(rhoProc.data_ptr(), particles.x.data(), particles.y.data(), grid, particles.get_charge() * particles.get_ptcls_per_macro(), NtotPerZeroProc);

        rhoMatrix.data = rhoProc.data;
        vector<scalar> rhoRecv;
        rhoRecv.resize(Nx * Ny);

        for (int proc = 1; proc < commSize; ++proc)
        {
            MPI_Recv(&rhoRecv[0], Nx * Ny, MPI_DOUBLE, proc, 878777, MPI_COMM_WORLD, &status);

            #pragma omp parallel for num_threads(NUM_THREADS)
            for (int i = 0; i < Nx * Ny; ++i)
                rhoMatrix.data[i] += rhoRecv[i];
        }
    }
    else{
        int NtotPerProc = 0;
        MPI_Recv(&NtotPerProc, 1, MPI_INT, 0, 1991, MPI_COMM_WORLD, &status);

        int sizeOfProcData = numOfPhysValues * NtotPerProc;

        vector<scalar> particlesData;
        particlesData.resize(sizeOfProcData);
        MPI_Recv(&particlesData[0], sizeOfProcData, MPI_DOUBLE, 0, 1221, MPI_COMM_WORLD, &status);

        vector<scalar> xProc, yProc;
        xProc.insert(xProc.end(), particlesData.begin(), particlesData.begin() + NtotPerProc);
        yProc.insert(yProc.end(), particlesData.begin() + NtotPerProc, particlesData.begin() + 2 * NtotPerProc);

        __LinearChargeInterpolation(rhoProc.data_ptr(), xProc.data(), yProc.data(), grid, particles.get_charge() * particles.get_ptcls_per_macro(), NtotPerProc);

        MPI_Send(&rhoProc.data[0], Nx * Ny, MPI_DOUBLE, 0, 878777, MPI_COMM_WORLD);
    }

}

void LinearFieldInterpolationMPI(Particles &particles, Matrix &Ex, Matrix &Ey, const Grid& grid)
{
    int rank, commSize;
    MPI_Status status;
    MPI_Comm comm_cart;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    
    int Nx = grid.Nx, Ny = grid.Ny;
    int numOfPhysValues = 6;


    if (rank == 0)
    {
        int Ntot_ = particles.get_Ntot();
        int NtotPerZeroProc = Ntot_ / commSize + Ntot_ % commSize;
        int NtotPerProc = Ntot_ / commSize;

        MPI_Datatype ParticlesDataType;
        int blockLength[numOfPhysValues];
        int displacements[numOfPhysValues];
        for (int i = 0; i < numOfPhysValues; ++i)
        {
            if (i == 0 || i == 1)
            {
                blockLength[i] = Nx * Ny;
                displacements[i] = i * Nx * Ny;
            }
            else
            {
                blockLength[i] = NtotPerProc;
                displacements[i] = (i - 2) * NtotPerProc + 2 * Nx * Ny;
            }
        }
        MPI_Type_indexed(numOfPhysValues, blockLength, displacements, MPI_DOUBLE, &ParticlesDataType);
        MPI_Type_commit(&ParticlesDataType);

        vector<scalar> particlesData;
        int start = 0;
        for (int proc = 1; proc < commSize; ++proc)
        {
            MPI_Send(&NtotPerProc, 1, MPI_INT, proc, 5775, MPI_COMM_WORLD);

            start = NtotPerZeroProc + (proc - 1) * NtotPerProc;
            particlesData.insert(particlesData.end(), Ex.data.begin(), Ex.data.end());
            particlesData.insert(particlesData.end(), Ey.data.begin(), Ey.data.end());
            particlesData.insert(particlesData.end(), particles.Ex.begin() + start, particles.Ex.begin() + start + NtotPerProc);
            particlesData.insert(particlesData.end(), particles.Ey.begin() + start, particles.Ey.begin() + start + NtotPerProc);
            particlesData.insert(particlesData.end(), particles.x.begin() + start, particles.x.begin() + start + NtotPerProc);
            particlesData.insert(particlesData.end(), particles.y.begin() + start, particles.y.begin() + start + NtotPerProc);

            MPI_Send(&particlesData[0], 1, ParticlesDataType, proc, 6776, MPI_COMM_WORLD);

            particlesData.resize(0);
        }
        
        __LinearFieldInterpolation(particles.Ex.data(), particles.Ey.data(), particles.x.data(), particles.y.data(), Ex.data_const_ptr(), Ey.data_const_ptr(), grid, NtotPerZeroProc);

        int numOfRecvData = 2;
        particlesData.resize(NtotPerProc * numOfRecvData);
        for (int proc = 1; proc < commSize; ++proc)
        {
            MPI_Recv(&particlesData[0], NtotPerProc * numOfRecvData, MPI_DOUBLE, proc, 4 * 99, MPI_COMM_WORLD, &status);
            //MPI_Recv(&efyRecv[0], NtotPerProc, MPI_DOUBLE, proc, 5 * 99, MPI_COMM_WORLD, &status);

            int start = NtotPerZeroProc + (proc - 1) * NtotPerProc;
            int finish = NtotPerZeroProc + proc * NtotPerProc;
            #pragma omp parallel for num_threads(NUM_THREADS)
            for (int ip = start; ip < finish; ++ip)
            {
                particles.Ex[ip] = particlesData[0 * NtotPerProc + ip - start];
                particles.Ey[ip] = particlesData[1 * NtotPerProc + ip - start];
            }
        }
    }

    else
    {
        int NtotPerProc = 0;
        MPI_Recv(&NtotPerProc, 1, MPI_INT, 0, 5775, MPI_COMM_WORLD, &status);

        int sizeOfProcData = (numOfPhysValues - 2) * NtotPerProc + 2 * Nx * Ny;

        vector<scalar> particlesData;
        particlesData.resize(sizeOfProcData);
        MPI_Recv(&particlesData[0], sizeOfProcData, MPI_DOUBLE, 0, 6776, MPI_COMM_WORLD, &status);

        vector<scalar> xProc;
        vector<scalar> yProc;
        vector<scalar> efxProc;
        vector<scalar> efyProc;
        
        Ex.data.resize(0);
        Ey.data.resize(0);
        Ex.data.insert(Ex.data.end(), particlesData.begin(), particlesData.begin() + Nx * Ny);
        Ey.data.insert(Ey.data.end(), particlesData.begin() + Nx * Ny, particlesData.begin() + 2 * Nx * Ny);
        efxProc.insert(efxProc.end(), particlesData.begin() + 2 * Nx * Ny, particlesData.begin() + 2 * Nx * Ny + NtotPerProc);
        efyProc.insert(efyProc.end(), particlesData.begin() + 2 * Nx * Ny + NtotPerProc, particlesData.begin() + 2 * Nx * Ny + 2 * NtotPerProc);
        xProc.insert(xProc.end(), particlesData.begin() + 2 * Nx * Ny + 2 * NtotPerProc, particlesData.begin() + 2 * Nx * Ny + 3 * NtotPerProc);
        yProc.insert(yProc.end(), particlesData.begin() + 2 * Nx * Ny + 3 * NtotPerProc, particlesData.begin() + 2 * Nx * Ny + 4 * NtotPerProc);

        __LinearFieldInterpolation(efxProc.data(), efyProc.data(), xProc.data(), yProc.data(), Ex.data_const_ptr(), Ey.data_const_ptr(), grid, NtotPerProc);

        particlesData.resize(0);
        int numOfSendingData = 2;
        particlesData.insert(particlesData.end(), efxProc.begin(), efxProc.end());
        particlesData.insert(particlesData.end(), efyProc.begin(), efyProc.end());
        MPI_Send(&particlesData[0], NtotPerProc * numOfSendingData, MPI_DOUBLE, 0, 4 * 99, MPI_COMM_WORLD);
        //MPI_Send(&efyProc[0], NtotPerProc, MPI_DOUBLE, 0, 5 * 99, MPI_COMM_WORLD);
    }
}
