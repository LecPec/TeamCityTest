//
// Created by Vladimir Smirnov on 06.11.2021.
//

#include "PoissonSolverCircle.h"
#include <cmath>
#include <cassert>
#include <mpi.h>
#include <omp.h>
#include <fstream>
#define EPSILON_0 8.854187817620389e-12

void MatrixToVector(const Matrix& m, vector<scalar>& vec)
{
    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < m.columns(); ++j)
        {
            vec.push_back(m(i, j));
        }
    }
}

void InitDirichletConditionsCircle(Matrix &phi, const Grid &grid, const int R) {
    assert(grid.dx == grid.dy);

    scalar phi_val = 0;

    for(int i = 0; i < phi.rows(); i++) {
        for(int j = 0; j < phi.columns(); j++) {
            if (pow(i - R, 2) + pow(j - R, 2) >= pow(R, 2)) {
                phi(i, j) = phi_val;
            }
        }
    }
}

bool convergence_check_circle(const Matrix &phi, const Matrix &rho, const Grid &grid, const int R, const scalar tol, const scalar betta) {
    scalar res;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;
    //#pragma omp parallel for private(i, j)
    for(int i = 0; i < phi.rows(); i++) {
        for(int j = 0; j < phi.columns(); j++) {
            if (pow(i - R, 2) + pow(j - R, 2) < pow(R, 2)) {
                res = phi(i, j) - (rho(i, j) / EPSILON_0 +
                                                (phi(i-1, j) + phi(i+1, j)) / dx2 +
                                                (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2);
                //cout << fabs(res) << endl;
                if (fabs(res) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

void PoissonSolverCircle(Matrix &phi, const Matrix &rho, const Grid &grid, const int R, const scalar tol, scalar betta,
                         int max_iter, int it_conv_check) {
    int Nx = grid.Nx;
    int Ny = grid.Ny;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;
    int i = 0, j = 0;

    for (int it = 0; it < max_iter; it++) {
        #pragma omp parallel for private(i, j) num_threads(NUM_THREADS)
        for(i = 0; i < Nx; i++) {
            for(j = 0; j < Ny; j++) {
                if (pow(i - R, 2) + pow(j - R, 2) < pow(R, 2)) {

                    phi(i, j) = betta * (rho(i, j) / EPSILON_0 +
                                        (phi(i-1, j) + phi(i+1, j)) / dx2 +
                                        (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2)
                                + phi(i, j) * (1 - betta);
                    /*
                    phi(i, j) = (rho(i, j) / EPSILON_0 +
                                 (phi(i-1, j) + phi(i+1, j)) / dx2 +
                                 (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2);
                    */
                }
            }
        }

        if (it % it_conv_check == 0) {
            if (convergence_check_circle(phi, rho, grid, R, tol, betta)) {
                return;
            }
        }
    }
}

void MpiConvergenceCheck(Matrix& phi, Matrix& oldPhi, const Grid& grid, const int R, scalar tol, int *initState)
{
    scalar res;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;
    int rank, commSize;
    int Nx = phi.columns();
    int Ny = phi.rows();

    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    int firstStripIdx = 0;
    int lastStripIdx = 0;

    if (rank == 0)
    {
        firstStripIdx = 0;
        lastStripIdx = Nx / commSize + Nx % commSize;
    }
    else if (rank == commSize - 1)
    {
        firstStripIdx = Nx - 1 - Nx / commSize;
        lastStripIdx = Nx - 1;
    }
    else if (rank != 0 and rank != commSize - 1)
    {
        firstStripIdx = Nx / commSize + Nx % commSize + (rank - 1) * (Nx / commSize) - 1;
        lastStripIdx =  Nx / commSize + Nx % commSize + rank * (Nx / commSize);
    }

    scalar finalError = 0;
    scalar currentError = 0;

    if (rank > 0)
    {
        vector<scalar> phiSend;
        phiSend.resize(Nx * Ny);

        vector<scalar> oldPhiSend;
        oldPhiSend.resize(Nx * Ny);

        phiSend = phi.ToVector();
        oldPhiSend = oldPhi.ToVector();

        MPI_Send(&phiSend[0], Nx * Ny, MPI_DOUBLE, 0, 12300, MPI_COMM_WORLD);
        MPI_Send(&oldPhiSend[0], Nx * Ny, MPI_DOUBLE, 0, 123100, MPI_COMM_WORLD);
    }
    else
    {
        vector<scalar> phiRecv;
        phiRecv.resize(Nx * Ny);

        vector<scalar> oldPhiRecv;
        oldPhiRecv.resize(Nx * Ny);

        for (int proc = 1; proc < commSize; ++proc)
        {
            int procStartStripIdx;
            int procEndStripIdx;
            if (proc != commSize - 1)
            {
                procStartStripIdx = Nx / commSize + Nx % commSize + (proc - 1) * (Nx / commSize) - 1;
                procEndStripIdx = Nx / commSize + Nx % commSize + proc * (Nx / commSize);
            }
            else if (proc == commSize - 1)
            {
                procStartStripIdx = Nx - 1 - Nx / commSize;
                procEndStripIdx = Nx - 1;
            }

            MPI_Recv(&phiRecv[0], Nx * Ny, MPI_DOUBLE, proc, 12300, MPI_COMM_WORLD, &status);
            MPI_Recv(&oldPhiRecv[0], Nx * Ny, MPI_DOUBLE, proc, 123100, MPI_COMM_WORLD, &status);

            for (int cellY = 0; cellY < Ny; ++cellY)
            {
                if (proc != commSize - 1)
                {
                    for (int cellX = procStartStripIdx + 1; cellX < procEndStripIdx; ++cellX)
                    {
                        phi(cellY, cellX) = phiRecv[cellY * Nx + cellX];
                        oldPhi(cellY, cellX) = oldPhiRecv[cellY * Nx + cellX];
                    }
                }
                else
                {
                    for (int cellX = procStartStripIdx + 1; cellX <= procEndStripIdx; ++cellX)
                    {
                        phi(cellY, cellX) = phiRecv[cellY * Nx + cellX];
                        oldPhi(cellY, cellX) = oldPhiRecv[cellY * Nx + cellX];
                    }
                }
            }
        }
    }
    if (rank == 0)
    {
        scalar recievedError;
        scalar maxError = -9999;
        int sendState = 0;
        int procWithMaxErr = 0;
        int averageMeanAbsError = finalError;

        scalar res;
        for (int i = 0; i < Ny; ++i)
        {
            for (int j = 0; j < Nx; ++j)
            {
                res = phi(i, j) - oldPhi(i, j);
                if (fabs(res) > maxError)
                    maxError = fabs(res);
            }
        }

        int needToSendStop = 0;

        if (maxError <= tol)
        {
            needToSendStop = 1;
            *initState = 1;
        }
        for (int proc = 1; proc < commSize; ++proc)
        {
            MPI_Send(&needToSendStop, 1, MPI_INT, proc, 99100, MPI_COMM_WORLD);
        }
    }
    else
    {
        int needToRecvStop = 0;

        MPI_Recv(&needToRecvStop, 1, MPI_INT, 0, 99100, MPI_COMM_WORLD, &status);

        if (needToRecvStop)
        {
            *initState = 1;
        }
    }
}

void OneIterationPoissonSolver(Matrix &phi, const Matrix &rho, const Grid &grid, const int R, scalar betta, int it)
{
    int Nx = grid.Nx;
    int Ny = grid.Ny;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;
    int rank, commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    int converge = 0;

    // <STRIPS_BORDERS_INIT>
    int firstStripIdx = 0;
    int lastStripIdx = 0;

    if (rank == 0)
    {
        firstStripIdx = 0;
        lastStripIdx = Nx / commSize + Nx % commSize;
    }
    else if (rank == commSize - 1)
    {
        firstStripIdx = Nx - 1 - Nx / commSize;
        lastStripIdx = Nx - 1;
    }
    else if (rank != 0 and rank != commSize - 1)
    {
        firstStripIdx = Nx / commSize + Nx % commSize + (rank - 1) * (Nx / commSize) - 1;
        lastStripIdx =  Nx / commSize + Nx % commSize + rank * (Nx / commSize);
    }
    // </STRIP_BORDERS_INIT>

    vector<scalar> phiLeftStrip;
    vector<scalar> phiRightStrip;
    phiLeftStrip.resize(Ny);
    phiRightStrip.resize(Ny);

    vector<scalar> leftSendPhi;
    vector<scalar> rightSendPhi;

    // <RECVS>
    if (rank == 0)
    {
        if (it > 0)
        {
            MPI_Recv(&phiRightStrip[0], Ny, MPI_DOUBLE, 1, rank * 100, MPI_COMM_WORLD, &status);

            for (int cellY = 0; cellY < Ny; ++cellY)
            {
                phi.SetData(cellY, lastStripIdx, phiRightStrip[cellY]);
            }
        }
    }
    else if (rank == commSize - 1)
    {
        if (it > 0)
        {
            MPI_Recv(&phiLeftStrip[0], Ny, MPI_DOUBLE, commSize - 2, rank * 100, MPI_COMM_WORLD, &status);

            for (int cellY = 0; cellY < Ny; ++cellY)
            {
                phi.SetData(cellY, firstStripIdx, phiLeftStrip[cellY]);
            }
        }
    }
    else
    {
        if (it > 0)
        {
            MPI_Recv(&phiLeftStrip[0], Ny, MPI_DOUBLE, rank - 1, rank * 100, MPI_COMM_WORLD, &status);
            MPI_Recv(&phiRightStrip[0], Ny, MPI_DOUBLE, rank + 1, rank * 100, MPI_COMM_WORLD, &status);

            for (int cellY = 0; cellY < Ny; ++cellY)
            {
                phi.SetData(cellY, firstStripIdx, phiLeftStrip[cellY]);
                phi.SetData(cellY, lastStripIdx, phiRightStrip[cellY]);
            }
        }
    }
    // </RECVS>

    // <PHI PROCESSING>
    if (rank == 0)
    {
        //#pragma omp parallel for collapse(2) num_threads(numThreads)
        for(int i = firstStripIdx; i < lastStripIdx; i++) {
            for(int j = 0; j < Ny; j++) {
                if (pow(j - R, 2) + pow(i - R, 2) < pow(R, 2)) {

                    phi(j, i) = betta * (rho(j, i) / EPSILON_0 +
                                    (phi(j, i-1) + phi(j, i+1)) / dx2 +
                                    (phi(j-1, i) + phi(j+1, i)) / dy2) / (2 / dx2 + 2 / dy2)
                                    + phi(j, i) * (1 - betta);
                }
            }
        }
    }
    else if (rank == commSize - 1)
    {
        //#pragma omp parallel for collapse(2) num_threads(numThreads)
        for(int i = firstStripIdx + 1; i <= lastStripIdx; i++) {
            for(int j = 0; j < Ny; j++) {
                if (pow(j - R, 2) + pow(i - R, 2) < pow(R, 2)) {

                    phi(j, i) = betta * (rho(j, i) / EPSILON_0 +
                                    (phi(j, i-1) + phi(j, i+1)) / dx2 +
                                    (phi(j-1, i) + phi(j+1, i)) / dy2) / (2 / dx2 + 2 / dy2)
                                    + phi(j, i) * (1 - betta);
                }
            }
        }
    }
    else
    {
        
        //#pragma omp parallel for collapse(2) num_threads(numThreads)
        for(int i = firstStripIdx + 1; i < lastStripIdx; i++) {
            for(int j = 0; j < Ny; j++) {
                if (pow(j - R, 2) + pow(i - R, 2) < pow(R, 2)) {

                    phi(j, i) = betta * (rho(j, i) / EPSILON_0 +
                                    (phi(j, i-1) + phi(j, i+1)) / dx2 +
                                    (phi(j-1, i) + phi(j+1, i)) / dy2) / (2 / dx2 + 2 / dy2)
                                    + phi(j, i) * (1 - betta);
                }
            }
        }
    }
    // </PHI_PROCESSING>

    // <SENDS>
    if (rank == 0)
    {
        rightSendPhi = phi.StripToVector(lastStripIdx - 1);
        MPI_Send(&rightSendPhi[0], Ny, MPI_DOUBLE, 1, (rank + 1) * 100, MPI_COMM_WORLD);
    }
    else if (rank == commSize - 1)
    {
        leftSendPhi = phi.StripToVector(firstStripIdx + 1);
        MPI_Send(&leftSendPhi[0], Ny, MPI_DOUBLE, commSize - 2, (rank - 1) * 100, MPI_COMM_WORLD);
    }
    else
    {
        leftSendPhi = phi.StripToVector(firstStripIdx + 1);
        rightSendPhi = phi.StripToVector(lastStripIdx - 1);
        MPI_Send(&leftSendPhi[0], Ny, MPI_DOUBLE, rank - 1, (rank - 1) * 100, MPI_COMM_WORLD);
        MPI_Send(&rightSendPhi[0], Ny, MPI_DOUBLE, rank + 1, (rank + 1) * 100, MPI_COMM_WORLD);
    }
    // </SENDS>
}

void PoissonSolverCircleMPI(Matrix &phi, Matrix &rho, const Grid &grid, const int R, const scalar tol, scalar betta,
                         int max_iter, int it_conv_check) {
    int Nx = grid.Nx;
    int Ny = grid.Ny;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;
    Matrix oldPhi(Nx, Ny);
    //InitDirichletConditionsCircle(oldPhi, grid, R);
    int rank, commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    int converged = 0;

    int firstStripIdx = 0;
    int lastStripIdx = 0;

    double t0 = 0, t = 0;
    int convIt = 0;

    if (rank == 0)
    {
        vector<scalar> rhoSend = rho.ToVector();
        for (int proc = 1; proc < commSize; ++proc)
            MPI_Send(&rhoSend[0], Nx * Ny, MPI_DOUBLE, proc, 606060, MPI_COMM_WORLD);
    }
    else
    {
        vector<scalar> recvRho;
        recvRho.resize(Nx * Ny);

        MPI_Recv(&recvRho[0], Nx * Ny, MPI_DOUBLE, 0, 606060, MPI_COMM_WORLD, &status);

        rho.InitFromVector(recvRho);
    }

    for(int it = 0; it < max_iter; ++it) 
    {
        oldPhi = phi;
        OneIterationPoissonSolver(phi, rho, grid, R, betta, it);
        
        if (it % it_conv_check == 0 and it > 0)
        {   
            MpiConvergenceCheck(phi, oldPhi, grid, R, tol, &converged);
            //convergence_check_circle(phi, rho, grid, R, tol);

            if (converged)
            {
                return;
            }
        }
    }   
    
}
