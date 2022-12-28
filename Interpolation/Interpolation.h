#ifndef CPP_2D_PIC_INTERPOLATION_H
#define CPP_2D_PIC_INTERPOLATION_H

#include "../Tools/Matrix.h"
#include "../Tools/Grid.h"
#include "../Particles/Particles.h"
#define NUM_THREADS 4

void __LinearFieldInterpolation(vector<scalar> &Ex, vector<scalar> &Ey, const vector<scalar> &x, const vector<scalar> &y,
                              const vector<scalar> &Ex_grid, const vector<scalar> &Ey_grid, const Grid& grid, int Ntot);

void __LinearChargeInterpolation(vector<scalar> &rho, const vector<scalar> &x, const vector<scalar> &y, const Grid &grid, scalar charge, int Ntot);

void __LinearFieldInterpolationMPI(scalar efx[], scalar efy[], const scalar x[], const scalar y[],
                              const vector<scalar> &Ex, const vector<scalar> &Ey, const Grid& grid, const size_t Ntot);

void LinearFieldInterpolation(Particles& ptcl, const Matrix& Ex_grid, const Matrix& Ey_grid, const Grid& grid);

void LinearChargeInterpolation(Matrix& rho, const Particles& ptcl, const Grid& grid);

void LinearChargeInterpolationOMP(scalar rho[], const scalar z[], const scalar r[], const Grid& grid,
                               const scalar charge, const size_t Ntot, int num_of_threads);

void LinearChargeInterpolationMPI(Matrix& rhoMatrix, Particles& particles, const Grid& grid);

void LinearFieldInterpolation(scalar efz[], scalar efr[], const scalar z[], const scalar r[],
                              const scalar Ez[], const scalar Er[], const Grid& grid, size_t Ntot);

void LinearFieldInterpolationMPI(Particles &particles, Matrix &Ex, Matrix &Ey, const Grid& grid, int iteration);

void LinearFieldInterpolationOMP(scalar efz[], scalar efr[], const scalar z[], const scalar r[],
                              const scalar Ez[], const scalar Er[], const Grid& grid, size_t Ntot, int numThreads);


#endif //CPP_2D_PIC_INTERPOLATION_H
