//
// Created by Vladimir Smirnov on 03.10.2021.
//

#ifndef CPP_2D_PIC_INTERPOLATION_H
#define CPP_2D_PIC_INTERPOLATION_H

#include "../Tools/Matrix.h"
#include "../Tools/Grid.h"
#include "../Particles/Particles.h"
#define NUM_THREADS 4

void __LinearFieldInterpolation(scalar Ex[], scalar Ey[], const scalar x[], const scalar y[],
                              const scalar Ex_grid[], const scalar Ey_grid[], const Grid& grid, int Ntot);

void __LinearChargeInterpolation(scalar *rho, const scalar *x, const scalar *y, const Grid &grid, scalar charge, int Ntot);

void __LinearFieldInterpolationMPI(scalar efx[], scalar efy[], const scalar x[], const scalar y[],
                              const scalar *Ex, const scalar *Ey, const Grid& grid, const size_t Ntot);

void LinearFieldInterpolation(Particles& ptcl, const Matrix& Ex_grid, const Matrix& Ey_grid, const Grid& grid);

void LinearChargeInterpolation(Matrix& rho, const Particles& ptcl, const Grid& grid);

void LinearChargeInterpolationOMP(scalar rho[], const scalar z[], const scalar r[], const Grid& grid,
                               const scalar charge, const size_t Ntot, int num_of_threads);

void LinearChargeInterpolationMPI(Matrix& rhoMatrix, const Particles& particles, const Grid& grid);

void LinearFieldInterpolation(scalar efz[], scalar efr[], const scalar z[], const scalar r[],
                              const scalar Ez[], const scalar Er[], const Grid& grid, size_t Ntot);

void LinearFieldInterpolationMPI(Particles &particles, Matrix &Ex, Matrix &Ey, const Grid& grid);

void LinearFieldInterpolationOMP(scalar efz[], scalar efr[], const scalar z[], const scalar r[],
                              const scalar Ez[], const scalar Er[], const Grid& grid, size_t Ntot, int numThreads);


#endif //CPP_2D_PIC_INTERPOLATION_H
