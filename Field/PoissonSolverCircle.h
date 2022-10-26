//
// Created by Vladimir Smirnov on 06.11.2021.
//

#ifndef CPP_2D_PIC_POISSONSOLVERCIRCLE_H
#define CPP_2D_PIC_POISSONSOLVERCIRCLE_H


#include "../Tools/Matrix.h"
#include "../Tools/Grid.h"
#define NUM_THREADS 4

void InitDirichletConditionsCircle(Matrix& phi, const Grid& grid, const int R);

bool convergence_check_circle(const Matrix& phi, const Matrix& rho, const Grid& grid, const int R, const scalar tol, const scalar betta);

void PoissonSolverCircle(Matrix& phi, const Matrix& rho, const Grid& grid, const int R, const scalar tol,
                         scalar betta = 1.93, int max_iter = 1e6, int it_conv_check=1);
void OneIterationPoissonSolver(Matrix& phi, const Matrix& rho, const Grid& grid, const int R, scalar betta, int it);


#endif //CPP_2D_PIC_POISSONSOLVERCIRCLE_H
