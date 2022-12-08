//
// Created by Vladimir Smirnov on 02.10.2021.
//

#ifndef CPP_2D_PIC_PUSHER_H
#define CPP_2D_PIC_PUSHER_H

#include "../Tools/ProjectTypes.h"
#include "Particles.h"
#include <mpi.h>


void CrossProduct(const scalar v1[], const scalar v2[], scalar result[]);

void UpdateSingleVelocityBoris(scalar& vel_x, scalar& vel_y, scalar& vel_z, scalar Ex, scalar Ey,
                               scalar Bx, scalar By, scalar Bz, scalar dt, scalar q, scalar m);

void UpdateVelocity(vector<scalar> &vel_x, vector<scalar> &vel_y, vector<scalar> &vel_z, const vector<scalar> &Ex,
                    const vector<scalar> &Ey, const vector<scalar> &Bx, const vector<scalar> &By, const vector<scalar> &Bz,
                    const scalar dt, const scalar q, const scalar m, const int Ntot);

void UpdatePosition(vector<scalar> &pos_x, vector<scalar> &pos_y, const vector<scalar> &vel_x, const vector<scalar> &vel_y, const scalar dt,
                    const int Ntot);

void ParticlePush(vector<scalar> &pos_x, vector<scalar> &pos_y, vector<scalar> &vel_x, vector<scalar> &vel_y,
                  vector<scalar> &vel_z, const vector<scalar> &Ex, const vector<scalar> &Ey,
                  const vector<scalar> &Bx, const vector<scalar> &By, const vector<scalar> &Bz, const scalar dt, const scalar q,
                  const scalar m, const int Ntot);

void PusherMpi(Particles& particles, const scalar dt);
#endif //CPP_2D_PIC_PUSHER_H
