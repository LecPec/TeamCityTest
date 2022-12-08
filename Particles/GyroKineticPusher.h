#ifndef CPP_2D_PIC_GYROKINETICPUSHER_H
#define CPP_2D_PIC_GYROKINETICPUSHER_H


#include "Pusher.h"
#include "GyroKineticParticles.h"


scalar DotProduct(const scalar v1[], const scalar v2[]);

void GyroUpdateSingleVelocityBoris(scalar &vel_x_c, scalar &vel_y_c, scalar &vel_z_c, scalar &vel_x, scalar &vel_y, scalar &vel_z,
                                   const scalar Ex, const scalar Ey, const scalar Bx, const scalar By, const scalar Bz,
                                   const scalar dt, const scalar q, const scalar m);

void GyroUpdateVelocity(vector<scalar> &vel_x_c, vector<scalar> &vel_y_c, vector<scalar> &vel_z_c, vector<scalar> &vel_x, vector<scalar> &vel_y, vector<scalar> &vel_z,
                        const vector<scalar> &Ex, const vector<scalar> &Ey, const vector<scalar> &Bx, const vector<scalar> &By, const vector<scalar> &Bz,
                        const scalar dt, const scalar q, const scalar m, const int Ntot);

void GyroParticlePush(vector<scalar> &pos_x, vector<scalar> &pos_y, vector<scalar> &vel_x_c, vector<scalar> &vel_y_c, vector<scalar> &vel_z_c,
                      vector<scalar> &vel_x, vector<scalar> &vel_y, vector<scalar> &vel_z,
                      const vector<scalar> &Ex, const vector<scalar> &Ey, const vector<scalar> &Bx, const vector<scalar> &By, const vector<scalar> &Bz,
                      const scalar dt, const scalar q, const scalar m, const int Ntot);



#endif //CPP_2D_PIC_GYROKINETICPUSHER_H
