//
// Created by Vladimir Smirnov on 27.10.2021.
//

#ifndef CPP_2D_PIC_GYROKINETICPARTICLES_H
#define CPP_2D_PIC_GYROKINETICPARTICLES_H


#include "Particles.h"

class GyroKineticParticles : public Particles {
public:
    vector<scalar> vx_c;
    vector<scalar> vy_c;
    vector<scalar> vz_c;
    vector<scalar> vx_c_;
    vector<scalar> vy_c_;
    vector<scalar> vz_c_;
    GyroKineticParticles(scalar m, scalar q, int N, string type, scalar N_per_macro = 1);
    void append(const array<scalar, 2>& position, const array<scalar, 3>& velocity) override;
    void pop(int ptcl_idx) override;
    void pop_list(const vector<int>& ptcl_idx_list) override;
    void vel_pusher(scalar  dt) override;
    void pusher(scalar  dt) override;
    void GyroPusherMPI(scalar dt, int iteration);
    void set_velocity(const int ptcl_idx, const array<scalar, 3>& velocity) override;
    array<scalar, 3> get_velocity(const int ptcl_idx) const override;

    //Particles configuration log
    void GetParticlesConfiguration() override;
    void InitConfigurationFromFile() override;

    void Resize(int newSize) override;
    void ShrinkToFit() override;
};


#endif //CPP_2D_PIC_GYROKINETICPARTICLES_H
