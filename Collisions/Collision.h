//
// Created by Vladimir Smirnov on 06.10.2021.
//

#ifndef CPP_2D_PIC_COLLISION_H
#define CPP_2D_PIC_COLLISION_H


#include <unordered_map>
#include <vector>
#include "../Particles/Particles.h"
#include "NeutralGas.h"
#include "EnergyCrossSection.h"

class Collision {
protected:
    scalar sigma;
    EnergyCrossSection* energy_sigma;
    scalar dt;
    NeutralGas* gas;
    scalar velocity_module(array<scalar, 3> vel) const;
    array<scalar, 3> sum(array<scalar, 3> vel1, array<scalar, 3> vel2) const;
    array<scalar, 3> subtraction(array<scalar, 3> vel1, array<scalar, 3> vel2) const;
    array<scalar, 3> multiplication_by_constant(array<scalar, 3> vel, scalar value) const;
    static array<scalar, 3> isotropic_velocity(scalar vel_module);
    array<scalar, 3> scattered_velocity(array<scalar, 3> vel_inc, scalar phi, scalar theta);
public:
    Particles* particles;
    Collision(scalar sigma, scalar dt, NeutralGas& gas, Particles& particles);
    Collision(EnergyCrossSection& sigma, scalar dt, NeutralGas& gas,
              Particles& particles);
    virtual void collision(int ptcl_idx) = 0;
    virtual scalar probability(int ptcl_idx) const = 0;
};

class ElectronNeutralElasticCollision : public Collision {
public:
    ElectronNeutralElasticCollision(scalar sigma, scalar dt, NeutralGas& gas, Particles& particles);
    ElectronNeutralElasticCollision(EnergyCrossSection& energy_sigma, scalar dt, NeutralGas& gas, Particles& particles);
    void collision(int ptcl_idx) override;
    void collisionNew(array<scalar, 3> &electronVel);
    scalar probability(int ptcl_idx) const override;
    scalar probabilityNew(const array<scalar, 3>& vel);
};

class IonNeutralElasticCollision : public Collision {
    // Hard-Sphere Interaction
    bool charge_exchange;
public:
    IonNeutralElasticCollision(scalar sigma, scalar dt, NeutralGas& gas, Particles& particles,
                               bool charge_exchange = true);
    IonNeutralElasticCollision(EnergyCrossSection& energy_sigma, scalar dt, NeutralGas& gas, Particles& particles,
                               bool charge_exchange = true);
    void collision(int ptcl_idx) override;
    scalar probability(int ptcl_idx) const override;
};

class Ionization : public Collision {
private:
    scalar ion_threshold;
public:
    Particles* ionized_particles;
    Ionization(scalar sigma, scalar ion_threshold, scalar dt, NeutralGas& gas, Particles& electrons,
               Particles& ions);
    Ionization(EnergyCrossSection& energy_sigma, scalar dt, NeutralGas& gas,
               Particles& electrons, Particles& ions);
    void collision(int ptcl_idx) override;
    void collisionNew(array<scalar, 2> &oldPtclPos, array<scalar, 3> &oldPtclVel,
                     array<scalar, 2> &newPtclPos, array<scalar, 3> &newPtclVel, array<scalar, 2> &ionPos, array<scalar, 3> &ionVel);
    scalar probabilityNew(const array<scalar, 3> &vel) const;
    scalar probability(int ptcl_idx) const override;
};


#endif //CPP_2D_PIC_COLLISION_H
