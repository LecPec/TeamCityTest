//
// Created by Vladimir Smirnov on 06.10.2021.
//

#ifndef CPP_2D_PIC_PARTICLEEMISSION_H
#define CPP_2D_PIC_PARTICLEEMISSION_H


#include "../Particles/Particles.h"

scalar GenRandomNormal(scalar _std);
void particle_emission(Particles& particles, const scalar emission_radius, const array<scalar, 2>& circle_center,
                       const int Ntot_emission, const scalar energy);

void particle_emission(Particles &particles, const scalar emission_radius_min, const scalar emission_radius_max,
                       const array<scalar, 2>& circle_center, const int Ntot_emission, const scalar energy);

void init_particle_emission(Particles& particles, const scalar emission_radius, const array<scalar, 2>& circle_center,
                            const int Ntot_emission, const scalar energy, int seed);

/// Function setting data to the particles
/// mode = 1 in case of file initiation
/// mode = 0 in case of initial calculation
void SetParticlesData(bool mode, Particles& particles, array<scalar, 3> mf, scalar radius_injection, array<scalar, 2> center_injection, int Ntot, scalar init_energy, int seed, scalar dt);

#endif //CPP_2D_PIC_PARTICLEEMISSION_H
