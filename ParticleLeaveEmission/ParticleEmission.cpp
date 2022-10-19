//
// Created by Vladimir Smirnov on 06.10.2021.
//

#include "ParticleEmission.h"
#include <cmath>
#include <random>
#include <mpi.h>

scalar GenRandomNormal(scalar _std)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);

    scalar u = rand() / (scalar)RAND_MAX;
    scalar v = rand() / (scalar)RAND_MAX;

    scalar res = sqrt(-2 * log(u)) * cos(2 * M_PI * v);

    return res * _std;
}

void particle_emission(Particles &particles, const scalar emission_radius, const array<scalar, 2>& circle_center,
                       const int Ntot_emission, const scalar energy) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);
    scalar radius, theta;
    array<scalar, 2> pos{};
    array<scalar, 3> vel{};
    for(int ptcl_idx = 0; ptcl_idx < Ntot_emission; ptcl_idx++) {
        //radius = emission_radius * distribution(generator);
        //theta = 2*M_PI*distribution(generator);
        //pos[0] = radius * cos(theta) + circle_center[0];
        //pos[1] = radius * sin(theta) + circle_center[1];
        pos[0] = circle_center[0] - emission_radius + (distribution(generator)) * 2 * emission_radius;
        pos[1] = circle_center[1] - emission_radius + (distribution(generator)) * 2 * emission_radius;
        vel[0] = 0;
        vel[1] = 0;
        vel[2] = sqrt(2 * energy / particles.get_mass());
        particles.append(pos, vel);
    }
}

void particle_emission(Particles &particles, const scalar emission_radius_min, const scalar emission_radius_max,
                           const array<scalar, 2>& circle_center,
                           const int Ntot_emission, const scalar energy) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);
    scalar radius, theta;
    array<scalar, 2> pos{};
    array<scalar, 3> vel{};
    for(int ptcl_idx = 0; ptcl_idx < Ntot_emission; ptcl_idx++) {
        radius = emission_radius_min + (emission_radius_max - emission_radius_min) * sqrt(distribution(generator));
        theta = 2 * M_PI * distribution(generator);
        pos[0] = radius * cos(theta) + circle_center[0];
        pos[1] = radius * sin(theta) + circle_center[1];
        vel[0] = 0;
        vel[1] = 0;
        vel[2] = sqrt(2 * energy / particles.get_mass());
        particles.append(pos, vel);
    }
}

void init_particle_emission(Particles &particles, const scalar emission_radius, const array<scalar, 2> &circle_center,
                            const int Ntot_emission, const scalar energy, int seed) {
    std::random_device rd;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar _std = sqrt(2*energy/(3*particles.get_mass()));
    std::normal_distribution<scalar> distribution_normal(0.0, _std);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);
    scalar radius, theta;
    array<scalar, 2> pos{};
    array<scalar, 3> vel{};
    for(int ptcl_idx = 0; ptcl_idx < Ntot_emission; ptcl_idx++) {
        radius = emission_radius * sqrt(distribution(generator));
        theta = 2*M_PI*distribution(generator);
        pos[0] = radius * cos(theta) + circle_center[0];
        pos[1] = radius * sin(theta) + circle_center[1];
        //pos[0] = circle_center[0] - emission_radius + distribution(generator) * 2 * emission_radius;
        //pos[1] = circle_center[1] - emission_radius + distribution(generator) * 2 * emission_radius;

        while (true) {
            vel[0] = distribution_normal(generator);
            vel[1] = distribution_normal(generator);
            vel[2] = distribution_normal(generator);
            if (sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]) < _std * 3) {
                break;
            }
        }

        particles.append(pos, vel);
    }
}

void SetParticlesData(bool mode, Particles& particles, array<scalar, 3> mf, scalar radius_injection, array<scalar, 2>
                     center_injection, int Ntot, scalar init_energy, int seed, scalar dt)
{
    if (mode == 0)
    {
        init_particle_emission(particles, radius_injection, center_injection, Ntot, init_energy, seed);
        particles.set_const_magnetic_field(mf);
        particles.vel_pusher(-0.5 * dt);
    }
    else
    {
        particles.InitConfigurationFromFile();
    }
}
