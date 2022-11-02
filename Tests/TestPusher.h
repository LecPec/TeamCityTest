#include "../Particles/Particles.h"
#include "../Particles/GyroKineticParticles.h"
#include "../ParticleLeaveEmission/ParticleEmission.h"
#include "cmath"
#include <omp.h>
#include <mpi.h>
#include <fstream>

#define E_M 9.10938356e-31
#define EV 1.6021766208e-19
#define EPSILON_0 8.854187817620389e-12
#define K_B 1.380649e-23

using namespace std;

scalar debye_radius(scalar n_e, scalar n_ion, scalar T_e, scalar T_ion) {
    scalar k_cgs = 1.38e-16;
    scalar q_cgs = 4.8e-10;
    scalar n_e_cgs = n_e / 1e6;
    scalar n_ion_cgs = n_ion / 1e6;
    scalar d_cgs = pow(4*M_PI*q_cgs*q_cgs*n_ion_cgs/(k_cgs*T_ion) + 4*M_PI*q_cgs*q_cgs*n_e_cgs/(k_cgs*T_e), -0.5);
    scalar d_si = d_cgs / 100.;
    return d_si;
}

void TestPusherMpi()
{
    int rank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    scalar m_e = E_M;

    scalar R = 0.26;
    scalar B = 0.1;
    scalar n_e = 1e15; // m^(-3)
    scalar n_i = 1e15; // m^(-3)
    scalar T_e = 10 * 11604.52500617; //10eV
    scalar T_i = 10 * 11604.52500617; //10eV
    scalar I_hot = 6; // Ampere
    scalar I_cold = 4; // Ampere
    scalar p = 4 * 1e-3 * 133; // 4mTorr to Pascal
    scalar T_gas = 500; // Kelvin
    scalar n_gas = p / (K_B * T_gas);
    scalar m_ion = 500 * E_M; //6.6464731E-27; // He+
    scalar scale = 0.02;
    scalar gyro_coeff = 100;
    int ptcls_per_cell = 1;

    scalar R_scaled = R * scale;
    scalar B_scaled = B / scale;
    scalar I_hot_scaled = I_hot * scale;
    scalar I_cold_scaled = I_cold * scale;
    scalar n_gas_scaled = n_gas / scale;
    scalar n_e_scaled = n_e / scale;
    scalar n_i_scaled = n_i / scale;

    scalar dt_e = 1 / (EV * B_scaled / E_M) / 10 * gyro_coeff;
    scalar dt_i = 1 / (EV * B_scaled / m_ion) / 10;
    scalar dt = dt_e;
    int seed = 1;
    scalar r_d = debye_radius(n_e_scaled, n_i_scaled, T_e, T_i) * 2;

    int Nx = R_scaled * 2 / r_d, Ny = R_scaled * 2 / r_d;
    scalar dx = r_d, dy = r_d;
    Grid grid(Nx, Ny, dx, dy);
    scalar init_dens = n_e_scaled / 10;
    scalar radius_injection = 0.3 * Nx * dx;
    array<scalar, 2> center_injection = {0.5 * Nx * dx, 0.5 * Ny * dy};
    int Ntot = M_PI * pow(radius_injection / dx, 2) * ptcls_per_cell;
    scalar ptcls_per_macro = init_dens * M_PI * pow(radius_injection, 2) / Ntot;
    

    ParticlesConstant *ptclConstants = new ParticlesConstant();

    GyroKineticParticles electrons(m_e, -1*EV, 0, ptclConstants->ElectronsTypeString(), ptcls_per_macro);
    Particles ions(m_ion, 1*EV, 0, ptclConstants->IonsTypeString(), ptcls_per_macro);
    
    enum InitializationMode {InitialConfiguration = 0, FromFile = 1};
    array<scalar, 3> mf = {0, 0, 5};
    scalar init_energy = 0.1*EV;
    SetParticlesData(InitializationMode::InitialConfiguration, electrons, mf, radius_injection, center_injection, Ntot, init_energy, seed, dt);
    SetParticlesData(InitializationMode::InitialConfiguration, ions, mf, radius_injection, center_injection, Ntot, init_energy, seed, dt);


    scalar energy = EV;
    scalar vel = sqrt(2*energy/m_e);
    scalar velI = sqrt(2*energy/m_ion);
    for (int i = 0; i < Ntot; i++) {
        electrons.set_velocity(i, {vel, vel, vel}); electrons.set_position(i, {Nx * dx / 2, Ny * dy / 2});
        ions.set_velocity(i, {velI, velI, velI}); ions.set_position(i, {Nx * dx / 4, Ny * dy / 4});
    }

    double t0 = omp_get_wtime();
    for (int i = 0; i < 500; ++i)
        electrons.GyroPusherMPI(dt);
    double t = omp_get_wtime();

    if (rank == 0)
    {
        ofstream fout("time.txt", ios::app);
        fout << commSize << ' ' << (t - t0) / 500 << endl;
    }
}

void TestPusher()
{
    scalar m_e = E_M;

    scalar R = 0.26;
    scalar B = 0.1;
    scalar n_e = 1e15; // m^(-3)
    scalar n_i = 1e15; // m^(-3)
    scalar T_e = 10 * 11604.52500617; //10eV
    scalar T_i = 10 * 11604.52500617; //10eV
    scalar I_hot = 6; // Ampere
    scalar I_cold = 4; // Ampere
    scalar p = 4 * 1e-3 * 133; // 4mTorr to Pascal
    scalar T_gas = 500; // Kelvin
    scalar n_gas = p / (K_B * T_gas);
    scalar m_ion = 500 * E_M; //6.6464731E-27; // He+
    scalar scale = 0.02;
    scalar gyro_coeff = 100;
    int ptcls_per_cell = 1;

    scalar R_scaled = R * scale;
    scalar B_scaled = B / scale;
    scalar I_hot_scaled = I_hot * scale;
    scalar I_cold_scaled = I_cold * scale;
    scalar n_gas_scaled = n_gas / scale;
    scalar n_e_scaled = n_e / scale;
    scalar n_i_scaled = n_i / scale;

    scalar dt_e = 1 / (EV * B_scaled / E_M) / 10 * gyro_coeff;
    scalar dt_i = 1 / (EV * B_scaled / m_ion) / 10;
    scalar dt = dt_e;
    int seed = 1;
    scalar r_d = debye_radius(n_e_scaled, n_i_scaled, T_e, T_i) * 2;

    int Nx = R_scaled * 2 / r_d, Ny = R_scaled * 2 / r_d;
    scalar dx = r_d, dy = r_d;
    Grid grid(Nx, Ny, dx, dy);
    scalar init_dens = n_e_scaled / 10;
    scalar radius_injection = 0.3 * Nx * dx;
    array<scalar, 2> center_injection = {0.5 * Nx * dx, 0.5 * Ny * dy};
    int Ntot = M_PI * pow(radius_injection / dx, 2) * ptcls_per_cell;
    scalar ptcls_per_macro = init_dens * M_PI * pow(radius_injection, 2) / Ntot;
    

    ParticlesConstant *ptclConstants = new ParticlesConstant();

    GyroKineticParticles electrons(m_e, -1*EV, 0, ptclConstants->ElectronsTypeString(), ptcls_per_macro);
    Particles ions(m_ion, 1*EV, 0, ptclConstants->IonsTypeString(), ptcls_per_macro);
    
    enum InitializationMode {InitialConfiguration = 0, FromFile = 1};
    array<scalar, 3> mf = {0, 0, 5};
    scalar init_energy = 0.1*EV;
    SetParticlesData(InitializationMode::InitialConfiguration, electrons, mf, radius_injection, center_injection, Ntot, init_energy, seed, dt);
    SetParticlesData(InitializationMode::InitialConfiguration, ions, mf, radius_injection, center_injection, Ntot, init_energy, seed, dt);


    scalar energy = EV;
    scalar vel = sqrt(2*energy/m_e);
    scalar velI = sqrt(2*energy/m_ion);
    for (int i = 0; i < Ntot; i++) {
        electrons.set_velocity(i, {vel, vel, vel}); electrons.set_position(i, {Nx * dx / 2, Ny * dy / 2});
        ions.set_velocity(i, {velI, velI, velI}); ions.set_position(i, {Nx * dx / 4, Ny * dy / 4});
    }

    double t0 = omp_get_wtime();
    for (int i = 0; i < 5000; ++i)
        electrons.pusher(dt);
    double t = omp_get_wtime();

    ofstream fout("time.txt", ios::app);
    fout << 1 << ' ' << (t - t0) / 5000 << endl;
}