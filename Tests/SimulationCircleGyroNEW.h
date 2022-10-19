#ifndef CPP_2D_PIC_SIMULATIONCIRCLEGYRONEW_H
#define CPP_2D_PIC_SIMULATIONCIRCLEGYRONEW_H


#include <iostream>
#include <fstream>
#include <cmath>
#include <array>
#include <mpi.h>
#include <cassert>
#include "../Tools/Grid.h"
#include "../Particles/Particles.h"
#include "../Particles/Pusher.h"
#include "../Particles/GyroKineticParticles.h"
#include "../Particles/GyroKineticPusher.h"
#include "../ParticleLeaveEmission/ParticleEmission.h"
#include "../ParticleLeaveEmission/ParticleLeave.h"
#include "../Collisions/NeutralGas.h"
#include "../Collisions/EnergyCrossSection.h"
#include "../Collisions/Collision.h"
#include "../Collisions/NullCollisions.h"
#include "../Field/PoissonSolver.h"
#include "../Field/PoissonSolverCircle.h"
#include "../Tools/ParticlesLogger.h"
#include "../Tools/Logger.h"
#include "../Tools/Helpers.h"
#include "../Interpolation/Interpolation.h"
#include <omp.h>

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


void rho_filter(Matrix& rho, int iterations=2) {
    Matrix rho_f(rho.rows(), rho.columns());
    for(int it = 0; it < iterations; it++) {
        for(int i = 1; i < rho.rows() - 1 ; i++) {
            for(int j = 1; j < rho.columns() - 1; j++) {
                rho_f(i, j) = 0.5*(0.25 * (rho(i - 1, j) + rho(i + 1, j) + rho(i, j - 1) + rho(i, j + 1))) + 0.5*rho(i, j);
            }
        }
    }
    for(int i = 1; i < rho.rows() - 1; i++) {
        for(int j = 1; j < rho.columns() - 1; j++) {
            rho(i, j) = rho_f(i, j);
        }
    }
}

void rho_filter_new(Matrix& rho) {
    Matrix rho_f_1(rho.rows(), rho.columns());
    for(int i = 1; i < rho.rows() - 1 ; i++) {
        for(int j = 1; j < rho.columns() - 1; j++) {
            rho_f_1(i, j) = 4 * rho(i, j) + 2 * (rho(i - 1, j) + rho(i + 1, j) + rho(i, j - 1) + rho(i, j + 1)) + (
                                        rho(i - 1, j - 1) + rho(i + 1, j - 1) + rho(i - 1, j + 1) + rho(i + 1, j + 1));
            rho_f_1(i, j) = rho_f_1(i, j) / 16;
        }
    }
    Matrix rho_f_2(rho.rows(), rho.columns());
    for(int i = 1; i < rho.rows() - 1 ; i++) {
        for(int j = 1; j < rho.columns() - 1; j++) {
            rho_f_2(i, j) = 20 * rho_f_1(i, j) + (-1) * (rho_f_1(i - 1, j) + rho_f_1(i + 1, j) + rho(i, j - 1) + rho_f_1(i, j + 1)) + (-1) * (
                    rho_f_1(i - 1, j - 1) + rho_f_1(i + 1, j - 1) + rho_f_1(i - 1, j + 1) + rho_f_1(i + 1, j + 1));
            rho_f_2(i, j) = rho_f_2(i, j) / 12;
        }
    }
    for(int i = 1; i < rho.rows() - 1; i++) {
        for(int j = 1; j < rho.columns() - 1; j++) {
            rho(i, j) = rho_f_2(i, j);
        }
    }
}

void PrintAnodeCurrentDencity(const array<array<scalar, NX>, NX>& JElectrons, const array<array<scalar, NX>, NX>& JIons, int iter){
    ofstream fout("anode_current_hist/anode_current_" + to_string(iter) + ".txt");
    for (int row = 0; row < NX; ++row){
        for (int col = 0; col < NX; ++col){
            fout << JElectrons[row][col] + JIons[row][col] << ' ';
        }
        fout << endl;
    }
}
void PrintAnodeCurrentParticles(const array<array<scalar, NX>, NX>& electronsInAnodeCells, const array<array<scalar, NX>, NX>& ionsInAnodeCells, int iter, int timeStep, scalar dt){
    ofstream fout("anode_current_hist_particles/anode_current_" + to_string(iter) + ".txt");
    scalar dN_div_dt = 0.0;
    const scalar e = 1.6e-19;
    for (int row = 0; row < NX; ++row){
        for (int col = 0; col < NX; ++col){
            dN_div_dt = (ionsInAnodeCells[row][col] - electronsInAnodeCells[row][col]) / (timeStep * dt);
            fout << dN_div_dt * e << endl;
        }
        fout << endl;
    }
}

void PrintMethodTime(scalar t0, scalar t, string methodName, ofstream& fout, int iterationsNumber)
{
    fout << methodName << ' ' << (t - t0) / iterationsNumber << endl;
}

void test_simulation_circle_gyro_new() {
    int rank, commSize;
    MPI_Status status;
    MPI_Comm comm_cart;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    if (rank == 0)
        cout << "simulation gyro start" << endl;

    /*
     0 1386
    1000 1777
     */

    scalar scale = 0.02;
    int gyro_coeff = 100;
    int it_num = 1e8;
    scalar ptcls_per_cell = 1; //svs
    int seed = 1;

    // real system
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

    //scaled system
    scalar R_scaled = R * scale;
    scalar B_scaled = B / scale;
    scalar I_hot_scaled = I_hot * scale;
    scalar I_cold_scaled = I_cold * scale;
    scalar n_gas_scaled = n_gas / scale;
    scalar n_e_scaled = n_e / scale;
    scalar n_i_scaled = n_i / scale;

    //Parameters of scaled system
    scalar dt_e = 1 / (EV * B_scaled / E_M) / 10 * gyro_coeff;
    scalar dt_i = 1 / (EV * B_scaled / m_ion) / 10;
    scalar r_d = debye_radius(n_e_scaled, n_i_scaled, T_e, T_i) * 2;
    if (rank == 0)
    {
        cout << "dt_e = " << dt_e << endl;
        cout << "dt_i = " << dt_i << endl;
        cout << "debye_radius = " << r_d << " at energy = " << T_e / 11604.52500617 << "EV" << endl;
    }
    scalar dt = dt_e;

    /*****************************************************/
    // Grid init
    int Nx = R_scaled * 2 / r_d, Ny = R_scaled * 2 / r_d;
    scalar dx = r_d, dy = r_d;
    Grid grid(Nx, Ny, dx, dy);
    if (rank == 0)
        cout << "Grid: (" << Nx << "; " << Ny << ")" << endl;
    /*****************************************************/
    // Overall particle information
    scalar init_dens = n_e_scaled / 10;
    scalar radius_injection = 0.3 * Nx * dx;
    array<scalar, 2> center_injection = {0.5 * Nx * dx, 0.5 * Ny * dy};
    int Ntot = M_PI * pow(radius_injection / dx, 2) * ptcls_per_cell;
    scalar ptcls_per_macro = init_dens * M_PI * pow(radius_injection, 2) / Ntot;
    if (rank == 0)
    {
        cout << "ptcls_per_macro: " << ptcls_per_macro << endl;
        cout << "num_of_macro_ptcls: " << Ntot << endl;
    }
    /*****************************************************/
    // Particle Init
    scalar m_e = E_M;
    ParticlesConstant *ptclConstants = new ParticlesConstant();

    GyroKineticParticles electrons(m_e, -1*EV, 0, ptclConstants->ElectronsTypeString(), ptcls_per_macro);
    Particles ions(m_ion, 1*EV, 0, ptclConstants->IonsTypeString(), ptcls_per_macro);
    
    enum InitializationMode {InitialConfiguration = 0, FromFile = 1};
    array<scalar, 3> mf = {0, 0, B_scaled};
    scalar init_energy = 0.1*EV;
    SetParticlesData(InitializationMode::FromFile, electrons, mf, radius_injection, center_injection, Ntot, init_energy, seed, dt);
    SetParticlesData(InitializationMode::FromFile, ions, mf, radius_injection, center_injection, Ntot, init_energy, seed, dt);

    /*****************************************************/
    // Set particle leave on Anode
    scalar radius_anode = (Nx - 1) * dx / 2;
    int Nr_anode = (Nx - 1) / 2;
    array<scalar, 2> domain_center = {0.5 * Nx * dx, 0.5 * Ny * dy};
    /*****************************************************/
    // Ion cathode leave
    scalar gamma = 0.1; //secondary emission coefficient
    scalar ion_radius_leave_min = 0.01 * scale; // 1cm in real system
    scalar ion_radius_leave_max = 0.15 * scale; // 15cm in real system
    array<scalar, 2> ion_center_leave = {0.5 * Nx * dx, 0.5 * Ny * dy};
    scalar I_ion_leave = I_cold_scaled / (1. + gamma);
    int ion_leave_step = 500;
    scalar dt_ion_leave = ion_leave_step * dt; // !!!!!
    int Ntot_ion_leave = I_ion_leave * dt_ion_leave / EV / ptcls_per_macro;
    if (rank == 0)
        cout << "Ntot_ion_cathode: " << Ntot_ion_leave << endl;
    /*****************************************************/
    array<scalar, 2> electron_center_emission = {0.5 * Nx * dx, 0.5 * Ny * dy};

    // Electron emission cold cathode
    scalar energy_emission_cold = 100 * EV;
    scalar I_electron_emission_cold = gamma * I_ion_leave;
    int electron_emission_cold_step = 500;
    scalar dt_electron_emission_cold = electron_emission_cold_step * dt;
    int Ntot_electron_emission_cold = I_electron_emission_cold * dt_electron_emission_cold / EV / ptcls_per_macro;
    scalar electron_radius_emission_cold_min = ion_radius_leave_min;
    scalar electron_radius_emission_cold_max = ion_radius_leave_max;
    if (rank == 0)
        cout << "Ntot_electron_emission_cold: " << Ntot_electron_emission_cold << endl;
    /*****************************************************/
    // Electron emission hot cathode
    scalar energy_emission_hot = 100 * EV;
    scalar I_electron_emission_hot = I_hot_scaled;
    int electron_emission_hot_step = 500;
    scalar dt_electron_emission_hot = electron_emission_hot_step * dt;
    int Ntot_electron_emission_hot = I_electron_emission_hot * dt_electron_emission_hot / EV / ptcls_per_macro;
    scalar electron_radius_emission_hot_min = 0;
    scalar electron_radius_emission_hot_max = ion_radius_leave_min;
    if (rank == 0)
        cout << "Ntot_electron_emission_hot: " << Ntot_electron_emission_hot << endl;

    /*****************************************************/
    // Neutral gas init
    NeutralGas gas(n_gas_scaled, m_ion, T_gas);
    /*****************************************************/
    // PIC cycle parameters
    int ion_step = dt_i / dt_e;
    int collision_step_electron = 5;
    int collision_step_ion = ion_step * 5;
    if (rank == 0)
    {
        cout << "it_num: " << it_num << endl;
        cout << "ion integration step: " << ion_step << endl;
        cout << "electron_emission_cold_step: " << electron_emission_cold_step << endl;
        cout << "electron_emission_hot_step: " << electron_emission_hot_step << endl;
        cout << "ion_leave_step: " << ion_leave_step << endl;
        cout << "collision_step_ion: " << collision_step_ion << endl;
        cout << "collision_step_electron: " << collision_step_electron << endl;
        assert(collision_step_electron > 0);
        assert(collision_step_ion > 0);
        assert(ion_leave_step > 0);
        assert(electron_emission_cold_step > 0);
        assert(electron_emission_hot_step > 0);
        assert(ion_step > 0);
    }
    /*****************************************************/
    // Collisions init
    EnergyCrossSection elastic_electron_sigma("../Collisions/CrossSectionData/e-Ar_elastic.txt");
    EnergyCrossSection ionization_sigma("../Collisions/CrossSectionData/e-Ar_ionization.txt");
    EnergyCrossSection elastic_helium_sigma("../Collisions/CrossSectionData/Ar+-Ar_elastic.txt");

    ElectronNeutralElasticCollision electron_elastic(elastic_electron_sigma, collision_step_electron * dt, gas, electrons);
    Ionization ionization(ionization_sigma, collision_step_electron * dt, gas, electrons, ions);
    IonNeutralElasticCollision ion_elastic(elastic_helium_sigma, collision_step_ion * dt, gas, ions);
    /*****************************************************/
    // Init for sum of electron and ion rho
    Matrix rho(Nx, Ny);
    Matrix rho_e(Nx, Ny);
    Matrix rho_i(Nx, Ny);
    /*****************************************************/
    // Electric Field Init
    Matrix phi(Nx, Ny);
    Matrix Ex(Nx, Ny);
    Matrix Ey(Nx, Ny);
    scalar tol = 1e-4, betta = 1.93, max_iter = 1e6, it_conv_check = 100;
    InitDirichletConditionsCircle(phi, grid, Nr_anode);
    /*****************************************************/
    // Logger
    if (rank == 0)
    {
        string phi_file = "phi_hist/phi.txt";
        Matrix phi_log(Nx, Ny);
        clear_file(phi_file);

        string phi_oscillations_file = "phi_oscillations.txt";
        clear_file(phi_oscillations_file);

        string electron_anode_current_file = "electron_anode_current.txt";
        clear_file(electron_anode_current_file);

        string ion_anode_current_file = "ion_anode_current.txt";
        clear_file(ion_anode_current_file);

        string Ntot_file = "Ntot_(iter).txt";
        int Ntot_step = 1000, energy_step = Ntot_step, pos_step = 100000, vel_step = 100000;
        ParticlesLogger electrons_logger(electrons, "electrons");
        ParticlesLogger ions_logger(ions, "ions");

        ParticlesLogger electrons_traj(electrons, "electrons_traj");
        ParticlesLogger ions_traj(ions, "ions_traj");
    }
    //ParticlesLogger Ag_traj(Ag, "Ag_traj");
    //ParticlesLogger Pb_traj(Pb, "Pb_traj");
    int pos_traj_step_electron = 1, pos_traj_step_ion = ion_step;
    vector<int> track_ptcls = {0, 1, 2};
    /*****************************************************/
    scalar time_charge_interp = 0, time_field_interp = 0, time_pusher = 0, time_col = 0, time_anode = 0;
    scalar time_cold_cathode = 0, time_hot_cathode = 0, time_log = 0;
    scalar tmp_start_time;
    /*****************************************************/
    // PIC cycle
    clock_t start = clock();

    int Ntot_ionized = 0;
    int Ntot_cold_cathode_leave = 0;
    int Ntot_anode_leave = 0;
    int Ntot_hot_cathode_emission = 0;
    int Ntot_tmp;

    electrons.ZeroAnodeCurrent();
    ions.ZeroAnodeCurrent();

    int NtotElectrons = electrons.get_Ntot(), NtotIons = ions.get_Ntot();
    int logStep = 10000;

    double t0Field = 0, tField = 0;
    double t0Charge = 0, tCharge = 0;
    double t0Pois = 0, tPois = 0;
    double t0Push = 0, tPush = 0;
    double t0Coll = 0, tColl = 0;
    double tFull0 = 0, tFull = 0;
    int timeLogStep = logStep;
    ofstream timeFout("timeMpi.txt");
    if (rank == 0)
    {
        timeFout << "Number of procs: " << commSize << endl; 
        timeFout << "Iter NtotElectrons NtotIons NtotPtcls Charge Pois Field Push Coll Full" << endl;
    }
    for (int it = 0; it < it_num; it++) 
    {
        tFull0 += omp_get_wtime(); //time of the calculation

        /// <summary>
        /// Linear charge interpolation block
        /// Description: updates values of rho matrix according the coordinates of particles
        /// Arguments:
        ///             1) Matrix& rho - the charge density matrix of specific kind of particles
        ///             2) const Particles& particles - object of class Particles which represents data for specific particles
        ///             3) const Grid& grid - grid which descretes the 2D space with Ny rows, Nx columns, dx and dy grid steps
        /// Returns: void
        /// </summary>

        t0Charge += omp_get_wtime();
        LinearChargeInterpolationMPI(rho_e, electrons, grid); // mistake is somwhere here
        tCharge += omp_get_wtime();

        if (it % ion_step == 0) 
        {
            LinearChargeInterpolationMPI(rho_i, ions, grid);
        }

        /// <summary>
        /// Rho filter block
        /// Description: smoothes the charge density if the space so that it helps hold on the system stable
        /// Arguments:
        ///             1) Matrix& rho - the sum charge density matrix of both electrons and ions
        ///             2) int iterations - number of smoothing iterations
        /// Returns: void
        /// </summary>

        if (rank == 0)
        {
            rho = rho_e + rho_i;
            rho_filter(rho, 2);
        }

        /// <summary>
        /// Poisson solver block
        /// Description: calculates phi matrix of the potential of the space according to the given charge dencity
        /// Arguments:
        ///             1) Matrix& phi - matrix, containing the data of the potential distribution in the space
        ///             2) const Matrixs& rho - the given charge dencity distribution in the space
        ///             3) const Grid& grid - grid which descretes the 2D space with Ny rows, Nx columns, dx and dy grid steps
        ///             4) int R - number of grid cells surrounding the anode position
        ///             5) scalar tol - the convergence limit for the numerical scheme
        ///             6) scalar betta - parameter in SOR poisson solver
        ///             7) int max_iter - max number of iteration for convergence of the numerical scheme
        ///             8) int it_conv_check - the convergence of the scheme is checked every it_conv_check iteration
        /// Returns: void
        /// </summary>

        if (rank == 0)
        {
            t0Pois += omp_get_wtime();
            PoissonSolverCircle(phi, rho, grid, Nr_anode, tol, betta, max_iter, it_conv_check);
            tPois += omp_get_wtime();
        }

        /// <summary>
        /// Electric field calculation block
        /// Description: method which calculates the distribution of electric field int the space according to given distribution of the potential
        /// Arguments:
        ///             1) Matrix& Ex - matrix containing the data of the x coordinate of electric field
        ///             2) Matrix& Ey - matrix containing the data of the y coordinate of electric field
        ///             3) const Matrix& phi - matrix which contains data of the potential distribution in the space
        ///             4) const Grid& grid - grid which descretes the 2D space with Ny rows, Nx columns, dx and dy grid steps
        /// Returns: void
        /// </summary>

        if (rank == 0)
            compute_E(Ex, Ey, phi, grid);

        /// <summary>
        /// Linear field interpolation block
        /// Description: method which interpolates electric field from the nodes of grid to the particles situated in the current cell
        /// Arguments:
        ///             1) Particels& particles - object of class Particles containing the data about the current particles type
        ///             2) const Matrix& Ex - matrix containing the data of the x coordinate of electric field
        ///             3) const Matrix& Ey - matrix containing the data of the y coordinate of electric field
        ///             4) const Grid& grid - grid which descretes the 2D space with Ny rows, Nx columns, dx and dy grid steps
        /// Returns: void
        /// </summary>

        t0Field += omp_get_wtime();
        LinearFieldInterpolationMPI(electrons, Ex, Ey, grid);
        tField += omp_get_wtime();

        if (it % ion_step == 0)
        {
            LinearFieldInterpolationMPI(ions, Ex, Ey, grid);
        }

        /// <summary>
        /// Particles equations of movement integration block
        /// Description: two methods for integration of the movement equation for ordinary particles and the gyrokinetic ones
        /// Arguments:
        ///             1) scalar dt - time step for the integration
        /// Beware not to missmatch the type of particle and the method for the integration!!!
        /// Returns: void
        /// </summary>

        t0Push += omp_get_wtime();
        electrons.GyroPusherMPI(dt);
        tPush += omp_get_wtime();

        if (it % ion_step == 0) 
        {
            ions.pusherMPI(ion_step * dt);
        }

        /// <summary>
        /// Electrons-neurtal collisions block
        /// Description: method performing the elastic collisions of electrons and neutral particles and the ionization of the neutrals by electrons
        /// Arguments:
        ///             1) ElectronNeutralElasticCollision& elastic_electron - object which contains the data for
        ///                  collision implementing of electrons and neutrals
        ///             2) Ionization& ionization - object which contains data for implementation of ionization of neutral particle by electron
        /// Returns: void
        /// </summary>

        if (it % collision_step_electron == 0) 
        {
            Ntot_tmp = electrons.get_Ntot();

            t0Coll += omp_get_wtime();
            electron_null_collisionsNew(electron_elastic, ionization);
            tColl += omp_get_wtime();

            Ntot_ionized += electrons.get_Ntot() - Ntot_tmp;
        }
        
        if (rank == 0)
        {
            /// <summary>
            /// Ions-neurtal elastic collisions block
            /// Description: method performing the elastic collisions of ions and neutral particles
            /// Arguments:
            ///             1) IonNeutralElasticCollision& ion_elastic - object which contains the data for
            ///                  elastic collision implementing of ions and neutrals
            /// Returns: void
            /// </summary>

            if (it % collision_step_ion == 0) 
            {
                ion_null_collisions(ion_elastic);
            }
            Ntot_tmp = electrons.get_Ntot();

            /// <summary>
            /// Particle leaving from the system block
            /// Description: methods performing leaving of system by particles as soon as the reach the anode radius
            /// Arguments:
            ///             1) Particels& particles - object of class Particles containing the data about the current particles type
            ///             2) const Grid& grid - grid which descretes the 2D space with Ny rows, Nx columns, dx and dy grid steps
            ///             3) scalar radius_anode - the size of anode in length units
            ///             4) const array<scalar, 2>& domain_centre - coordinates of the centre of system
            ///             5) scalar dt - integration step
            ///             6) const Matrix& rho - matrix of charge dencity in the space for current calculation
            /// Returns: void
            /// </summary>

            particle_leave(electrons, grid, radius_anode, domain_center, dt, rho_e);
            
            Ntot_anode_leave += electrons.get_Ntot() - Ntot_tmp;

            if (it % ion_step == 0)
            {
                particle_leave(ions, grid, radius_anode, domain_center, dt, rho_i);
            }

            /// <summary>
            /// Cold anode electrons emission block
            /// Description: methods performing emission of electrons to the system
            /// Arguments:
            ///             1) Particels& particles - object of class Particles containing the data about the current particles type
            ///             2) scalar electron_radius_emission_cold_min - start radius of cold anode
            ///             3) scalar electron_radius_emission_cold_max - the cold anode border radius
            ///             4) const array<scalar, 2>& electron_centre_emission - coordinates of the emission centre
            ///             5) int N_electron_emission_cold - number of emitted electrons
            ///             6) scalar energy_emission_cold - energy of emitted electrons
            /// Returns: void
            /// </summary>

            if (it % electron_emission_cold_step == 0) 
            {
                Ntot_tmp = electrons.get_Ntot();

                particle_emission(electrons, electron_radius_emission_cold_min, electron_radius_emission_cold_max,
                                electron_center_emission, Ntot_electron_emission_cold, energy_emission_cold);

                Ntot_cold_cathode_leave += electrons.get_Ntot() - Ntot_tmp;
            }

            /// <summary>
            /// Additional particles leave block (especially for ions)
            /// Description: methods performing emission of electrons to the system
            /// Arguments:
            ///             1) Particels& particles - object of class Particles containing the data about the current particles type
            ///             2) scalar ion_radius_leave_cold_min - start of ions leaving zone
            ///             3) scalar ion_radius_leave_cold_max - end of ions leaving zone
            ///             4) const array<scalar, 2>& ion_centre_leave - coordinates of the ions leaving centre
            ///             5) int N_ion_leave - number of ions for additional leaving
            ///             6) scalar energy_emission_cold - energy of emitted electrons
            /// Returns: void
            /// </summary>

            if (it % ion_leave_step == 0) 
            {
                some_particle_leave(ions, ion_radius_leave_min, ion_radius_leave_max, ion_center_leave, Ntot_ion_leave);
            }

            /// <summary>
            /// Hot anode electrons emission block
            /// Description: methods performing emission of electrons to the system
            /// Arguments:
            ///             1) Particels& particles - object of class Particles containing the data about the current particles type
            ///             2) scalar electron_radius_emission_hot_min - start radius of hot anode
            ///             3) scalar electron_radius_emission_hot_max - the hot anode border radius
            ///             4) const array<scalar, 2>& electron_centre_emission - coordinates of the emission centre
            ///             5) int N_electron_emission_hot - number of emitted electrons
            ///             6) scalar energy_emission_hot - energy of emitted electrons
            /// Returns: void
            /// </summary>

            if (it % electron_emission_hot_step == 0) 
            {
                Ntot_tmp = electrons.get_Ntot();
                particle_emission(electrons, electron_radius_emission_hot_min, electron_radius_emission_hot_max,
                                electron_center_emission, Ntot_electron_emission_hot, energy_emission_hot);

                Ntot_hot_cathode_emission += electrons.get_Ntot() - Ntot_tmp;
            }

            tFull += omp_get_wtime();

            /// !!!END OF THE MAIN CYCLE!!!
            /// LOG OF DATA IS PERFORMED BELOW

            if (it == 10)
            {
                electrons.GetParticlesConfiguration();
                ions.GetParticlesConfiguration();
            }

            if (it % timeLogStep == 0)
            {
                timeFout << it << ' ' << electrons.get_Ntot() << ' ' << ions.get_Ntot() << ' ' << electrons.get_Ntot() + ions.get_Ntot()
                         << ' ' << (tCharge - t0Charge) / timeLogStep << ' '
                         << (tPois - t0Pois) / timeLogStep << ' ' << (tField - t0Field) / timeLogStep << ' ' << (tPush - t0Push) / timeLogStep << ' '
                         << (tColl - t0Coll) / timeLogStep << ' ' << tFull - tFull0 << endl;

                t0Charge = 0;
                tCharge = 0;
                t0Pois = 0;
                tPois = 0;
                t0Field = 0;
                tField = 0;
                t0Push = 0;
                tPush = 0;
                t0Coll = 0;
                tColl = 0;
                tFull0 = 0; 
                tFull = 0;
            }

            if (it % logStep == 0) 
            {
                cout << "iter" << it << endl;
                cout << "time_charge_interp: " << time_charge_interp << endl;
                cout << "time_field_interp: " << time_field_interp << endl;
                cout << "time_pusher: " << time_pusher << endl;
                cout << "time_col: " << time_col << endl;
                cout << "time_anode: " << time_anode << endl;
                cout << "time_hot_cathode: " << time_hot_cathode << endl;
                cout << "time_cold_cathode: " << time_cold_cathode << endl;
                cout << "time_log: " << time_log << endl;
                cout << endl;
            }
        }

        if (rank == 0)
        {
            if (it % logStep == 0){
                cout << electrons.get_Ntot() << ' ' << ions.get_Ntot() << endl;
                string phi_pth = "phi_hist/phi_" + to_string(it) + ".txt"; 
                string rho_i_pth = "rho_i_hist/rho_i_" + to_string(it) + ".txt"; 
                string rho_e_pth = "rho_e_hist/rho_e_" + to_string(it) + ".txt"; 

                phi.print_to_file(phi_pth);
                rho_i.print_to_file(rho_i_pth);
                rho_e.print_to_file(rho_e_pth);

                electrons.UpdateAnodeCurrent(500);
                ions.UpdateAnodeCurrent(500);
                //PrintAnodeCurrentDencity(electrons.GetJ(), ions.GetJ(), it);
                PrintAnodeCurrentParticles(electrons.numPtclsOnAnode, ions.numPtclsOnAnode, it, 500, dt);
                electrons.ZeroAnodeCurrent();
                ions.ZeroAnodeCurrent();
            }
        }
    }

    if (rank == 0)
    {
        cout << "time_charge_interp: " << time_charge_interp << endl;
        cout << "time_field_interp: " << time_field_interp << endl;
        cout << "time_pusher: " << time_pusher << endl;
        cout << "time_col: " << time_col << endl;
        cout << "time_anode: " << time_anode << endl;
        cout << "time_hot_cathode: " << time_hot_cathode << endl;
        cout << "time_cold_cathode: " << time_cold_cathode << endl;
    }

    //scalar end = omp_get_wtime();
    clock_t end = clock();
    cout << "elapsed time: " << (scalar)(end - start) / CLOCKS_PER_SEC << endl;
    //cout << "elapsed time: " << (scalar)(end - start) << endl;

    if (rank == 0)
        cout << "simulation gyro end" << endl;

    //MPI_Finalize();
}

#endif //CPP_2D_PIC_SIMULATIONCIRCLEGYRONEW_H
