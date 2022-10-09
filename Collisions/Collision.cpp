#include "Collision.h"
#include <cmath>
#include <random>
#include <stdlib.h>
#include <mpi.h>

#define K_b 1.380649e-23
#define EV 1.6021766208e-19
#define C 299792458
#define E_M 9.10938356e-31


array<scalar, 3> cross(const array<scalar, 3>& a, const array<scalar, 3>& b) {
    return {a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]};
}

Collision::Collision(scalar sigma, scalar dt, NeutralGas& gas, Particles& particles) :
        sigma(sigma), particles(&particles), dt(dt), gas(&gas) {
    EnergyCrossSection default_energy_sigma;
    energy_sigma = &default_energy_sigma;
}

Collision::Collision(EnergyCrossSection& energy_sigma, scalar dt,
                     NeutralGas &gas, Particles &particles) : energy_sigma(&energy_sigma), dt(dt), gas(&gas), particles(&particles) {
    sigma = -1;
}

scalar Collision::velocity_module(array<scalar, 3> vel) const {
    return sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
}

array<scalar, 3> Collision::subtraction(array<scalar, 3> vel1, array<scalar, 3> vel2) const {
    return {vel1[0] - vel2[0], vel1[1] - vel2[1], vel1[2] - vel2[2]};
}

array<scalar, 3> Collision::sum(array<scalar, 3> vel1, array<scalar, 3> vel2) const {
    return {vel1[0] + vel2[0], vel1[1] + vel2[1], vel1[2] + vel2[2]};
}

array<scalar, 3> Collision::multiplication_by_constant(array<scalar, 3> vel, scalar value) const {
    return {vel[0]*value, vel[1]*value, vel[2]*value};
}

array<scalar, 3> Collision::isotropic_velocity(scalar vel_module) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar theta = M_PI*distribution(generator);
    scalar phi = 2*M_PI*distribution(generator);
    return {vel_module*sin(theta)*cos(phi), vel_module*sin(theta)*sin(phi), vel_module*cos(theta)};
}

array<scalar, 3> Collision::scattered_velocity(array<scalar, 3> vel_inc, scalar phi, scalar theta) {
    scalar vel_module = velocity_module(vel_inc);
    vel_inc[0] = vel_inc[0] / vel_module;
    vel_inc[1] = vel_inc[1] / vel_module;
    vel_inc[2] = vel_inc[2] / vel_module;

    array<scalar, 3> i = {0, 1, 0};

    scalar sin_alpha = sin(acos(vel_inc[0]*i[0] + vel_inc[1]*i[1] + vel_inc[2]*i[2]));

    array<scalar, 3> vel_1 = {vel_inc[0]*cos(theta), vel_inc[1]*cos(theta), vel_inc[2]*cos(theta)};

    array<scalar, 3> cross_v_inc__i = cross(vel_inc, i);
    array<scalar, 3> vel_2 = {0, 0, 0};
    if (sin_alpha != 0) {
        vel_2 = {cross_v_inc__i[0] * sin(theta) * sin(phi) / (sin_alpha),
                 cross_v_inc__i[1] * sin(theta) * sin(phi) / (sin_alpha),
                 cross_v_inc__i[2] * sin(theta) * sin(phi) / (sin_alpha)};
    }

    array<scalar, 3> cross_v_inc__cross_i__v_inc = cross(vel_inc, cross(i, vel_inc));
    array<scalar, 3> vel_3 = {0, 0, 0};
    if (sin_alpha != 0) {
        vel_3 = {cross_v_inc__cross_i__v_inc[0] * sin(theta) * cos(phi) / (sin_alpha),
                 cross_v_inc__cross_i__v_inc[1] * sin(theta) * cos(phi) / (sin_alpha),
                 cross_v_inc__cross_i__v_inc[2] * sin(theta) * cos(phi) / (sin_alpha)};
    }

    array<scalar, 3> new_vel = {(vel_1[0] + vel_2[0] + vel_3[0]) * vel_module,
                                (vel_1[1] + vel_2[1] + vel_3[1]) * vel_module,
                                (vel_1[2] + vel_2[2] + vel_3[2]) * vel_module};
    return new_vel;
}





ElectronNeutralElasticCollision::ElectronNeutralElasticCollision(scalar sigma, scalar dt, NeutralGas& gas,
                                                                 Particles& particles) : Collision(sigma, dt, gas, particles) {}

ElectronNeutralElasticCollision::ElectronNeutralElasticCollision(EnergyCrossSection& energy_sigma,
                                                                 scalar dt, NeutralGas& gas, Particles& particles) : Collision(energy_sigma, dt, gas, particles) {}

void ElectronNeutralElasticCollision::collision(int ptcl_idx) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar theta;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);
    scalar phi = 2*M_PI*(distribution(generator));
    scalar vel_module = velocity_module(particles->get_velocity(ptcl_idx));
    scalar energy = particles->get_mass() * vel_module * vel_module / 2;
    if (energy / EV <= 100 and energy / EV > 0) {
        //theta = M_PI*distribution(generator);
        scalar energy_EV = energy / EV;
        theta = acos((2 + energy_EV - 2 * pow((1 + energy_EV), distribution(generator))) / energy_EV);
        if (1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)) < 0) {
            cout << "ion_mass/gas_mass is too small to use electron neutral collision" << endl;
            throw;
        }
        scalar new_vel_module = vel_module * sqrt(1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)));

        array<scalar, 3> vel_inc = particles->get_velocity(ptcl_idx);
        vel_inc[0] = vel_inc[0] / vel_module * new_vel_module;
        vel_inc[1] = vel_inc[1] / vel_module * new_vel_module;
        vel_inc[2] = vel_inc[2] / vel_module * new_vel_module;

	array<scalar, 3> new_vel = scattered_velocity(vel_inc, phi, theta);

        particles->set_velocity(ptcl_idx, new_vel);
    } else if (energy / EV > 100 and energy / EV <= 1e4) {
        scalar r_var = distribution(generator);
        scalar Z = 18; // Ar
        scalar p = sqrt(energy * 2 * E_M) / (E_M * C);
        scalar nu = 1.7e-5 * (0.556 - 0.0825 * log(energy / (E_M*C*C))) * pow(Z, (2/3.)) / (p * p);
        theta = acos(1 - 2 * nu * r_var / (1 + nu - r_var));
        if (1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)) < 0) {
            cout << "ion_mass/gas_mass is too small to use electron neutral collision" << endl;
            throw;
        }
        scalar new_vel_module = vel_module * sqrt(1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)));

        array<scalar, 3> vel_inc = particles->get_velocity(ptcl_idx);
        vel_inc[0] = vel_inc[0] / vel_module * new_vel_module;
        vel_inc[1] = vel_inc[1] / vel_module * new_vel_module;
        vel_inc[2] = vel_inc[2] / vel_module * new_vel_module;

	array<scalar, 3> new_vel = scattered_velocity(vel_inc, phi, theta);

        particles->set_velocity(ptcl_idx, new_vel);
    } else if(energy == 0) {
        array<scalar, 3> new_vel = gas->generate_velocity();
        particles->set_velocity(ptcl_idx, new_vel);
    } else {
        //cout << "electron energy > 1e4 eV" << endl;
        //throw;
	    scalar r_var = distribution(generator);
        scalar Z = 18; // Ar
        scalar p = sqrt(energy * 2 * E_M) / (E_M * C);
        //scalar nu = 1.7e-5 * (0.556 - 0.0825 * log(energy / (E_M*C*C))) * pow(Z, (2/3.)) / (p * p);
	    scalar nu = 1.2e-5 * pow(Z, (2/3.)) / (p * p);
        theta = acos(1 - 2 * nu * r_var / (1 + nu - r_var));
        if (1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)) < 0) {
            cout << "ion_mass/gas_mass is too small to use electron neutral collision" << endl;
            throw;
        }
        scalar new_vel_module = vel_module * sqrt(1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)));

        array<scalar, 3> vel_inc = particles->get_velocity(ptcl_idx);
        vel_inc[0] = vel_inc[0] / vel_module * new_vel_module;
        vel_inc[1] = vel_inc[1] / vel_module * new_vel_module;
        vel_inc[2] = vel_inc[2] / vel_module * new_vel_module;

        array<scalar, 3> new_vel = scattered_velocity(vel_inc, phi, theta);

        particles->set_velocity(ptcl_idx, new_vel);
    }
}

void ElectronNeutralElasticCollision::collisionNew(array<scalar, 3> &electronVel) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar theta;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);
    scalar phi = 2*M_PI*(distribution(generator));
    scalar vel_module = velocity_module(electronVel);
    scalar energy = particles->get_mass() * vel_module * vel_module / 2;
    if (energy / EV <= 100 and energy / EV > 0) {
        //theta = M_PI*distribution(generator);
        scalar energy_EV = energy / EV;
        theta = acos((2 + energy_EV - 2 * pow((1 + energy_EV), distribution(generator))) / energy_EV);
        if (1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)) < 0) {
            cout << "ion_mass/gas_mass is too small to use electron neutral collision" << endl;
            throw;
        }
        scalar new_vel_module = vel_module * sqrt(1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)));

        array<scalar, 3> vel_inc = electronVel;
        vel_inc[0] = vel_inc[0] / vel_module * new_vel_module;
        vel_inc[1] = vel_inc[1] / vel_module * new_vel_module;
        vel_inc[2] = vel_inc[2] / vel_module * new_vel_module;

        array<scalar, 3> new_vel = scattered_velocity(vel_inc, phi, theta);

        electronVel = new_vel;
    } else if (energy / EV > 100 and energy / EV <= 1e4) {
        scalar r_var = distribution(generator);
        scalar Z = 18; // Ar
        scalar p = sqrt(energy * 2 * E_M) / (E_M * C);
        scalar nu = 1.7e-5 * (0.556 - 0.0825 * log(energy / (E_M*C*C))) * pow(Z, (2/3.)) / (p * p);
        theta = acos(1 - 2 * nu * r_var / (1 + nu - r_var));
        if (1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)) < 0) {
            cout << "ion_mass/gas_mass is too small to use electron neutral collision" << endl;
            throw;
        }
        scalar new_vel_module = vel_module * sqrt(1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)));

        array<scalar, 3> vel_inc = electronVel;
        vel_inc[0] = vel_inc[0] / vel_module * new_vel_module;
        vel_inc[1] = vel_inc[1] / vel_module * new_vel_module;
        vel_inc[2] = vel_inc[2] / vel_module * new_vel_module;

	    array<scalar, 3> new_vel = scattered_velocity(vel_inc, phi, theta);

        electronVel = new_vel;
    } else if(energy == 0) {
        array<scalar, 3> new_vel = gas->generate_velocity();
        electronVel = new_vel;
    } else {
        //cout << "electron energy > 1e4 eV" << endl;
        //throw;
	    scalar r_var = distribution(generator);
        scalar Z = 18; // Ar
        scalar p = sqrt(energy * 2 * E_M) / (E_M * C);
        //scalar nu = 1.7e-5 * (0.556 - 0.0825 * log(energy / (E_M*C*C))) * pow(Z, (2/3.)) / (p * p);
	    scalar nu = 1.2e-5 * pow(Z, (2/3.)) / (p * p);
        theta = acos(1 - 2 * nu * r_var / (1 + nu - r_var));
        if (1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)) < 0) {
            cout << "ion_mass/gas_mass is too small to use electron neutral collision" << endl;
            throw;
        }
        scalar new_vel_module = vel_module * sqrt(1 - 2 * particles->get_mass() / gas->mass * (1 - cos(theta)));

        array<scalar, 3> vel_inc = electronVel;
        vel_inc[0] = vel_inc[0] / vel_module * new_vel_module;
        vel_inc[1] = vel_inc[1] / vel_module * new_vel_module;
        vel_inc[2] = vel_inc[2] / vel_module * new_vel_module;

        array<scalar, 3> new_vel = scattered_velocity(vel_inc, phi, theta);

        electronVel = new_vel;
    }
}

scalar ElectronNeutralElasticCollision::probability(int ptcl_idx) const {
    array<scalar, 3> vel = particles->get_velocity(ptcl_idx);
    scalar vel_module = velocity_module(vel);
    scalar energy = 0.5 * particles->get_mass() * vel_module * vel_module;
    if (sigma != -1) {
        return sigma * gas->n * vel_module * dt;
    } else if (not energy_sigma->empty()) {
        if (energy / EV > 100) {
            scalar p = sqrt(energy * 2 * E_M) / (E_M * C);
            scalar Z = 18; // Ar
            scalar nu = 1.7e-5 * (0.556 - 0.0825 * log(energy / (E_M*C*C))) * pow(Z, (2/3.)) / (p * p);
            scalar r_e = 2.817940326727e-15;
            scalar v = sqrt(2 * energy / E_M);
            scalar betta = v / C;
            scalar sigma_theory = Z * Z * M_PI * r_e * r_e / (betta * betta * p * p * nu * (nu + 1));
            //cout << "energy at energy > 50eV: " << sigma_theory * gas->n * vel_module * dt << " " << energy / EV << " " << sigma_theory << endl;
            //cout << "crosssec at energy > 50eV " << sigma_theory << endl;
            return sigma_theory * gas->n * vel_module * dt;
        }
        return energy_sigma->get_cross_section(energy) * gas->n * vel_module * dt;
    }
    else {
        cout << "cross section is not set!" << endl;
        throw;
    }
}

scalar ElectronNeutralElasticCollision::probabilityNew(const array<scalar, 3>& vel) {
    scalar vel_module = velocity_module(vel);
    scalar energy = 0.5 * particles->get_mass() * vel_module * vel_module;
    if (sigma != -1) {
        return sigma * gas->n * vel_module * dt;
    } else if (not energy_sigma->empty()) {
        if (energy / EV > 100) {
            scalar p = sqrt(energy * 2 * E_M) / (E_M * C);
            scalar Z = 18; // Ar
            scalar nu = 1.7e-5 * (0.556 - 0.0825 * log(energy / (E_M*C*C))) * pow(Z, (2/3.)) / (p * p);
            scalar r_e = 2.817940326727e-15;
            scalar v = sqrt(2 * energy / E_M);
            scalar betta = v / C;
            scalar sigma_theory = Z * Z * M_PI * r_e * r_e / (betta * betta * p * p * nu * (nu + 1));
            //cout << "energy at energy > 50eV: " << sigma_theory * gas->n * vel_module * dt << " " << energy / EV << " " << sigma_theory << endl;
            //cout << "crosssec at energy > 50eV " << sigma_theory << endl;
            return sigma_theory * gas->n * vel_module * dt;
        }
        return energy_sigma->get_cross_section(energy) * gas->n * vel_module * dt;
    }
    else {
        cout << "cross section is not set!" << endl;
        throw;
    }
}

IonNeutralElasticCollision::IonNeutralElasticCollision(scalar sigma, scalar dt, NeutralGas& gas,
                                                       Particles& particles, bool charge_exchange) : Collision(sigma, dt, gas, particles),
                                                                                                     charge_exchange(charge_exchange) {}

IonNeutralElasticCollision::IonNeutralElasticCollision(EnergyCrossSection& energy_sigma,
                                                       scalar dt, NeutralGas& gas, Particles& particles, bool charge_exchange) :
        Collision(energy_sigma, dt, gas, particles), charge_exchange(charge_exchange) {}

void IonNeutralElasticCollision::collision(int ptcl_idx) {
    //hard sphere approximation
    //cout << "IonNeutralElasticCollision" << endl;
    array<scalar, 3> new_ion_vel{}, gas_vel{};
    std::random_device rd;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0, 1.0);
    float random_num;
    if (charge_exchange)
        random_num = distribution(generator);
    else
        random_num = 0;
    if (random_num <= 0.5) {
        //elastic collision
        array<scalar, 3> ion_vel{}, delta_vel{}, delta_p{};
        scalar ion_mass, gas_mass;
        ion_vel = particles->get_velocity(ptcl_idx);
        gas_vel = gas->generate_velocity();
        ion_mass = particles->get_mass();
        gas_mass = gas->mass;
        for (int i = 0; i < 3; i++) {
            new_ion_vel[i] = (ion_vel[i] * ion_mass + gas_vel[i] * gas_mass + gas_mass * (gas_vel[i] - ion_vel[i])) /
                             (ion_mass + gas_mass);
        }
        particles->set_velocity(ptcl_idx, new_ion_vel);
    } else {
        //charge exchange
        new_ion_vel = gas->generate_velocity();
        particles->set_velocity(ptcl_idx, new_ion_vel);
    }
}

scalar IonNeutralElasticCollision::probability(int ptcl_idx) const {
    scalar vel_module = velocity_module(particles->get_velocity(ptcl_idx));
    if (sigma != -1) {
        scalar relative_vel, R_B, nu_A;
        R_B = K_b/gas->mass;
        nu_A = vel_module/(sqrt(2*R_B*gas->T));
        relative_vel = sqrt(2*R_B*gas->T)*((nu_A+1/(2*nu_A))*erf(nu_A) + 1/(sqrt(M_PI))*exp(-1*nu_A*nu_A));
        return gas->n*relative_vel*sigma*dt;
    } else if (not energy_sigma->empty()) {
        scalar energy = 0.5 * particles->get_mass() * vel_module * vel_module;
        scalar g_sigma = energy_sigma->get_cross_section(energy);
        return g_sigma * gas->n * dt;
    } else {
        cout << "cross section is not set!" << endl;
        throw;
    }
}




Ionization::Ionization(scalar sigma, scalar ion_threshold, scalar dt, NeutralGas& gas,
                       Particles& electrons, Particles& ions) : Collision(sigma, dt, gas, electrons),
                                                                ion_threshold(ion_threshold), ionized_particles(&ions) {}

Ionization::Ionization(EnergyCrossSection& energy_sigma, scalar dt, NeutralGas &gas,
                       Particles& electrons, Particles& ions) : Collision(energy_sigma, dt, gas, electrons),
                                                                ionized_particles(&ions) {
    ion_threshold = energy_sigma.get_energy_cross_section(0)[0];
}

void Ionization::collision(int ptcl_idx) {
    //cout << "Ionization process" << endl;
    array<scalar, 3> electron_vel = particles->get_velocity(ptcl_idx);
    scalar electron_vel_module = velocity_module(electron_vel);
    scalar electron_energy = particles->get_mass() * electron_vel_module * electron_vel_module / 2;
    scalar electron_energy_new = electron_energy - ion_threshold;
    if (electron_energy_new < 0) {
        return;
    }

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar theta, phi;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);

    scalar electron_energy_emitted = 10 * tan((distribution(generator)) * atan(electron_energy_new / (2*10)));
    scalar electron_energy_scattered = electron_energy_new - electron_energy_emitted;
    scalar electron_vel_module_emitted = sqrt(2 * electron_energy_emitted / particles->get_mass());
    scalar electron_vel_module_scattered = sqrt(2 * electron_energy_scattered / particles->get_mass());

    phi = 2*M_PI*(distribution(generator));
    scalar electron_energy_scattered_EV = electron_energy_scattered / EV;
    theta = acos((2 + electron_energy_scattered_EV - 2 * pow((1 + electron_energy_scattered_EV), (distribution(generator)))) / electron_energy_scattered_EV);
    array<scalar, 3> vel_tmp = {electron_vel[0] * electron_vel_module_scattered / electron_vel_module,
                                electron_vel[1] * electron_vel_module_scattered / electron_vel_module,
                                electron_vel[2] * electron_vel_module_scattered / electron_vel_module};
    array<scalar, 3> electron_vel_scattered = scattered_velocity(vel_tmp, phi, theta);

    phi = 2*M_PI*(distribution(generator));
    scalar electron_energy_emitted_EV = electron_energy_emitted / EV;
    theta = acos((2 + electron_energy_emitted_EV - 2 * pow((1 + electron_energy_emitted_EV), (distribution(generator)))) / electron_energy_emitted_EV);
    vel_tmp = {electron_vel[0] * electron_vel_module_emitted / electron_vel_module,
               electron_vel[1] * electron_vel_module_emitted / electron_vel_module,
               electron_vel[2] * electron_vel_module_emitted / electron_vel_module};
    array<scalar, 3> electron_vel_emitted = scattered_velocity(vel_tmp, phi, theta);

    array<scalar, 3> ion_vel_emitted = {0, 0, 0}; // null ion velocity (maybe better to set energy to gas T)
    
    particles->set_velocity(ptcl_idx, electron_vel_scattered);
    particles->append(particles->get_position(ptcl_idx), electron_vel_emitted);
    ionized_particles->append(particles->get_position(ptcl_idx), ion_vel_emitted);

}
void Ionization::collisionNew(array<scalar, 2> &oldElectronPos, array<scalar, 3> &oldElectronVel,
                     array<scalar, 2> &newElectronPos, array<scalar, 3> &newElectronVel, array<scalar, 2> &ionPos, array<scalar, 3> &ionVel)
{
    array<scalar, 3> electron_vel = oldElectronVel;
    scalar electron_vel_module = velocity_module(electron_vel);
    scalar electron_energy = particles->get_mass() * electron_vel_module * electron_vel_module / 2;
    scalar electron_energy_new = electron_energy - ion_threshold;
    if (electron_energy_new < 0) {
        return;
    }

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar theta, phi;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);

    scalar electron_energy_emitted = 10 * tan(distribution(generator) * atan(electron_energy_new / (2*10)));
    scalar electron_energy_scattered = electron_energy_new - electron_energy_emitted;
    scalar electron_vel_module_emitted = sqrt(2 * electron_energy_emitted / particles->get_mass());
    scalar electron_vel_module_scattered = sqrt(2 * electron_energy_scattered / particles->get_mass());

    phi = 2*M_PI*distribution(generator);
    scalar electron_energy_scattered_EV = electron_energy_scattered / EV;
    theta = acos((2 + electron_energy_scattered_EV - 2 * pow((1 + electron_energy_scattered_EV), distribution(generator))) / electron_energy_scattered_EV);
    array<scalar, 3> vel_tmp = {electron_vel[0] * electron_vel_module_scattered / electron_vel_module,
                                electron_vel[1] * electron_vel_module_scattered / electron_vel_module,
                                electron_vel[2] * electron_vel_module_scattered / electron_vel_module};
    array<scalar, 3> electron_vel_scattered = scattered_velocity(vel_tmp, phi, theta);

    phi = 2*M_PI*distribution(generator);
    scalar electron_energy_emitted_EV = electron_energy_emitted / EV;
    theta = acos((2 + electron_energy_emitted_EV - 2 * pow((1 + electron_energy_emitted_EV), distribution(generator))) / electron_energy_emitted_EV);
    vel_tmp = {electron_vel[0] * electron_vel_module_emitted / electron_vel_module,
               electron_vel[1] * electron_vel_module_emitted / electron_vel_module,
               electron_vel[2] * electron_vel_module_emitted / electron_vel_module};
    array<scalar, 3> electron_vel_emitted = scattered_velocity(vel_tmp, phi, theta);

    array<scalar, 3> ion_vel_emitted = {0, 0, 0}; // null ion velocity (maybe better to set energy to gas T)
    
    //particles->set_velocity(ptcl_idx, electron_vel_scattered);
    oldElectronVel = electron_vel_scattered;
    //particles->append(particles->get_position(ptcl_idx), electron_vel_emitted);
    newElectronPos = oldElectronPos;
    newElectronVel = electron_vel_emitted;
    //ionized_particles->append(particles->get_position(ptcl_idx), ion_vel_emitted);
    ionPos = oldElectronPos;
    ionVel = ion_vel_emitted;
}

scalar Ionization::probability(int ptcl_idx) const {
    array<scalar, 3> vel = particles->get_velocity(ptcl_idx);
    scalar vel_module = velocity_module(vel);
    scalar energy = 0.5*vel_module*vel_module*particles->get_mass();
    if (sigma != -1) {
        if (energy >= ion_threshold) {
            return sigma * gas->n * vel_module * dt;
        }
        return 0;
    } else if (not energy_sigma->empty()) {
        return energy_sigma->get_cross_section(energy) * gas->n * vel_module * dt;
    } else {
        cout << "cross section is not set!" << endl;
        throw;
    }
}

scalar Ionization::probabilityNew(const array<scalar, 3> &vel) const {
    scalar vel_module = velocity_module(vel);
    scalar energy = 0.5*vel_module*vel_module*particles->get_mass();
    if (sigma != -1) {
        if (energy >= ion_threshold) {
            return sigma * gas->n * vel_module * dt;
        }
        return 0;
    } else if (not energy_sigma->empty()) {
        return energy_sigma->get_cross_section(energy) * gas->n * vel_module * dt;
    } else {
        cout << "cross section is not set!" << endl;
        throw;
    }
}
