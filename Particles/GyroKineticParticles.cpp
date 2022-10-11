//
// Created by Vladimir Smirnov on 27.10.2021.
//

#include "GyroKineticParticles.h"
#include "GyroKineticPusher.h"
#include <cassert>

void GyroKineticParticles::vel_pusher(scalar dt) {
    GyroUpdateVelocity(vx_c.data(), vy_c.data(), vz_c.data(),
            vx.data(), vy.data(), vz.data(),Ex.data(), Ey.data(),
            Bx.data(), By.data(), Bz.data(), dt, charge, mass, Ntot);
}

void GyroKineticParticles::pusher(scalar dt) {
    GyroParticlePush(x.data(), y.data(), vx_c.data(), vy_c.data(), vz_c.data(),
            vx.data(), vy.data(), vz.data(), Ex.data(), Ey.data(),
            Bx.data(), By.data(), Bz.data(), dt, charge, mass, Ntot);
}

void GyroKineticParticles::set_velocity(const int ptcl_idx, const array<scalar, 3> &velocity) {
    Particles::set_velocity(ptcl_idx, velocity);
    vx_c[ptcl_idx] = velocity[0];
    vy_c[ptcl_idx] = velocity[1];
    vz_c[ptcl_idx] = velocity[2];
}

array<scalar, 3> GyroKineticParticles::get_velocity(const int ptcl_idx) const {
    return Particles::get_velocity(ptcl_idx);
}

GyroKineticParticles::GyroKineticParticles(scalar m, scalar q, int N, string type, scalar N_per_macro) : Particles(m, q, N, type,
                                                                                                      N_per_macro) {
    vx_c.resize(Ntot);
    vy_c.resize(Ntot);
    vz_c.resize(Ntot);
}

void GyroKineticParticles::append(const array<scalar, 2> &position, const array<scalar, 3> &velocity) {
    Particles::append(position, velocity);
    vx_c.push_back(velocity[0]);
    vy_c.push_back(velocity[1]);
    vz_c.push_back(velocity[2]);
}

void GyroKineticParticles::pop(int ptcl_idx) {
    Particles::pop(ptcl_idx);
    vx_c.pop_back();
    vy_c.pop_back();
    vz_c.pop_back();
}

void GyroKineticParticles::pop_list(const vector<int> &ptcl_idx_list) {
    //Particles::pop_list(ptcl_idx_list);
    assert(x.size() == Ntot);
    assert(ptcl_idx_list.size() <= Ntot);

    int leave_ptcl_idx;
    int main_ptcl_idx = Ntot - 1;
    for(int i = 0; i < ptcl_idx_list.size(); i++) {
        leave_ptcl_idx = ptcl_idx_list[i];
        swap(x[leave_ptcl_idx], x[main_ptcl_idx]);
        swap(y[leave_ptcl_idx], y[main_ptcl_idx]);
        swap(vx_c[leave_ptcl_idx], vx_c[main_ptcl_idx]);
        swap(vy_c[leave_ptcl_idx], vy_c[main_ptcl_idx]);
        swap(vz_c[leave_ptcl_idx], vz_c[main_ptcl_idx]);
        swap(vx[leave_ptcl_idx], vx[main_ptcl_idx]);
        swap(vy[leave_ptcl_idx], vy[main_ptcl_idx]);
        swap(vz[leave_ptcl_idx], vz[main_ptcl_idx]);
        swap(Bx[leave_ptcl_idx], Bx[main_ptcl_idx]);
        swap(By[leave_ptcl_idx], By[main_ptcl_idx]);
        swap(Bz[leave_ptcl_idx], Bz[main_ptcl_idx]);
        swap(Ex[leave_ptcl_idx], Ex[main_ptcl_idx]);
        swap(Ey[leave_ptcl_idx], Ey[main_ptcl_idx]);
        main_ptcl_idx--;
    }

    for(int i = 0; i < ptcl_idx_list.size(); i++) {
        x.pop_back();
        y.pop_back();
        vx_c.pop_back();
        vy_c.pop_back();
        vz_c.pop_back();
        vx.pop_back();
        vy.pop_back();
        vz.pop_back();
        Bx.pop_back();
        By.pop_back();
        Bz.pop_back();
        Ex.pop_back();
        Ey.pop_back();
    }
    Ntot = x.size();
}

void GyroKineticParticles::GyroPusherMPI(scalar dt)
{
    int rank, commSize;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    scalar mass = this->get_mass() * this->get_ptcls_per_macro();
    scalar charge = this->get_charge() * this->get_ptcls_per_macro();

    if (rank == 0){
        int Ntot = get_Ntot();
        int Ntot_per_proc = Ntot / commSize;
        int Ntot_per_0_proc = Ntot / commSize + Ntot % commSize;

        for (int proc = 1; proc < commSize; ++proc){
            MPI_Send(&Ntot_per_proc, 1, MPI_INT, proc, 5665, MPI_COMM_WORLD);
            MPI_Send(&(x)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 660 * 71, MPI_COMM_WORLD);
            MPI_Send(&(y)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 661 * 71, MPI_COMM_WORLD);
            MPI_Send(&(vx_c)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 141 * 71, MPI_COMM_WORLD);
            MPI_Send(&(vy_c)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 142 * 71, MPI_COMM_WORLD);
            MPI_Send(&(vz_c)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 143 * 71, MPI_COMM_WORLD);
            MPI_Send(&(vx)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 662 * 71, MPI_COMM_WORLD);
            MPI_Send(&(vy)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 663 * 71, MPI_COMM_WORLD);
            MPI_Send(&(vz)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 664 * 71, MPI_COMM_WORLD);
            MPI_Send(&(Ex)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 665 * 71, MPI_COMM_WORLD);
            MPI_Send(&(Ey)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 666 * 71, MPI_COMM_WORLD);
            MPI_Send(&(Bx)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 667 * 71, MPI_COMM_WORLD);
            MPI_Send(&(By)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 668 * 71, MPI_COMM_WORLD);
            MPI_Send(&(Bz)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 6699 * 71, MPI_COMM_WORLD);
        }

        GyroParticlePush(x.data(), y.data(), vx_c.data(), vy_c.data(), vz_c.data(),
                    vx.data(), vy.data(), vz.data(),
                    Ex.data(), Ey.data(), Bx.data(), By.data(), Bz.data(),
                    dt, charge, mass, Ntot_per_0_proc);

        vector<scalar> x_recv;
        vector<scalar> y_recv;
        vector<scalar> vx_c_recv;
        vector<scalar> vy_c_recv;
        vector<scalar> vz_c_recv;
        vector<scalar> vx_recv;
        vector<scalar> vy_recv;
        vector<scalar> vz_recv;

        x_recv.resize(Ntot_per_proc);
        y_recv.resize(Ntot_per_proc);
        vx_c_recv.resize(Ntot_per_proc);
        vy_c_recv.resize(Ntot_per_proc);
        vz_c_recv.resize(Ntot_per_proc);
        vx_recv.resize(Ntot_per_proc);
        vy_recv.resize(Ntot_per_proc);
        vz_recv.resize(Ntot_per_proc);

        for (int proc = 1; proc < commSize; ++proc){
            MPI_Recv(&x_recv[0], Ntot_per_proc, MPI_DOUBLE, proc, 669 * 71, MPI_COMM_WORLD, &status);
            MPI_Recv(&y_recv[0], Ntot_per_proc, MPI_DOUBLE, proc, 6610 * 71, MPI_COMM_WORLD, &status);
            MPI_Recv(&vx_c_recv[0], Ntot_per_proc, MPI_DOUBLE, proc, 151 * 71, MPI_COMM_WORLD, &status);
            MPI_Recv(&vy_c_recv[0], Ntot_per_proc, MPI_DOUBLE, proc, 152 * 71, MPI_COMM_WORLD, &status);
            MPI_Recv(&vz_c_recv[0], Ntot_per_proc, MPI_DOUBLE, proc, 153 * 71, MPI_COMM_WORLD, &status);
            MPI_Recv(&vx_recv[0], Ntot_per_proc, MPI_DOUBLE, proc, 6611 * 71, MPI_COMM_WORLD, &status);
            MPI_Recv(&vy_recv[0], Ntot_per_proc, MPI_DOUBLE, proc, 6612 * 71, MPI_COMM_WORLD, &status);
            MPI_Recv(&vz_recv[0], Ntot_per_proc, MPI_DOUBLE, proc, 6613 * 71, MPI_COMM_WORLD, &status);

            int start = Ntot_per_0_proc + (proc - 1) * Ntot_per_proc;
            int finish = Ntot_per_0_proc + proc * Ntot_per_proc;
            int ip_proc = 0;
            for (int ip = start; ip < finish; ++ip){
                x[ip] = x_recv[ip_proc];
                y[ip] = y_recv[ip_proc];
                vx_c[ip] = vx_c_recv[ip_proc];
                vy_c[ip] = vy_c_recv[ip_proc];
                vz_c[ip] = vz_c_recv[ip_proc];
                vx[ip] = vx_recv[ip_proc];
                vy[ip] = vy_recv[ip_proc];
                vz[ip] = vz_recv[ip_proc];
                ip_proc++;
            }
        }
    }
    else{
        int Ntot_per_proc = 0;
        MPI_Recv(&Ntot_per_proc, 1, MPI_INT, 0, 5665, MPI_COMM_WORLD, &status);

        vector<scalar> x_proc;
        vector<scalar> y_proc;
        vector<scalar> vx_c_proc;
        vector<scalar> vy_c_proc;
        vector<scalar> vz_c_proc;
        vector<scalar> vx_proc;
        vector<scalar> vy_proc;
        vector<scalar> vz_proc;
        vector<scalar> Ex_proc;
        vector<scalar> Ey_proc;
        vector<scalar> Bx_proc;
        vector<scalar> By_proc;
        vector<scalar> Bz_proc;
        
        x_proc.resize(Ntot_per_proc);
        y_proc.resize(Ntot_per_proc);
        vx_c_proc.resize(Ntot_per_proc);
        vy_c_proc.resize(Ntot_per_proc);
        vz_c_proc.resize(Ntot_per_proc);
        vx_proc.resize(Ntot_per_proc);
        vy_proc.resize(Ntot_per_proc);
        vz_proc.resize(Ntot_per_proc);
        Ex_proc.resize(Ntot_per_proc);
        Ey_proc.resize(Ntot_per_proc);
        Bx_proc.resize(Ntot_per_proc);
        By_proc.resize(Ntot_per_proc);
        Bz_proc.resize(Ntot_per_proc);

        MPI_Recv(&x_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 660 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&y_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 661 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&vx_c_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 141 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&vy_c_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 142 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&vz_c_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 143 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&vx_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 662 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&vy_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 663 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&vz_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 664 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&Ex_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 665 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&Ey_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 666 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&Bx_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 667 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&By_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 668 * 71, MPI_COMM_WORLD, &status);
        MPI_Recv(&Bz_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 6699 * 71, MPI_COMM_WORLD, &status);
        

        GyroParticlePush(x_proc.data(), y_proc.data(), vx_c_proc.data(), vy_c_proc.data(), vz_c_proc.data(),
                    vx_proc.data(), vy_proc.data(), vz_proc.data(),
                    Ex_proc.data(), Ey_proc.data(), Bx_proc.data(), By_proc.data(), Bz_proc.data(),
                    dt, charge, mass, Ntot_per_proc);  

        MPI_Send(&x_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 669 * 71, MPI_COMM_WORLD);
        MPI_Send(&y_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 6610 * 71, MPI_COMM_WORLD);
        MPI_Send(&vx_c_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 151 * 71, MPI_COMM_WORLD);
        MPI_Send(&vy_c_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 152 * 71, MPI_COMM_WORLD);
        MPI_Send(&vz_c_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 153 * 71, MPI_COMM_WORLD);
        MPI_Send(&vx_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 6611 * 71, MPI_COMM_WORLD);
        MPI_Send(&vy_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 6612 * 71, MPI_COMM_WORLD);
        MPI_Send(&vz_proc[0], Ntot_per_proc, MPI_DOUBLE, 0, 6613 * 71, MPI_COMM_WORLD);
    }
}

//Particles configuration log
void GyroKineticParticles::GetParticlesConfiguration()
{
    ofstream outCoords(ptclType + "Coords.txt");
    ofstream outVel(ptclType + "Velocities.txt");
    ofstream outVelC(ptclType + "VelocitiesC.txt");
    ofstream outE(ptclType + "E.txt");
    ofstream outB(ptclType + "B.txt");

    outCoords << "Ntot: " << Ntot << endl;
    outCoords << "PtclsPerMacro: " << ptcls_per_macro << endl;
    outCoords << "Mass: " << mass / ptcls_per_macro << endl;
    outCoords << "Charge: " << charge / ptcls_per_macro << endl;

    for (int i = 0; i < Ntot; ++i)
    {
        outCoords << x[i] << ' ' << y[i] << endl;
        outVel << vx[i] << ' ' << vy[i] << ' ' << vz[i] << endl;
        outVelC << vx_c[i] << ' ' << vy_c[i] << ' ' << vz_c[i] << endl;
        outB << Bx[i] << ' ' << By[i] << ' ' << Bz[i] << endl;
        outE << Ex[i] << ' ' << Ey[i] << endl;
    }
}
