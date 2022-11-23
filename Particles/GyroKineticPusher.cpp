//
// Created by Vladimir Smirnov on 16.11.2021.
//

#include "GyroKineticPusher.h"
//#include "../Tools/Names.h"
#include <cmath>
#include <omp.h>

scalar DotProduct(const scalar v1[], const scalar v2[]) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

void GyroUpdateSingleVelocityBoris(scalar &vel_x_c, scalar &vel_y_c, scalar &vel_z_c, scalar &vel_x, scalar &vel_y, scalar &vel_z,
                                   const scalar Ex, const scalar Ey, const scalar Bx, const scalar By, const scalar Bz,
                                   const scalar dt, const scalar q, const scalar m) {
    int vel_dim = 3;
    scalar v_B_dot, E_B_dot, phi, v_perpend_norm;
    scalar E_B_cross[vel_dim], v_E[vel_dim], v[vel_dim], v_parallel[vel_dim], E_parallel[vel_dim], v_perpend[vel_dim];
    scalar E[] = {Ex, Ey, 0};
    scalar B[] = {Bx, By, Bz};

    scalar B_norm_2 = Bx*Bx + By*By + Bz*Bz;
    CrossProduct(E, B, E_B_cross);

    v_E[0] = E_B_cross[0] / B_norm_2;
    v_E[1] = E_B_cross[1] / B_norm_2;
    v_E[2] = E_B_cross[2] / B_norm_2;

    v[0] = vel_x - v_E[0];
    v[1] = vel_y - v_E[1];
    v[2] = vel_z - v_E[2];

    v_B_dot = DotProduct(v, B);
    E_B_dot = DotProduct(E, B);

    v_parallel[0] = B[0] * v_B_dot / B_norm_2;
    v_parallel[1] = B[1] * v_B_dot / B_norm_2;
    v_parallel[2] = B[2] * v_B_dot / B_norm_2;

    v_perpend[0] = v[0] - v_parallel[0];
    v_perpend[1] = v[1] - v_parallel[1];
    v_perpend[2] = v[2] - v_parallel[2];


    E_parallel[0] = B[0] * E_B_dot / B_norm_2;
    E_parallel[1] = B[1] * E_B_dot / B_norm_2;
    E_parallel[2] = B[2] * E_B_dot / B_norm_2;

    v_parallel[0] = v_parallel[0] + (q/m) * E_parallel[0] * dt;
    v_parallel[1] = v_parallel[1] + (q/m) * E_parallel[1] * dt;
    v_parallel[2] = v_parallel[2] + (q/m) * E_parallel[2] * dt;

    // right only for B = [0, 0, Bz] !!!!!
    v_perpend_norm = sqrt(pow(v_perpend[0], 2) + pow(v_perpend[1], 2) + pow(v_perpend[2], 2));
    phi = acos(v_perpend[0] / v_perpend_norm) + std::abs(q/m) * sqrt(B_norm_2) * dt;
    v_perpend[0] = v_perpend_norm * cos(phi);
    v_perpend[1] = v_perpend_norm * sin(phi);
    v_perpend[2] = 0;

    //update gyrocenter velocity
    vel_x_c = v_parallel[0] + v_E[0];
    vel_y_c = v_parallel[1] + v_E[1];
    vel_z_c = v_parallel[2] + v_E[2];

    //update velocity
    vel_x = vel_x_c + v_perpend[0];
    vel_y = vel_y_c + v_perpend[1];
    vel_z = vel_z_c + v_perpend[2];
}

void GyroUpdateVelocity(scalar vel_x_c[], scalar vel_y_c[], scalar vel_z_c[], scalar vel_x[], scalar vel_y[], scalar vel_z[],
                        const scalar Ex[], const scalar Ey[], const scalar Bx[], const scalar By[], const scalar Bz[],
                        const scalar dt, const scalar q, const scalar m, const int Ntot) {
    #pragma omp for
    for (int ip = 0; ip < Ntot; ip++) {
        GyroUpdateSingleVelocityBoris(vel_x_c[ip], vel_y_c[ip], vel_z_c[ip], vel_x[ip], vel_y[ip], vel_z[ip], Ex[ip], Ey[ip], Bx[ip], By[ip], Bz[ip], dt, q, m);
    }
}

void GyroParticlePush(scalar pos_x[], scalar pos_y[], scalar vel_x_c[], scalar vel_y_c[], scalar vel_z_c[],
                      scalar vel_x[], scalar vel_y[], scalar vel_z[],
                      const scalar Ex[], const scalar Ey[], const scalar Bx[], const scalar By[], const scalar Bz[],
                      const scalar dt, const scalar q, const scalar m, const int Ntot) {
    auto *settings = new SettingNames();
    int numThreads = settings->GetNumberOfThreadsPerCore();
    #pragma omp parallel num_threads(numThreads)
    {
    GyroUpdateVelocity(vel_x_c, vel_y_c, vel_z_c, vel_x, vel_y, vel_z, Ex, Ey, Bx, By, Bz, dt, q, m, Ntot);
    UpdatePosition(pos_x, pos_y, vel_x_c, vel_y_c, dt, Ntot);
    }
}

void GyroPusherMpi(GyroKineticParticles &pt, scalar dt)
{
    int rank, commSize;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    scalar mass = pt.get_mass() * pt.get_ptcls_per_macro();
    scalar charge = pt.get_charge() * pt.get_ptcls_per_macro();

    if (rank == 0){
        int Ntot = pt.get_Ntot();
        int Ntot_per_proc = Ntot / commSize;
        int Ntot_per_0_proc = Ntot / commSize + Ntot % commSize;

        for (int proc = 1; proc < commSize; ++proc){
            MPI_Send(&Ntot_per_proc, 1, MPI_INT, proc, 5665, MPI_COMM_WORLD);
            MPI_Send(&(pt.x)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 660 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.y)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 661 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.vx_c)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 141 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.vy_c)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 142 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.vz_c)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 143 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.vx)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 662 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.vy)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 663 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.vz)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 664 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.Ex)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 665 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.Ey)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 666 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.Bx)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 667 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.By)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 668 * 71, MPI_COMM_WORLD);
            MPI_Send(&(pt.Bz)[Ntot_per_0_proc + (proc - 1) * Ntot_per_proc], Ntot_per_proc, MPI_DOUBLE, proc, 6699 * 71, MPI_COMM_WORLD);
        }

        GyroParticlePush(pt.x.data(), pt.y.data(), pt.vx_c.data(), pt.vy_c.data(), pt.vz_c.data(),
                    pt.vx.data(), pt.vy.data(), pt.vz.data(),
                    pt.Ex.data(), pt.Ey.data(), pt.Bx.data(), pt.By.data(), pt.Bz.data(),
                    dt, charge, mass, Ntot_per_0_proc);

        vector<scalar> x_recv;
        vector<scalar> y_recv;
        vector<scalar> vx_c_recv;
        vector<scalar> vy_c_recv;
        vector<scalar> vz_c_recv;
        vector<scalar> vx_recv;
        vector<scalar> vy_recv;
        vector<scalar> vz_recv;
        for (int proc = 1; proc < commSize; ++proc){
            x_recv.resize(Ntot_per_proc);
            y_recv.resize(Ntot_per_proc);
            vx_c_recv.resize(Ntot_per_proc);
            vy_c_recv.resize(Ntot_per_proc);
            vz_c_recv.resize(Ntot_per_proc);
            vx_recv.resize(Ntot_per_proc);
            vy_recv.resize(Ntot_per_proc);
            vz_recv.resize(Ntot_per_proc);

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
                pt.x[ip] = x_recv[ip_proc];
                pt.y[ip] = y_recv[ip_proc];
                pt.vx_c[ip] = vx_c_recv[ip_proc];
                pt.vy_c[ip] = vy_c_recv[ip_proc];
                pt.vz_c[ip] = vz_c_recv[ip_proc];
                pt.vx[ip] = vx_recv[ip_proc];
                pt.vy[ip] = vy_recv[ip_proc];
                pt.vz[ip] = vz_recv[ip_proc];
                ip_proc++;
            }

            x_recv.resize(0);
            y_recv.resize(0);
            vx_c_recv.resize(0);
            vy_c_recv.resize(0);
            vz_c_recv.resize(0);
            vx_recv.resize(0);
            vy_recv.resize(0);
            vz_recv.resize(0);
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
