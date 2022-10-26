//
// Created by Vladimir Smirnov on 11.09.2021.
//

#include "Particles.h"
#include "Pusher.h"
#include <algorithm>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <mpi.h>

void Particles::append(const array<scalar, 2> &position, const array<scalar, 3> &velocity) {
    x.push_back(position[0]);
    y.push_back(position[1]);
    vx.push_back(velocity[0]);
    vy.push_back(velocity[1]);
    vz.push_back(velocity[2]);
    Bx.push_back(Bx_const);
    By.push_back(By_const);
    Bz.push_back(Bz_const);
    Ex.push_back(0);
    Ey.push_back(0);
    Ntot++;
}

void Particles::pop(const int ptcl_idx) {
    swap(x[ptcl_idx], x[Ntot-1]);
    swap(y[ptcl_idx], y[Ntot-1]);
    swap(vx[ptcl_idx], vx[Ntot-1]);
    swap(vy[ptcl_idx], vy[Ntot-1]);
    swap(vz[ptcl_idx], vz[Ntot-1]);
    swap(Bx[ptcl_idx], Bx[Ntot-1]);
    swap(By[ptcl_idx], By[Ntot-1]);
    swap(Bz[ptcl_idx], Bz[Ntot-1]);
    swap(Ex[ptcl_idx], Ex[Ntot-1]);
    swap(Ey[ptcl_idx], Ey[Ntot-1]);
    x.pop_back();
    y.pop_back();
    vx.pop_back();
    vy.pop_back();
    vz.pop_back();
    Bx.pop_back();
    By.pop_back();
    Bz.pop_back();
    Ex.pop_back();
    Ey.pop_back();
    Ntot--;
}

void Particles::vel_pusher(const scalar dt) {
    UpdateVelocity(vx.data(), vy.data(), vz.data(), Ex.data(), Ey.data(), Bx.data(), By.data(), Bz.data(), dt, charge, mass, Ntot);
}

void Particles::pusher(const scalar dt) {
    ParticlePush(x.data(), y.data(), vx.data(), vy.data(), vz.data(), Ex.data(), Ey.data(), Bx.data(), By.data(), Bz.data(), dt, charge, mass, Ntot);
}

void Particles::pusherMPI(const scalar dt)
{
    int rank, commSize;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    scalar mass = this->get_mass() * this->get_ptcls_per_macro();
    scalar charge = this->get_charge() * this->get_ptcls_per_macro();
    int numOfPhysValues = 10;

    if (rank == 0){
        int Ntot = get_Ntot();
        int Ntot_per_proc = Ntot / commSize;
        int Ntot_per_0_proc = Ntot / commSize + Ntot % commSize;

        int fragmentSize = x.size();
        int totalSize = fragmentSize * numOfPhysValues;

        MPI_Datatype ParticlesDataType;
        int blockLength[numOfPhysValues];
        int displacements[numOfPhysValues];
        for (int i = 0; i < numOfPhysValues; ++i)
        {
            blockLength[i] = Ntot_per_proc;
            displacements[i] = i * Ntot_per_proc;
        }
        MPI_Type_indexed(numOfPhysValues, blockLength, displacements, MPI_DOUBLE, &ParticlesDataType);
        MPI_Type_commit(&ParticlesDataType);

        vector<scalar> particlesData;
        int start = 0;

        for (int proc = 1; proc < commSize; ++proc){
            MPI_Send(&Ntot_per_proc, 1, MPI_INT, proc, 5665, MPI_COMM_WORLD);

            start = Ntot_per_0_proc + (proc - 1) * Ntot_per_proc;
            particlesData.insert(particlesData.end(), x.begin() + start, x.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), y.begin() + start, y.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), vx.begin() + start, vx.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), vy.begin() + start, vy.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), vz.begin() + start, vz.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), Bx.begin() + start, Bx.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), By.begin() + start, By.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), Bz.begin() + start, Bz.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), Ex.begin() + start, Ex.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), Ey.begin() + start, Ey.begin() + start + Ntot_per_proc);
            MPI_Send(&particlesData[0], 1, ParticlesDataType, proc, 8778, MPI_COMM_WORLD);

            particlesData.resize(0);
        }

        ParticlePush(x.data(), y.data(),
                    vx.data(), vy.data(), vz.data(),
                    Ex.data(), Ey.data(), Bx.data(), By.data(), Bz.data(),
                    dt, charge, mass, Ntot_per_0_proc);

        vector<scalar> recvParticlesData;
        int numOfNotNeededPhysValues = 5;
        int sizeOfRecievedData = (numOfPhysValues - numOfNotNeededPhysValues) * Ntot_per_proc;
        recvParticlesData.resize(sizeOfRecievedData);

        for (int proc = 1; proc < commSize; ++proc){
            MPI_Recv(&recvParticlesData[0], recvParticlesData.size(), MPI_DOUBLE, proc, 8998, MPI_COMM_WORLD, &status);

            int start = Ntot_per_0_proc + (proc - 1) * Ntot_per_proc;
            int finish = Ntot_per_0_proc + proc * Ntot_per_proc;
            #pragma omp parallel for num_threads(NUM_THREADS)
            for (int ip = start; ip < finish; ++ip){
                x[ip] = recvParticlesData[0 * Ntot_per_proc + ip - start];
                y[ip] = recvParticlesData[1 * Ntot_per_proc + ip - start];
                vx[ip] = recvParticlesData[2 * Ntot_per_proc + ip - start];
                vy[ip] = recvParticlesData[3 * Ntot_per_proc + ip - start];
                vz[ip] = recvParticlesData[4 * Ntot_per_proc + ip - start];
            }
        }
    }
    else{
        int Ntot_per_proc = 0;
        MPI_Recv(&Ntot_per_proc, 1, MPI_INT, 0, 5665, MPI_COMM_WORLD, &status);

        int sizeOfProcData = numOfPhysValues * Ntot_per_proc;

        vector<scalar> particlesData;
        particlesData.resize(sizeOfProcData);
        MPI_Recv(&particlesData[0], sizeOfProcData, MPI_DOUBLE, 0, 8778, MPI_COMM_WORLD, &status);

        vector<scalar> x_proc;
        vector<scalar> y_proc;
        vector<scalar> vx_proc;
        vector<scalar> vy_proc;
        vector<scalar> vz_proc;
        vector<scalar> Ex_proc;
        vector<scalar> Ey_proc;
        vector<scalar> Bx_proc;
        vector<scalar> By_proc;
        vector<scalar> Bz_proc;
        
        x_proc.insert(x_proc.end(), particlesData.begin(), particlesData.begin() + Ntot_per_proc);
        y_proc.insert(y_proc.end(), particlesData.begin() + Ntot_per_proc, particlesData.begin() + 2 * Ntot_per_proc);
        vx_proc.insert(vx_proc.end(), particlesData.begin() + 2 * Ntot_per_proc, particlesData.begin() + 3 * Ntot_per_proc);
        vy_proc.insert(vy_proc.end(), particlesData.begin() + 3 * Ntot_per_proc, particlesData.begin() + 4 * Ntot_per_proc);
        vz_proc.insert(vz_proc.end(), particlesData.begin() + 4 * Ntot_per_proc, particlesData.begin() + 5 * Ntot_per_proc);
        Bx_proc.insert(Bx_proc.end(), particlesData.begin() + 5 * Ntot_per_proc, particlesData.begin() + 6 * Ntot_per_proc);
        By_proc.insert(By_proc.end(), particlesData.begin() + 6 * Ntot_per_proc, particlesData.begin() + 7 * Ntot_per_proc);
        Bz_proc.insert(Bz_proc.end(), particlesData.begin() + 7 * Ntot_per_proc, particlesData.begin() + 8 * Ntot_per_proc);
        Ex_proc.insert(Ex_proc.end(), particlesData.begin() + 8 * Ntot_per_proc, particlesData.begin() + 9 * Ntot_per_proc);
        Ey_proc.insert(Ey_proc.end(), particlesData.begin() + 9 * Ntot_per_proc, particlesData.begin() + 10 * Ntot_per_proc);
        
        ParticlePush(x_proc.data(), y_proc.data(),
                    vx_proc.data(), vy_proc.data(), vz_proc.data(),
                    Ex_proc.data(), Ey_proc.data(), Bx_proc.data(), By_proc.data(), Bz_proc.data(),
                    dt, charge, mass, Ntot_per_proc);

        particlesData.resize(0);
        particlesData.insert(particlesData.end(), x_proc.begin(), x_proc.end());
        particlesData.insert(particlesData.end(), y_proc.begin(), y_proc.end());
        particlesData.insert(particlesData.end(), vx_proc.begin(), vx_proc.end());
        particlesData.insert(particlesData.end(), vy_proc.begin(), vy_proc.end());
        particlesData.insert(particlesData.end(), vz_proc.begin(), vz_proc.end()); 

        MPI_Send(&particlesData[0], particlesData.size(), MPI_DOUBLE, 0, 8998, MPI_COMM_WORLD);
    }
}

Particles::Particles(scalar m, scalar q, int N, scalar N_per_macro) {
    Ntot = N;
    if (N_per_macro > 1) {
        ptcls_per_macro = N_per_macro;
    } else {
        ptcls_per_macro = 1;
    }
    mass = m * ptcls_per_macro;
    charge = q * ptcls_per_macro;
    Bx_const = 0;
    By_const = 0;
    Bz_const = 0;
    x.resize(Ntot, 0);
    y.resize(Ntot, 0);
    vx.resize(Ntot, 0);
    vy.resize(Ntot, 0);
    vz.resize(Ntot, 0);
    Ex.resize(Ntot, 0);
    Ey.resize(Ntot, 0);
    Bx.resize(Ntot, 0);
    By.resize(Ntot, 0);
    Bz.resize(Ntot, 0);
}

Particles::Particles(scalar m, scalar q, int N, string type, scalar N_per_macro) {
    Ntot = N;
    if (N_per_macro > 1) {
        ptcls_per_macro = N_per_macro;
    } else {
        ptcls_per_macro = 1;
    }
    ptclType = type;
    mass = m * ptcls_per_macro;
    charge = q * ptcls_per_macro;
    Bx_const = 0;
    By_const = 0;
    Bz_const = 0;
    x.resize(Ntot, 0);
    y.resize(Ntot, 0);
    vx.resize(Ntot, 0);
    vy.resize(Ntot, 0);
    vz.resize(Ntot, 0);
    Ex.resize(Ntot, 0);
    Ey.resize(Ntot, 0);
    Bx.resize(Ntot, 0);
    By.resize(Ntot, 0);
    Bz.resize(Ntot, 0);
}

scalar Particles::get_ptcls_per_macro() const {
    return ptcls_per_macro;
}

scalar Particles::get_charge() const {
    return charge/ptcls_per_macro;;
}

scalar Particles::get_mass() const {
    return mass/ptcls_per_macro;
}

int Particles::get_Ntot() const {
    return Ntot;
}

array<scalar, 3> Particles::get_velocity(const int ptcl_idx) const {
    array<scalar, 3> vel = {vx[ptcl_idx], vy[ptcl_idx], vz[ptcl_idx]};
    return vel;
}

array<scalar, 2> Particles::get_position(const int ptcl_idx) const {
    array<scalar, 2> pos = {x[ptcl_idx], y[ptcl_idx]};
    return pos;
}

void Particles::set_velocity(const int ptcl_idx, const array<scalar, 3> &velocity) {
    vx[ptcl_idx] = velocity[0];
    vy[ptcl_idx] = velocity[1];
    vz[ptcl_idx] = velocity[2];
}

void Particles::set_position(const int ptcl_idx, const array<scalar, 2> &position) {
    x[ptcl_idx] = position[0];
    y[ptcl_idx] = position[1];
}

void Particles::set_const_magnetic_field(const array<scalar, 3> &B) {
    Bx.assign(Ntot, B[0]);
    By.assign(Ntot, B[1]);
    Bz.assign(Ntot, B[2]);
    Bx_const = B[0];
    By_const = B[1];
    Bz_const = B[2];
}

void Particles::SetNtot(int NtotNew)
{
    Ntot = NtotNew;
}

void Particles::AddNtot(int dN)
{
    Ntot += dN;
}

void Particles::pop_list(const vector<int> &ptcl_idx_list) {
    assert(x.size() == Ntot);
    assert(ptcl_idx_list.size() <= Ntot);

    int leave_ptcl_idx;
    int main_ptcl_idx = Ntot - 1;
    for(int i = 0; i < ptcl_idx_list.size(); i++) {
        leave_ptcl_idx = ptcl_idx_list[i];
        swap(x[leave_ptcl_idx], x[main_ptcl_idx]);
        swap(y[leave_ptcl_idx], y[main_ptcl_idx]);
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

//<<<Checking if the ptcl is on the anode surface>>>//
//<<<{circle, left, right, up, down}>>>//
bool Particles::ptcl_is_on_anode(int row, int col, const Grid& grid){
    bool is_in_circle;
    bool is_in_first_col;
    bool is_in_last_col;
    bool is_in_first_row;
    bool is_in_last_row;
    int R = (grid.Nx - 1) / 2;

    is_in_circle = pow(row - R, 2) + pow(col - R, 2) < pow(R, 2);
    is_in_first_col = col == 0;
    is_in_last_col = col == grid.Nx - 1;
    is_in_first_row = row == 0;
    is_in_last_row = row == grid.Ny - 1;


    return is_in_circle and (is_in_first_col or is_in_last_col or is_in_first_row or is_in_last_row);
}



void Particles::update_ptcls_on_anode(const Grid& grid){
    int row = 0;
    int col = 0;

    bool is_in_circle;
    bool is_in_first_col;
    bool is_in_last_col;
    bool is_in_first_row;
    bool is_in_last_row;
    bool ptcl_on_anode;

    for (int ptcl = 0; ptcl < Ntot; ++ptcl){
        row = floor(y[ptcl] / grid.dy);
        col = floor(x[ptcl] / grid.dx);

        ptcl_on_anode = ptcl_is_on_anode(row, col, grid);
        //cout << ptcl_on_anode << endl;

        if (ptcl_on_anode){
            ptcls_on_anode[row][col]++;
        }
    }
}

void Particles::update_ptcls_on_anode_leave(int row, int col){
    ptcls_on_anode[row][col]++;
}

void Particles::print_ptcls_on_anode(const string& ptcl_type, int it, const Grid& grid){
    ofstream fout("current_hist/N_on_anode_" + ptcl_type + '_' + to_string(it) + ".txt"); 

    for (int i = 0; i < grid.Nx; ++i){
        for (int j = 0; j < grid.Ny; ++j){
            fout << ptcls_on_anode[i][j] << ' ';
        }
        fout << endl;
    }  
}

void Particles::update_anode_current(const Grid& grid){
    vector<vector<double> > v_average;
    vector<vector<int> > N;
    vector<int> tmp_int;
    vector<double> tmp_double;

    for (int row = 0; row < grid.Ny; ++row){
        for (int col = 0; col < grid.Nx; ++col){
            tmp_int.push_back(0);
            tmp_double.push_back(0);
        }
        v_average.push_back(tmp_double);
        N.push_back(tmp_int);
        tmp_int.clear();
        tmp_double.clear();
    }

    int row = 0, col = 0;
    scalar vr = 0;
    scalar L = 1.;
    scalar V = grid.dx * grid.dy * L;
    scalar S = grid.dx * L;

    for (int ptcl = 0; ptcl < Ntot; ++ptcl){
        row = floor(y[ptcl] / grid.dy);
        col = floor(x[ptcl] / grid.dx);

        if (ptcl_is_on_anode(row, col, grid)){
            vr = sqrt(vx[ptcl] * vx[ptcl] + vy[ptcl] * vy[ptcl]);
            v_average[row][col] += vr;
            N[row][col]++;
        }
    }

    for (int row = 0; row < grid.Ny; ++row){
        for (int col = 0; col < grid.Nx; ++col){
            if (N[row][col] != 0){
                v_average[row][col] /= N[row][col];
            }
            J[row][col] += N[row][col] * N[row][col] / V * S * ELECTRON_CHARGE * v_average[row][col];
        }
    }
}

void Particles::print_current_on_anode(const string& ptcl_type, int it, const Grid& grid){
    ofstream fout("current_hist/anode_current_" + ptcl_type + '_' + to_string(it) + ".txt"); 

    for (int i = 0; i < grid.Nx; ++i){
        for (int j = 0; j < grid.Ny; ++j){
            fout << J[i][j] << ' ';
        }
        fout << endl;
    }  
}

void Particles::ZeroAnodeCurrent(){
    for (int i = 0; i < J.size(); ++i){
        J[i].fill(0);
        numPtclsOnAnode[i].fill(0);
        Vr[i].fill(0);
        VrTimesRho[i].fill(0);
    }
}

void Particles::UpdateAnodeCurrent(int anodeCurrentStep){
    for (int row = 0; row < NX; row++){
        for (int col = 0; col < NX; col++){
            if (numPtclsOnAnode[row][col] != 0)
                J[row][col] = VrTimesRho[row][col] / (numPtclsOnAnode[row][col] * anodeCurrentStep);
            else 
                J[row][col] = 0;
        }
    }
}

void Particles::UpdateNumOfPtclsOnAnode(const Grid& grid, const Matrix& rho, int ptclIdx){
    int row = floor(get_position(ptclIdx)[0] / grid.dx);
    int col = floor(get_position(ptclIdx)[1] / grid.dy);
    numPtclsOnAnode[row][col] += 1;
    array<scalar, 3> v = get_velocity(ptclIdx);
    scalar vr2 = v[0] * v[0] + v[1] * v[1]; 
    VrTimesRho[row][col] += rho(row, col) * sqrt(vr2);
}

void Particles::PrintAnodeCurrentDencity(int iter){
    ofstream fout("anode_current_hist/anode_current_" + to_string(iter) + ".txt");
    for (int row = 0; row < NX; ++row){
        for (int col = 0; col < NX; ++col){
            fout << J[row][col] << endl;
        }
        fout << endl;
    }
}

void Particles::PrintAnodeCurrentParticles(const array<array<scalar, NX>, NX>& electronsInAnodeCells, const array<array<scalar, NX>, NX>& ionsInAnodeCells, int iter, int timeStep, scalar dt){
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

array<array<scalar, NX>, NX> Particles::GetJ(){
    return J;
}

void Particles::Resize(int _size)
    {
        x.resize(_size);
        y.resize(_size);
        vx.resize(_size);
        vy.resize(_size);
        vz.resize(_size);
        Bx.resize(_size);
        By.resize(_size);
        Bz.resize(_size);
        Ex.resize(_size);
        Ey.resize(_size);
        Ntot = _size;
    }

array<vector<scalar>, 2> Particles::GetPositions()
{
    return {x, y};
}

array<vector<scalar>, 3> Particles::GetVelocities()
{
    return {vx, vy, vz};
}

//Particles configuration log
void Particles::GetParticlesConfiguration()
{
    ParticlesConstant *ptclConstants = new ParticlesConstant();
    ofstream outBaseData(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->BaseFileName());
    ofstream outCoords(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->CoordinatesFileSuffix());
    ofstream outVel(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->VelocityFileSuffix());
    ofstream outE(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->ElectricFieldFileSuffix());
    ofstream outB(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->MagneticFieldFileSuffix());

    outBaseData << Ntot << endl;
    outBaseData << ptcls_per_macro << endl;
    outBaseData << mass / ptcls_per_macro << endl;
    outBaseData << charge / ptcls_per_macro << endl;

    for (int i = 0; i < Ntot; ++i)
    {
        outCoords << x[i] << ' ' << y[i] << endl;
        outVel << vx[i] << ' ' << vy[i] << ' ' << vz[i] << endl;
        outB << Bx[i] << ' ' << By[i] << ' ' << Bz[i] << endl;
        outE << Ex[i] << ' ' << Ey[i] << endl;
    }
}

void Particles::InitConfigurationFromFile()
{
    ParticlesConstant *ptclConstants = new ParticlesConstant();
    ifstream fBaseData(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->BaseFileName());
    ifstream fCoords(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->CoordinatesFileSuffix());
    ifstream fVel(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->VelocityFileSuffix());
    ifstream fE(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->ElectricFieldFileSuffix());
    ifstream fB(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->MagneticFieldFileSuffix());

    //<<Base data init>>//
    vector<scalar> baseData;

    while(!fBaseData.eof())
    {
        scalar tmpScalar;
        fBaseData >> tmpScalar;
        baseData.push_back(tmpScalar);
    }

    int N = baseData[0];
    scalar N_per_macro = baseData[1];
    scalar m = baseData[2];
    scalar q = baseData[3];
    Ntot = N;
    if (N_per_macro > 1) {
        ptcls_per_macro = N_per_macro;
    } else {
        ptcls_per_macro = 1;
    }
    mass = m * ptcls_per_macro;
    charge = q * ptcls_per_macro;
    Bx_const = 0;
    By_const = 0;
    Bz_const = 0;
    x.resize(Ntot, 0);
    y.resize(Ntot, 0);
    vx.resize(Ntot, 0);
    vy.resize(Ntot, 0);
    vz.resize(Ntot, 0);
    Ex.resize(Ntot, 0);
    Ey.resize(Ntot, 0);
    Bx.resize(Ntot, 0);
    By.resize(Ntot, 0);
    Bz.resize(Ntot, 0);

    //<<Particles data init>>
    int numOfParticle = 0;

    while(!fCoords.eof())
    {
        fCoords >> x[numOfParticle] >> y[numOfParticle];
        fVel >> vx[numOfParticle] >> vy[numOfParticle] >> vz[numOfParticle];
        fE >> Ex[numOfParticle] >> Ey[numOfParticle];
        fB >> Bx[numOfParticle] >> By[numOfParticle] >> Bz[numOfParticle];

        numOfParticle++;
    }
}



