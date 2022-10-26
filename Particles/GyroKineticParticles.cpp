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
    int numOfPhysValues = 13;

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
            particlesData.insert(particlesData.end(), vx_c.begin() + start, vx_c.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), vy_c.begin() + start, vy_c.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), vz_c.begin() + start, vz_c.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), Bx.begin() + start, Bx.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), By.begin() + start, By.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), Bz.begin() + start, Bz.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), Ex.begin() + start, Ex.begin() + start + Ntot_per_proc);
            particlesData.insert(particlesData.end(), Ey.begin() + start, Ey.begin() + start + Ntot_per_proc);
            MPI_Send(&particlesData[0], 1, ParticlesDataType, proc, 8778, MPI_COMM_WORLD);

            particlesData.resize(0);
        }

        GyroParticlePush(x.data(), y.data(), vx_c.data(), vy_c.data(), vz_c.data(),
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
                vx_c[ip] = recvParticlesData[5 * Ntot_per_proc + ip - start];
                vy_c[ip] = recvParticlesData[6 * Ntot_per_proc + ip - start];
                vz_c[ip] = recvParticlesData[7 * Ntot_per_proc + ip - start];
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
        
        x_proc.insert(x_proc.end(), particlesData.begin(), particlesData.begin() + Ntot_per_proc);
        y_proc.insert(y_proc.end(), particlesData.begin() + Ntot_per_proc, particlesData.begin() + 2 * Ntot_per_proc);
        vx_proc.insert(vx_proc.end(), particlesData.begin() + 2 * Ntot_per_proc, particlesData.begin() + 3 * Ntot_per_proc);
        vy_proc.insert(vy_proc.end(), particlesData.begin() + 3 * Ntot_per_proc, particlesData.begin() + 4 * Ntot_per_proc);
        vz_proc.insert(vz_proc.end(), particlesData.begin() + 4 * Ntot_per_proc, particlesData.begin() + 5 * Ntot_per_proc);
        vx_c_proc.insert(vx_c_proc.end(), particlesData.begin() + 5 * Ntot_per_proc, particlesData.begin() + 6 * Ntot_per_proc);
        vy_c_proc.insert(vy_c_proc.end(), particlesData.begin() + 6 * Ntot_per_proc, particlesData.begin() + 7 * Ntot_per_proc);
        vz_c_proc.insert(vz_c_proc.end(), particlesData.begin() + 7 * Ntot_per_proc, particlesData.begin() + 8 * Ntot_per_proc);
        Bx_proc.insert(Bx_proc.end(), particlesData.begin() + 8 * Ntot_per_proc, particlesData.begin() + 9 * Ntot_per_proc);
        By_proc.insert(By_proc.end(), particlesData.begin() + 9 * Ntot_per_proc, particlesData.begin() + 10 * Ntot_per_proc);
        Bz_proc.insert(Bz_proc.end(), particlesData.begin() + 10 * Ntot_per_proc, particlesData.begin() + 11 * Ntot_per_proc);
        Ex_proc.insert(Ex_proc.end(), particlesData.begin() + 11 * Ntot_per_proc, particlesData.begin() + 12 * Ntot_per_proc);
        Ey_proc.insert(Ey_proc.end(), particlesData.begin() + 12 * Ntot_per_proc, particlesData.begin() + 13 * Ntot_per_proc);
        
        GyroParticlePush(x_proc.data(), y_proc.data(), vx_c_proc.data(), vy_c_proc.data(), vz_c_proc.data(),
                    vx_proc.data(), vy_proc.data(), vz_proc.data(),
                    Ex_proc.data(), Ey_proc.data(), Bx_proc.data(), By_proc.data(), Bz_proc.data(),
                    dt, charge, mass, Ntot_per_proc);

        particlesData.resize(0);
        particlesData.insert(particlesData.end(), x_proc.begin(), x_proc.end());
        particlesData.insert(particlesData.end(), y_proc.begin(), y_proc.end());
        particlesData.insert(particlesData.end(), vx_proc.begin(), vx_proc.end());
        particlesData.insert(particlesData.end(), vy_proc.begin(), vy_proc.end());
        particlesData.insert(particlesData.end(), vz_proc.begin(), vz_proc.end());
        particlesData.insert(particlesData.end(), vx_c_proc.begin(), vx_c_proc.end());
        particlesData.insert(particlesData.end(), vy_c_proc.begin(), vy_c_proc.end());
        particlesData.insert(particlesData.end(), vz_c_proc.begin(), vz_c_proc.end());  

        MPI_Send(&particlesData[0], particlesData.size(), MPI_DOUBLE, 0, 8998, MPI_COMM_WORLD);
    }
}

//Particles configuration log
void GyroKineticParticles::GetParticlesConfiguration()
{
    ParticlesConstant *ptclConstants = new ParticlesConstant();
    ofstream outBase(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->BaseFileName());
    ofstream outCoords(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->CoordinatesFileSuffix());
    ofstream outVelC(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->VelocityCenterFileSuffix());
    ofstream outVel(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->VelocityFileSuffix());
    ofstream outE(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->ElectricFieldFileSuffix());
    ofstream outB(ptclConstants->InitialConfigurationFolderOut() + ptclType + ptclConstants->MagneticFieldFileSuffix());


    outBase << Ntot << endl;
    outBase << ptcls_per_macro << endl;
    outBase << mass / ptcls_per_macro << endl;
    outBase << charge / ptcls_per_macro << endl;

    for (int i = 0; i < Ntot; ++i)
    {
        outCoords << x[i] << ' ' << y[i] << endl;
        outVel << vx[i] << ' ' << vy[i] << ' ' << vz[i] << endl;
        outVelC << vx_c[i] << ' ' << vy_c[i] << ' ' << vz_c[i] << endl;
        outB << Bx[i] << ' ' << By[i] << ' ' << Bz[i] << endl;
        outE << Ex[i] << ' ' << Ey[i] << endl;
    }
}

void GyroKineticParticles::InitConfigurationFromFile()
{
    ParticlesConstant *ptclConstants = new ParticlesConstant();
    ifstream fBaseData(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->BaseFileName());
    ifstream fCoords(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->CoordinatesFileSuffix());
    ifstream fVel(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->VelocityFileSuffix());
    ifstream fVelC(ptclConstants->InitialConfigurationFolderIn() + ptclType + ptclConstants->VelocityCenterFileSuffix());
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
    vx_c.resize(Ntot, 0);
    vy_c.resize(Ntot, 0);
    vz_c.resize(Ntot, 0);

    //<<Particles data init>>
    int numOfParticle = 0;
    while(!fCoords.eof())
    {
        fCoords >> x[numOfParticle] >> y[numOfParticle];
        fVel >> vx[numOfParticle] >> vy[numOfParticle] >> vz[numOfParticle];
        fVelC >> vx_c[numOfParticle] >> vy_c[numOfParticle] >> vz_c[numOfParticle];
        fE >> Ex[numOfParticle] >> Ey[numOfParticle];
        fB >> Bx[numOfParticle] >> By[numOfParticle] >> Bz[numOfParticle];

        numOfParticle++;
    }
}
