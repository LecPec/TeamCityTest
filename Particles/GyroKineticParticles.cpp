//
// Created by Vladimir Smirnov on 27.10.2021.
//

#include "GyroKineticParticles.h"
//#include "../Tools/Names.h"
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
    vx_c_->resize(Ntot);
    vy_c_->resize(Ntot);
    vz_c_->resize(Ntot);
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

void GyroKineticParticles::GyroPusherMPI(scalar dt, int iteration)
{
    int rank, commSize;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    int Ntot_per_proc = Ntot / commSize;
    int Ntot_per_0_proc = Ntot_per_proc + Ntot % commSize;
    scalar mass = this->get_mass() * this->get_ptcls_per_macro();
    scalar charge = this->get_charge() * this->get_ptcls_per_macro();

    MPI_Bcast(&Ntot_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ntot_per_0_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int numOfPtclsToCalculate  = (rank == 0) ? Ntot_per_0_proc : Ntot_per_proc;

    int counts[commSize], displs[commSize];
    counts[0] = Ntot_per_0_proc;
    displs[0] = 0;
    for (int i = 1; i < commSize; ++i)
    {
        counts[i] = Ntot_per_proc;
        displs[i] = Ntot_per_0_proc + (i - 1) * Ntot_per_proc;
    }

    Resize(numOfPtclsToCalculate);
    vector<scalar> *v = new vector<scalar>();

    MPI_Scatterv(&x[0], counts, displs, MPI_DOUBLE, x_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&y[0], counts, displs, MPI_DOUBLE, y_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&vx[0], counts, displs, MPI_DOUBLE, vx_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&vy[0], counts, displs, MPI_DOUBLE, vy_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&vz[0], counts, displs, MPI_DOUBLE, vz_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&vx_c[0], counts, displs, MPI_DOUBLE, vx_c_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&vy_c[0], counts, displs, MPI_DOUBLE, vy_c_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&vz_c[0], counts, displs, MPI_DOUBLE, vz_c_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&Ex[0], counts, displs, MPI_DOUBLE, Ex_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&Ey[0], counts, displs, MPI_DOUBLE, Ey_->data(), numOfPtclsToCalculate, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    GyroParticlePush(x_->data(), y_->data(), vx_c_->data(), vy_c_->data(), vz_c_->data(), vx_->data(), vy_->data(), vz_->data(),
                    Ex_->data(), Ey_->data(), Bx_->data(), By_->data(), Bz_->data(), dt, charge, mass, numOfPtclsToCalculate);

    MPI_Gatherv(x_->data(), numOfPtclsToCalculate, MPI_DOUBLE, &x[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_->data(), numOfPtclsToCalculate, MPI_DOUBLE, &y[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vx_->data(), numOfPtclsToCalculate, MPI_DOUBLE, &vx[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vy_->data(), numOfPtclsToCalculate, MPI_DOUBLE, &vy[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vz_->data(), numOfPtclsToCalculate, MPI_DOUBLE, &vz[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vx_c_->data(), numOfPtclsToCalculate, MPI_DOUBLE, &vx_c[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vy_c_->data(), numOfPtclsToCalculate, MPI_DOUBLE, &vy_c[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vz_c_->data(), numOfPtclsToCalculate, MPI_DOUBLE, &vz_c[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (iteration % 100 == 0)
    {
        ShrinkToFit();
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

void GyroKineticParticles::Resize(int newSize)
{
    Particles::Resize(newSize);
    vx_c_->resize(newSize);
    vy_c_->resize(newSize);
    vz_c_->resize(newSize);
}

void GyroKineticParticles::ShrinkToFit()
{
    Particles::ShrinkToFit();
    vx_c_->shrink_to_fit();
    vy_c_->shrink_to_fit();
    vz_c_->shrink_to_fit();
}