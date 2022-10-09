//
// Created by Vladimir Smirnov on 09.10.2021.
//

#include "NullCollisions.h"
#include <random>
#include <mpi.h>
#include <stdlib.h>

#define E_M 9.10938356e-31
#define EV 1.6021766208e-19
#define K_B 1.380649e-23

void electron_null_collisionsNew(ElectronNeutralElasticCollision &electron_elastic, Ionization &ionization) {
    int Ntot;
    scalar prob_1, prob_2;
    int rank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    int seed = rank + 1;
    srand(rank + 1);
    scalar random_number;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    if (rank == 0)
    {
        

        int nIonsStart = ionization.ionized_particles->get_Ntot();
        int nElectronsStart = electron_elastic.particles->get_Ntot();
        Ntot = electron_elastic.particles->get_Ntot();
        int NtotPerZeroProc = Ntot / commSize + Ntot % commSize;
        int NtotPerProc = Ntot / commSize;

        array<vector<scalar>, 2> electronsXY = electron_elastic.particles->GetPositions();
        vector<scalar> electronsX = electronsXY[0];
        vector<scalar> electronsY = electronsXY[1];

        array<vector<scalar>, 3> electronsVxVyVz = electron_elastic.particles->GetVelocities();
        vector<scalar> electronsVx = electronsVxVyVz[0];
        vector<scalar> electronsVy = electronsVxVyVz[1];
        vector<scalar> electronsVz = electronsVxVyVz[2];

        for (int proc = 1; proc < commSize; ++proc)
        {
            MPI_Send(&NtotPerProc, 1, MPI_INT, proc, 81180, MPI_COMM_WORLD);

            MPI_Send(&electronsX[NtotPerZeroProc + (proc - 1) * NtotPerProc], NtotPerProc, MPI_DOUBLE, proc, 81181, MPI_COMM_WORLD);
            MPI_Send(&electronsY[NtotPerZeroProc + (proc - 1) * NtotPerProc], NtotPerProc, MPI_DOUBLE, proc, 81182, MPI_COMM_WORLD);
            
            MPI_Send(&electronsVx[NtotPerZeroProc + (proc - 1) * NtotPerProc], NtotPerProc, MPI_DOUBLE, proc, 81183, MPI_COMM_WORLD);
            MPI_Send(&electronsVy[NtotPerZeroProc + (proc - 1) * NtotPerProc], NtotPerProc, MPI_DOUBLE, proc, 81184, MPI_COMM_WORLD);
            MPI_Send(&electronsVz[NtotPerZeroProc + (proc - 1) * NtotPerProc], NtotPerProc, MPI_DOUBLE, proc, 81185, MPI_COMM_WORLD);
        }
        
        array<scalar, 3> electronVel;
        array<scalar, 2> electronPos;
        array<scalar, 3> newElectronVel, newIonVel;
        array<scalar, 2> newElectronPos, newIonPos;
        int sumElastic = 0, sumIonized = 0;
        for (int pt = 0; pt < NtotPerZeroProc; ++pt)
        {
            electronVel = {electronsVx[pt], electronsVy[pt], electronsVz[pt]};
            electronPos = {electronsX[pt], electronsY[pt]};
            prob_1 = electron_elastic.probabilityNew(electronVel);
            prob_2 = ionization.probabilityNew(electronVel);
            random_number = distribution(generator);
            if (random_number < prob_1)
            {
                sumElastic++;
                electron_elastic.collisionNew(electronVel);
                electron_elastic.particles->set_velocity(pt, electronVel);
            }
            else if (random_number >= prob_1 and random_number < prob_1 + prob_2) 
            {
                sumIonized++;
                ionization.collisionNew(electronPos, electronVel, newElectronPos, newElectronVel, newIonPos, newIonVel);
                electron_elastic.particles->set_velocity(pt, electronVel);
                electron_elastic.particles->append(electronPos, newElectronVel);
                ionization.ionized_particles->append(electronPos, newIonVel);
            }
        }

        int elasticCount = 0, ionizationCount = 0;
        MPI_Status status;

        vector<int> idxsOfElectrons, idxsOfElectronsIonized;
        vector<scalar> oldElectronsVxRecv, oldElectronsVyRecv, oldElectronsVzRecv;
        vector<scalar> newElectronsXRecv, newElectronsYRecv, newElectronsVxRecv, newElectronsVyRecv, newElectronsVzRecv;
        vector<scalar> newIonsXRecv, newIonsYRecv, newIonsVxRecv, newIonsVyRecv, newIonsVzRecv;
        vector<scalar> electronsVxIonized, electronsVyIonized, electronsVzIonized;
        for (int proc = 1; proc < commSize; ++proc)
        {
            MPI_Recv(&elasticCount, 1, MPI_INT, proc, 91190, MPI_COMM_WORLD, &status);
            MPI_Recv(&ionizationCount, 1, MPI_INT, proc, 91191, MPI_COMM_WORLD, &status);
            sumElastic += elasticCount;
            sumIonized += ionizationCount;

            idxsOfElectrons.resize(elasticCount);
            idxsOfElectronsIonized.resize(ionizationCount);

            oldElectronsVxRecv.resize(elasticCount);
            oldElectronsVyRecv.resize(elasticCount);
            oldElectronsVzRecv.resize(elasticCount);

            newElectronsVxRecv.resize(ionizationCount);
            newElectronsVyRecv.resize(ionizationCount);
            newElectronsVzRecv.resize(ionizationCount);

            newIonsVxRecv.resize(ionizationCount);
            newIonsVyRecv.resize(ionizationCount);
            newIonsVzRecv.resize(ionizationCount);

            electronsVxIonized.resize(ionizationCount);
            electronsVyIonized.resize(ionizationCount);
            electronsVzIonized.resize(ionizationCount);

            MPI_Recv(&idxsOfElectrons[0], elasticCount, MPI_INT, proc, 91192, MPI_COMM_WORLD, &status);
            MPI_Recv(&oldElectronsVxRecv[0], elasticCount, MPI_DOUBLE, proc, 91193, MPI_COMM_WORLD, &status);
            MPI_Recv(&oldElectronsVyRecv[0], elasticCount, MPI_DOUBLE, proc, 91194, MPI_COMM_WORLD, &status);
            MPI_Recv(&oldElectronsVzRecv[0], elasticCount, MPI_DOUBLE, proc, 91195, MPI_COMM_WORLD, &status);

            MPI_Recv(&newIonsVxRecv[0], ionizationCount, MPI_DOUBLE, proc, 91198, MPI_COMM_WORLD, &status);
            MPI_Recv(&newIonsVyRecv[0], ionizationCount, MPI_DOUBLE, proc, 91199, MPI_COMM_WORLD, &status);
            MPI_Recv(&newIonsVzRecv[0], ionizationCount, MPI_DOUBLE, proc, 911910, MPI_COMM_WORLD, &status);

            MPI_Recv(&newElectronsVxRecv[0], ionizationCount, MPI_DOUBLE, proc, 911913, MPI_COMM_WORLD, &status);
            MPI_Recv(&newElectronsVyRecv[0], ionizationCount, MPI_DOUBLE, proc, 911914, MPI_COMM_WORLD, &status);
            MPI_Recv(&newElectronsVzRecv[0], ionizationCount, MPI_DOUBLE, proc, 911915, MPI_COMM_WORLD, &status);

            MPI_Recv(&electronsVxIonized[0], ionizationCount, MPI_DOUBLE, proc, 911916, MPI_COMM_WORLD, &status);
            MPI_Recv(&electronsVyIonized[0], ionizationCount, MPI_DOUBLE, proc, 911917, MPI_COMM_WORLD, &status);
            MPI_Recv(&electronsVzIonized[0], ionizationCount, MPI_DOUBLE, proc, 911918, MPI_COMM_WORLD, &status);
            MPI_Recv(&idxsOfElectronsIonized[0], ionizationCount, MPI_INT, proc, 911919, MPI_COMM_WORLD, &status);

            int idx = 0;
            array<scalar, 2> pos;
            
            for (int i = 0; i < elasticCount; ++i)
            {
                idx = NtotPerZeroProc + (proc - 1) * NtotPerProc + idxsOfElectrons[i];
                electron_elastic.particles->set_velocity(idx, {oldElectronsVxRecv[i], oldElectronsVyRecv[i], oldElectronsVzRecv[i]});
            }

            for (int i = 0; i < ionizationCount; ++i)
            {

                idx = NtotPerZeroProc + (proc - 1) * NtotPerProc + idxsOfElectronsIonized[i];
                pos = electron_elastic.particles->get_position(idx);
                electron_elastic.particles->set_velocity(idx, {electronsVxIonized[i], electronsVyIonized[i], electronsVzIonized[i]});
                electron_elastic.particles->append(pos, {newElectronsVxRecv[i], newElectronsVyRecv[i], newElectronsVzRecv[i]});
                ionization.ionized_particles->append(pos, {newIonsVxRecv[i], newIonsVyRecv[i], newIonsVzRecv[i]});
            }
        }
    }

    else
    {
        int NtotPerProc = 0;
        vector<scalar> electronsX, electronsY, electronsVx, electronsVy, electronsVz;
        MPI_Status status;

        MPI_Recv(&NtotPerProc, 1, MPI_INT, 0, 81180, MPI_COMM_WORLD, &status);

        electronsX.resize(NtotPerProc);
        electronsY.resize(NtotPerProc);
        electronsVx.resize(NtotPerProc);
        electronsVy.resize(NtotPerProc);
        electronsVz.resize(NtotPerProc);

        MPI_Recv(&electronsX[0], NtotPerProc, MPI_DOUBLE, 0, 81181, MPI_COMM_WORLD, &status);
        MPI_Recv(&electronsY[0], NtotPerProc, MPI_DOUBLE, 0, 81182, MPI_COMM_WORLD, &status);

        MPI_Recv(&electronsVx[0], NtotPerProc, MPI_DOUBLE, 0, 81183, MPI_COMM_WORLD, &status);
        MPI_Recv(&electronsVy[0], NtotPerProc, MPI_DOUBLE, 0, 81184, MPI_COMM_WORLD, &status);
        MPI_Recv(&electronsVz[0], NtotPerProc, MPI_DOUBLE, 0, 81185, MPI_COMM_WORLD, &status);

        array<scalar, 3> electronVel;
        array<scalar, 2> electronPos;
        array<scalar, 3> newElectronVel, newIonVel;
        array<scalar, 2> newElectronPos, newIonPos;

        vector<scalar> newIonsVx, newIonsVy, newIonsVz;
        vector<scalar> newElectronsVx, newElectronsVy, newElectronsVz;
        vector<scalar> oldElectronsNewVx, oldElectronsNewVy, oldElectronsNewVz;
        vector<scalar> electronsVxIonized, electronsVyIonized, electronsVzIonized;
        

        vector<int> idxsOfElectrons;
        vector<int> idxsOfElectronsIonized;
        for (int pt = 0; pt < NtotPerProc; ++pt)
        {
            electronVel = {electronsVx[pt], electronsVy[pt], electronsVz[pt]};
            electronPos = {electronsX[pt], electronsY[pt]};
            prob_1 = electron_elastic.probabilityNew(electronVel);
            prob_2 = ionization.probabilityNew(electronVel);
            random_number = distribution(generator);
            if (random_number < prob_1) 
            {
                electron_elastic.collisionNew(electronVel);

                idxsOfElectrons.push_back(pt);

                oldElectronsNewVx.push_back(electronVel[0]);
                oldElectronsNewVy.push_back(electronVel[1]);
                oldElectronsNewVz.push_back(electronVel[2]);
            }
            else if (random_number >= prob_1 and random_number < prob_1 + prob_2) 
            {
                idxsOfElectronsIonized.push_back(pt);
                ionization.collisionNew(electronPos, electronVel, newElectronPos, newElectronVel, newIonPos, newIonVel);

                newIonsVx.push_back(newIonVel[0]);
                newIonsVy.push_back(newIonVel[1]);
                newIonsVz.push_back(newIonVel[2]);

                newElectronsVx.push_back(newElectronVel[0]);
                newElectronsVy.push_back(newElectronVel[1]);
                newElectronsVz.push_back(newElectronVel[2]);

                electronsVxIonized.push_back(electronVel[0]);
                electronsVyIonized.push_back(electronVel[1]);
                electronsVzIonized.push_back(electronVel[2]);
            }
        }

        int elasticCount = idxsOfElectrons.size();
        int ionizationCount = newIonsVx.size();

        MPI_Send(&elasticCount, 1, MPI_INT, 0, 91190, MPI_COMM_WORLD);
        MPI_Send(&ionizationCount, 1, MPI_INT, 0, 91191, MPI_COMM_WORLD);

        MPI_Send(&idxsOfElectrons[0], elasticCount, MPI_INT, 0, 91192, MPI_COMM_WORLD);
        MPI_Send(&oldElectronsNewVx[0], elasticCount, MPI_DOUBLE, 0, 91193, MPI_COMM_WORLD);
        MPI_Send(&oldElectronsNewVy[0], elasticCount, MPI_DOUBLE, 0, 91194, MPI_COMM_WORLD);
        MPI_Send(&oldElectronsNewVz[0], elasticCount, MPI_DOUBLE, 0, 91195, MPI_COMM_WORLD);

        MPI_Send(&newIonsVx[0], ionizationCount, MPI_DOUBLE, 0, 91198, MPI_COMM_WORLD);
        MPI_Send(&newIonsVy[0], ionizationCount, MPI_DOUBLE, 0, 91199, MPI_COMM_WORLD);
        MPI_Send(&newIonsVz[0], ionizationCount, MPI_DOUBLE, 0, 911910, MPI_COMM_WORLD);

        MPI_Send(&newElectronsVx[0], ionizationCount, MPI_DOUBLE, 0, 911913, MPI_COMM_WORLD);
        MPI_Send(&newElectronsVy[0], ionizationCount, MPI_DOUBLE, 0, 911914, MPI_COMM_WORLD);
        MPI_Send(&newElectronsVz[0], ionizationCount, MPI_DOUBLE, 0, 911915, MPI_COMM_WORLD);

        MPI_Send(&electronsVxIonized[0], ionizationCount, MPI_DOUBLE, 0, 911916, MPI_COMM_WORLD);
        MPI_Send(&electronsVyIonized[0], ionizationCount, MPI_DOUBLE, 0, 911917, MPI_COMM_WORLD);
        MPI_Send(&electronsVzIonized[0], ionizationCount, MPI_DOUBLE, 0, 911918, MPI_COMM_WORLD);
        MPI_Send(&idxsOfElectronsIonized[0], ionizationCount, MPI_INT, 0, 911919, MPI_COMM_WORLD);
    }
}

void electron_null_collisions(ElectronNeutralElasticCollision &electron_elastic, Ionization &ionization) {
    int Ntot;
    scalar prob_1, prob_2, random_number;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::random_device rd;
    int seed = rd();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    Ntot = electron_elastic.particles->get_Ntot();
    srand(rank + 1);
    int nIonized = 0, nElastic = 0;

    for (int ptcl_idx = 0; ptcl_idx < Ntot; ptcl_idx++) {
        prob_1 = electron_elastic.probability(ptcl_idx);
        prob_2 = ionization.probability(ptcl_idx);
        random_number = distribution(generator); //distribution(generator);
        //cout << rank << ' ' << random_number << " -> " << prob_1 << ' ' << prob_2 << endl;
        if (random_number < prob_1) {
            electron_elastic.collision(ptcl_idx);
        }
        else if (random_number >= prob_1 and random_number < prob_1 + prob_2) {
            ionization.collision(ptcl_idx);
        }

    }
}

void electron_null_collisionsMpi(ElectronNeutralElasticCollision &electron_elastic, Ionization &ionization) {
    int Ntot;
    scalar prob_1, prob_2, random_number;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int seed = rank + 1;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    Ntot = electron_elastic.particles->get_Ntot();
    for (int ptcl_idx = 0; ptcl_idx < Ntot; ptcl_idx++) {
        prob_1 = electron_elastic.probability(ptcl_idx);
        prob_2 = ionization.probability(ptcl_idx);
        random_number = distribution(generator);
        if (random_number < prob_1) {
            //cout << "elastic " << random_number << endl;
            electron_elastic.collision(ptcl_idx);
        }
        else if (random_number >= prob_1 and random_number < prob_1 + prob_2) {
            //cout << "ionization " << random_number << endl;
            ionization.collision(ptcl_idx);
        }

    }
}

void electron_null_collisionsMPI(EnergyCrossSection& elastic_electron_sigma, EnergyCrossSection& ionization_sigma,
                                 const int coll_step, const scalar dt, NeutralGas& gas, Particles& electrons, Particles& ions, const scalar m_ion)
{
    int rank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    int electronsPerMacro = electrons.get_ptcls_per_macro();
    int ionsPerMacro = ions.get_ptcls_per_macro();
    
    if (rank == 0)
    {
        int NtotPerZeroProcE = electrons.get_Ntot() / commSize + electrons.get_Ntot() % commSize;
        int NtotPerZeroProcI = ions.get_Ntot() / commSize + ions.get_Ntot() % commSize;
        int NtotPerProcE = electrons.get_Ntot() / commSize;
        int NtotPerProcI = ions.get_Ntot() / commSize;
        
        Particles electronsZeroProc(electrons.get_mass(), electrons.get_charge(), NtotPerZeroProcE, electrons.get_ptcls_per_macro());
        Particles ionsZeroProc(ions.get_mass(), ions.get_charge(), NtotPerZeroProcI, ions.get_ptcls_per_macro());

        int startE = 0, startI = 0;
        int startTag = 0;

        for (int proc = 1; proc < commSize; ++proc)
        {
            MPI_Send(&NtotPerProcE, 1, MPI_INT, proc, 6446, MPI_COMM_WORLD);
            MPI_Send(&NtotPerProcI, 1, MPI_INT, proc, 64461, MPI_COMM_WORLD);

            startTag = 0;

            startE = NtotPerZeroProcE + (proc - 1) * NtotPerProcE;
            startI = NtotPerZeroProcI + (proc - 1) * NtotPerProcI;

            MPI_Send(&(electrons.vx)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag, MPI_COMM_WORLD);
            MPI_Send(&(electrons.vy)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 1, MPI_COMM_WORLD);
            MPI_Send(&(electrons.vz)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 2, MPI_COMM_WORLD);
            MPI_Send(&(electrons.x)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 3, MPI_COMM_WORLD);
            MPI_Send(&(electrons.y)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 4, MPI_COMM_WORLD);
            MPI_Send(&(electrons.Bx)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 5, MPI_COMM_WORLD);
            MPI_Send(&(electrons.By)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 6, MPI_COMM_WORLD);
            MPI_Send(&(electrons.Bz)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 7, MPI_COMM_WORLD);
            MPI_Send(&(electrons.Ex)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 8, MPI_COMM_WORLD);
            MPI_Send(&(electrons.Ey)[startE], NtotPerProcE, MPI_DOUBLE, proc, startTag + 9, MPI_COMM_WORLD);

            startTag = 10;

            MPI_Send(&(ions.vx)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag, MPI_COMM_WORLD);
            MPI_Send(&(ions.vy)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 1, MPI_COMM_WORLD);
            MPI_Send(&(ions.vz)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 2, MPI_COMM_WORLD);
            MPI_Send(&(ions.x)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 3, MPI_COMM_WORLD);
            MPI_Send(&(ions.y)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 4, MPI_COMM_WORLD);
            MPI_Send(&(ions.Bx)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 5, MPI_COMM_WORLD);
            MPI_Send(&(ions.By)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 6, MPI_COMM_WORLD);
            MPI_Send(&(ions.Bz)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 7, MPI_COMM_WORLD);
            MPI_Send(&(ions.Ex)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 8, MPI_COMM_WORLD);
            MPI_Send(&(ions.Ey)[startI], NtotPerProcI, MPI_DOUBLE, proc, startTag + 9, MPI_COMM_WORLD);
        }

        electronsZeroProc.AssignData(
            electrons.vx,
            electrons.vy,
            electrons.vz,
            electrons.x,
            electrons.y,
            electrons.Bx,
            electrons.By,
            electrons.Bz,
            electrons.Ex,
            electrons.Ey
        );

        ionsZeroProc.AssignData(
            ions.vx,
            ions.vy,
            ions.vz,
            ions.x,
            ions.y,
            ions.Bx,
            ions.By,
            ions.Bz,
            ions.Ex,
            ions.Ey
        );

        ElectronNeutralElasticCollision electron_elastic(elastic_electron_sigma, coll_step * dt, gas, electronsZeroProc);
        Ionization ionization(ionization_sigma, coll_step * dt, gas, electronsZeroProc, ionsZeroProc);

        MPI_Status status;

        electron_null_collisions(electron_elastic, ionization);

        int newNtotE = electronsZeroProc.get_Ntot(), newNtotI = ionsZeroProc.get_Ntot();
        MPI_Status st;
        for (int proc = 1; proc < commSize; ++proc)
        {
            int numOfElectrons = 0;
            int numOfIons = 0;

            MPI_Recv(&numOfElectrons, 1, MPI_INT, proc, 111, MPI_COMM_WORLD, &st);
            MPI_Recv(&numOfIons, 1, MPI_INT, proc, 222, MPI_COMM_WORLD, &st);

            newNtotE += numOfElectrons;
            newNtotI += numOfIons;
        }

        electrons.ClearData();
        electrons.Resize(newNtotE);
        for (int i = 0; i < electronsZeroProc.get_Ntot(); ++i)
        {
            electrons.x[i] = electronsZeroProc.x[i];
            electrons.y[i] = electronsZeroProc.y[i];
            electrons.vx[i] = electronsZeroProc.vx[i];
            electrons.vy[i] = electronsZeroProc.vy[i];
            electrons.vz[i] = electronsZeroProc.vz[i];
            electrons.Bx[i] = electronsZeroProc.Bx[i];
            electrons.By[i] = electronsZeroProc.By[i];
            electrons.Bz[i] = electronsZeroProc.Bz[i];
            electrons.Ex[i] = electronsZeroProc.Ex[i];
            electrons.Ey[i] = electronsZeroProc.Ey[i];
        }

        ions.Resize(newNtotI);
        for (int i = 0; i < ionsZeroProc.get_Ntot(); ++i)
        {
            ions.x[i] = ionsZeroProc.x[i];
            ions.y[i] = ionsZeroProc.y[i];
            ions.vx[i] = ionsZeroProc.vx[i];
            ions.vy[i] = ionsZeroProc.vy[i];
            ions.vz[i] = ionsZeroProc.vz[i];
            ions.Bx[i] = ionsZeroProc.Bx[i];
            ions.By[i] = ionsZeroProc.By[i];
            ions.Bz[i] = ionsZeroProc.Bz[i];
            ions.Ex[i] = ionsZeroProc.Ex[i];
            ions.Ey[i] = ionsZeroProc.Ey[i];
        }

        vector<scalar> vxRecv;
        vector<scalar> vyRecv;
        vector<scalar> vzRecv;
        vector<scalar> xRecv;
        vector<scalar> yRecv;
        vector<scalar> BxRecv;
        vector<scalar> ByRecv;
        vector<scalar> BzRecv;
        vector<scalar> ExRecv;
        vector<scalar> EyRecv;

        int newStartE = electronsZeroProc.get_Ntot();
        int newStartI = ionsZeroProc.get_Ntot();
        int newFinishE = 0, newFinishI = 0;

        for (int proc = 1; proc < commSize; ++proc)
        {
            int numOfElectrons = 0;
            int numOfIons = 0;

            MPI_Recv(&numOfElectrons, 1, MPI_INT, proc, 1114, MPI_COMM_WORLD, &st);
            MPI_Recv(&numOfIons, 1, MPI_INT, proc, 2224, MPI_COMM_WORLD, &st);

            newFinishE = newStartE + numOfElectrons;
            newFinishI = newStartI + numOfIons;

            vxRecv.clear();
            vyRecv.clear();
            vzRecv.clear();
            xRecv.clear();
            yRecv.clear();
            BxRecv.clear();
            ByRecv.clear();
            BzRecv.clear();
            ExRecv.clear();
            EyRecv.clear();

            vxRecv.resize(numOfElectrons);
            vyRecv.resize(numOfElectrons);
            vzRecv.resize(numOfElectrons);
            xRecv.resize(numOfElectrons);
            yRecv.resize(numOfElectrons);
            BxRecv.resize(numOfElectrons);
            ByRecv.resize(numOfElectrons);
            BzRecv.resize(numOfElectrons);
            ExRecv.resize(numOfElectrons);
            EyRecv.resize(numOfElectrons);

            startTag = 20;

            MPI_Recv(&vxRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&vyRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&vzRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&xRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 3, MPI_COMM_WORLD, &status);
            MPI_Recv(&yRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 4, MPI_COMM_WORLD, &status);
            MPI_Recv(&BxRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 5, MPI_COMM_WORLD, &status);
            MPI_Recv(&ByRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 6, MPI_COMM_WORLD, &status);
            MPI_Recv(&BzRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 7, MPI_COMM_WORLD, &status);
            MPI_Recv(&ExRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 8, MPI_COMM_WORLD, &status);
            MPI_Recv(&EyRecv[0], numOfElectrons, MPI_DOUBLE, proc, startTag + 9, MPI_COMM_WORLD, &status);

            int ip = 0;
            for (int i = newStartE; i < newFinishE; ++i)
            {
                electrons.x[i] = xRecv[ip];
                electrons.y[i] = yRecv[ip];
                electrons.vx[i] = vxRecv[ip];
                electrons.vy[i] = vyRecv[ip];
                electrons.vz[i] = vzRecv[ip];
                electrons.Bx[i] = BxRecv[ip];
                electrons.By[i] = ByRecv[ip];
                electrons.Bz[i] = BzRecv[ip];
                electrons.Ex[i] = ExRecv[ip];
                electrons.Ey[i] = EyRecv[ip];
                ip++;
            }

            vxRecv.resize(0);
            vyRecv.resize(0);
            vzRecv.resize(0);
            xRecv.resize(0);
            yRecv.resize(0);
            BxRecv.resize(0);
            ByRecv.resize(0);
            BzRecv.resize(0);
            ExRecv.resize(0);
            EyRecv.resize(0);

            vxRecv.resize(numOfIons);
            vyRecv.resize(numOfIons);
            vzRecv.resize(numOfIons);
            xRecv.resize(numOfIons);
            yRecv.resize(numOfIons);
            BxRecv.resize(numOfIons);
            ByRecv.resize(numOfIons);
            BzRecv.resize(numOfIons);
            ExRecv.resize(numOfIons);
            EyRecv.resize(numOfIons);

            startTag = 30;

            MPI_Recv(&vxRecv[0], numOfIons, MPI_DOUBLE, proc, startTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&vyRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&vzRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&xRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 3, MPI_COMM_WORLD, &status);
            MPI_Recv(&yRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 4, MPI_COMM_WORLD, &status);
            MPI_Recv(&BxRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 5, MPI_COMM_WORLD, &status);
            MPI_Recv(&ByRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 6, MPI_COMM_WORLD, &status);
            MPI_Recv(&BzRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 7, MPI_COMM_WORLD, &status);
            MPI_Recv(&ExRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 8, MPI_COMM_WORLD, &status);
            MPI_Recv(&EyRecv[0], numOfIons, MPI_DOUBLE, proc, startTag + 9, MPI_COMM_WORLD, &status);

            ip = 0;
            for (int i = newStartI; i < newFinishI; ++i)
            {
                ions.x[i] = xRecv[ip];
                ions.y[i] = yRecv[ip];
                ions.vx[i] = vxRecv[ip];
                ions.vy[i] = vyRecv[ip];
                ions.vz[i] = vzRecv[ip];
                ions.Bx[i] = BxRecv[ip];
                ions.By[i] = ByRecv[ip];
                ions.Bz[i] = BzRecv[ip];
                ions.Ex[i] = ExRecv[ip];
                ions.Ey[i] = EyRecv[ip];
                ip++;
            }
            vxRecv.resize(0);
            vyRecv.resize(0);
            vzRecv.resize(0);
            xRecv.resize(0);
            yRecv.resize(0);
            BxRecv.resize(0);
            ByRecv.resize(0);
            BzRecv.resize(0);
            ExRecv.resize(0);
            EyRecv.resize(0);

            newStartE += numOfElectrons;
            newStartI += numOfIons;
        }
    }
    else
    {
        int startTag = 0;
        MPI_Status status;

        int NtotPerProcE = 0, NtotPerProcI = 0;
        MPI_Recv(&NtotPerProcE, 1, MPI_INT, 0, 6446, MPI_COMM_WORLD, &status);
        MPI_Recv(&NtotPerProcI, 1, MPI_INT, 0, 64461, MPI_COMM_WORLD, &status);

        vector<scalar> vxProc;
        vector<scalar> vyProc;
        vector<scalar> vzProc;
        vector<scalar> xProc;
        vector<scalar> yProc;
        vector<scalar> BxProc;
        vector<scalar> ByProc;
        vector<scalar> BzProc;
        vector<scalar> ExProc;
        vector<scalar> EyProc;

        vxProc.resize(NtotPerProcE);
        vyProc.resize(NtotPerProcE);
        vzProc.resize(NtotPerProcE);
        xProc.resize(NtotPerProcE);
        yProc.resize(NtotPerProcE);
        BxProc.resize(NtotPerProcE);
        ByProc.resize(NtotPerProcE);
        BzProc.resize(NtotPerProcE);
        ExProc.resize(NtotPerProcE);
        EyProc.resize(NtotPerProcE);

        MPI_Recv(&vxProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag, MPI_COMM_WORLD, &status);
        MPI_Recv(&vyProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&vzProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&xProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 3, MPI_COMM_WORLD, &status);
        MPI_Recv(&yProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 4, MPI_COMM_WORLD, &status);
        MPI_Recv(&BxProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 5, MPI_COMM_WORLD, &status);
        MPI_Recv(&ByProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 6, MPI_COMM_WORLD, &status);
        MPI_Recv(&BzProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 7, MPI_COMM_WORLD, &status);
        MPI_Recv(&ExProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 8, MPI_COMM_WORLD, &status);
        MPI_Recv(&EyProc[0], NtotPerProcE, MPI_DOUBLE, 0, startTag + 9, MPI_COMM_WORLD, &status);

        Particles electronsProc(electrons.get_mass(), electrons.get_charge(), NtotPerProcE, electrons.get_ptcls_per_macro());

        electronsProc.AssignData(
            vxProc,
            vyProc,
            vzProc,
            xProc,
            yProc,
            BxProc,
            ByProc,
            BzProc,
            ExProc,
            EyProc
        );

        vxProc.resize(0);
        vyProc.resize(0);
        vzProc.resize(0);
        xProc.resize(0);
        yProc.resize(0);
        BxProc.resize(0);
        ByProc.resize(0);
        BzProc.resize(0);
        ExProc.resize(0);
        EyProc.resize(0);

        vxProc.resize(NtotPerProcI);
        vyProc.resize(NtotPerProcI);
        vzProc.resize(NtotPerProcI);
        xProc.resize(NtotPerProcI);
        yProc.resize(NtotPerProcI);
        BxProc.resize(NtotPerProcI);
        ByProc.resize(NtotPerProcI);
        BzProc.resize(NtotPerProcI);
        ExProc.resize(NtotPerProcI);
        EyProc.resize(NtotPerProcI);

        startTag = 10;

        MPI_Recv(&vxProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag, MPI_COMM_WORLD, &status);
        MPI_Recv(&vyProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&vzProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&xProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 3, MPI_COMM_WORLD, &status);
        MPI_Recv(&yProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 4, MPI_COMM_WORLD, &status);
        MPI_Recv(&BxProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 5, MPI_COMM_WORLD, &status);
        MPI_Recv(&ByProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 6, MPI_COMM_WORLD, &status);
        MPI_Recv(&BzProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 7, MPI_COMM_WORLD, &status);
        MPI_Recv(&ExProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 8, MPI_COMM_WORLD, &status);
        MPI_Recv(&EyProc[0], NtotPerProcI, MPI_DOUBLE, 0, startTag + 9, MPI_COMM_WORLD, &status);

        Particles ionsProc(ions.get_mass(), ions.get_charge(), NtotPerProcI, ions.get_ptcls_per_macro());
        
        ionsProc.AssignData(
            vxProc,
            vyProc,
            vzProc,
            xProc,
            yProc,
            BxProc,
            ByProc,
            BzProc,
            ExProc,
            EyProc
        );

        vxProc.resize(0);
        vyProc.resize(0);
        vzProc.resize(0);
        xProc.resize(0);
        yProc.resize(0);
        BxProc.resize(0);
        ByProc.resize(0);
        BzProc.resize(0);
        ExProc.resize(0);
        EyProc.resize(0);

        ElectronNeutralElasticCollision electron_elastic(elastic_electron_sigma, coll_step * dt, gas, electronsProc);
        Ionization ionization(ionization_sigma, coll_step * dt, gas, electronsProc, ionsProc);
        int NtotPermanent = 0;

        electron_null_collisions(electron_elastic, ionization);

        int numOfElectrons = electronsProc.get_Ntot();
        int numOfIons = ionsProc.get_Ntot();
        MPI_Send(&numOfElectrons, 1, MPI_INT, 0, 111, MPI_COMM_WORLD);
        MPI_Send(&numOfIons, 1, MPI_INT, 0, 222, MPI_COMM_WORLD);

        MPI_Send(&numOfElectrons, 1, MPI_INT, 0, 1114, MPI_COMM_WORLD);
        MPI_Send(&numOfIons, 1, MPI_INT, 0, 2224, MPI_COMM_WORLD);
        
        startTag = 20;

        MPI_Send(&(electronsProc.vx)[0], numOfElectrons, MPI_DOUBLE, 0, startTag, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.vy)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 1, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.vz)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 2, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.x)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 3, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.y)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 4, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.Bx)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 5, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.By)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 6, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.Bz)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 7, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.Ex)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 8, MPI_COMM_WORLD);
        MPI_Send(&(electronsProc.Ey)[0], numOfElectrons, MPI_DOUBLE, 0, startTag + 9, MPI_COMM_WORLD);

        startTag = 30;

        MPI_Send(&(ionsProc.vx)[0], numOfIons, MPI_DOUBLE, 0, startTag, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.vy)[0], numOfIons, MPI_DOUBLE, 0, startTag + 1, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.vz)[0], numOfIons, MPI_DOUBLE, 0, startTag + 2, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.x)[0], numOfIons, MPI_DOUBLE, 0, startTag + 3, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.y)[0], numOfIons, MPI_DOUBLE, 0, startTag + 4, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.Bx)[0], numOfIons, MPI_DOUBLE, 0, startTag + 5, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.By)[0], numOfIons, MPI_DOUBLE, 0, startTag + 6, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.Bz)[0], numOfIons, MPI_DOUBLE, 0, startTag + 7, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.Ex)[0], numOfIons, MPI_DOUBLE, 0, startTag + 8, MPI_COMM_WORLD);
        MPI_Send(&(ionsProc.Ey)[0], numOfIons, MPI_DOUBLE, 0, startTag + 9, MPI_COMM_WORLD);

    }

}

void ion_null_collisions(IonNeutralElasticCollision &ion_elastic) {
    int Ntot;
    scalar prob_1, random_number;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + 1);
    std:random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    Ntot = ion_elastic.particles->get_Ntot();
    for (int ptcl_idx = 0; ptcl_idx < Ntot; ptcl_idx++) {
        prob_1 = ion_elastic.probability(ptcl_idx);
        random_number = distribution(generator);
        if (random_number < prob_1) {
            ion_elastic.collision(ptcl_idx);
        }
    }
}
