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
