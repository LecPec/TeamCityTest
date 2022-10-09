//
// Created by Vladimir Smirnov on 09.10.2021.
//

#ifndef CPP_2D_PIC_NULLCOLLISIONS_H
#define CPP_2D_PIC_NULLCOLLISIONS_H


#include "Collision.h"

void electron_null_collisionsNew(ElectronNeutralElasticCollision& electron_elastic, Ionization& ionization);
void electron_null_collisions(ElectronNeutralElasticCollision& electron_elastic, Ionization& ionization);
void electron_null_collisionsMpi(ElectronNeutralElasticCollision &electron_elastic, Ionization &ionization);
void electron_null_collisionsMPI(EnergyCrossSection& elastic_electron_sigma, EnergyCrossSection& ionization_sigma,
                                 const int coll_step, const scalar dt, NeutralGas& gas, Particles& electrons, Particles& ions, const scalar m_ion);
void ion_null_collisions(IonNeutralElasticCollision& ion_elastic);


#endif //CPP_2D_PIC_NULLCOLLISIONS_H
