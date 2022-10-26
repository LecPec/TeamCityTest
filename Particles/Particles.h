//
// Created by Vladimir Smirnov on 11.09.2021.
//

#ifndef CPP_2D_PIC_PARTICLES_H
#define CPP_2D_PIC_PARTICLES_H
#define NX 69
#define NUM_THREADS 4
#define ELECTRON_CHARGE 1.6e-19

#include "../Tools/ProjectTypes.h"
#include "../Tools/Matrix.h"
#include "../Tools/Grid.h"
#include "../Tools/Names.h"
#include <vector>
#include <array>
#include <algorithm>
#include <fstream>
using namespace std;

class Particles {
protected:
    int Ntot;
    string ptclType;
    scalar ptcls_per_macro, mass, charge, Bx_const, By_const, Bz_const;
public:
    vector<scalar> vx;
    vector<scalar> vy;
    vector<scalar> vz;
    vector<scalar> x;
    vector<scalar> y;
    vector<scalar> Bx;
    vector<scalar> By;
    vector<scalar> Bz;
    vector<scalar> Ex;
    vector<scalar> Ey;
    Particles(scalar m, scalar q, int N, scalar N_per_macro = 1);
    Particles(scalar m, scalar q, int N, string type, scalar N_per_macro = 1);

    virtual void append(const array<scalar, 2>& position, const array<scalar, 3>& velocity);

    virtual void pop(int ptcl_idx);

    virtual void pop_list(const vector<int>& ptcl_idx_list);

    virtual void vel_pusher(scalar  dt);
    virtual void pusher(scalar  dt);
    virtual void pusherMPI(scalar dt);

    void Resize(int _size);

    //setters & getters
    void set_const_magnetic_field(const array<scalar, 3>& B);
    void set_position(const int ptcl_idx, const array<scalar, 2>& position);
    virtual void set_velocity(const int ptcl_idx, const array<scalar, 3>& velocity);
    array<scalar, 2> get_position(const int ptcl_idx) const;

    virtual array<scalar, 3> get_velocity(const int ptcl_idx) const;

    int get_Ntot() const;
    scalar get_mass() const;
    scalar get_charge() const;
    scalar get_ptcls_per_macro() const;
    array<vector<scalar>, 2> GetPositions();
    array<vector<scalar>, 3> GetVelocities();

    //<<<Checking if the ptcl is on anode surface>>>//
    //<<<Init of info about ptcls on anode>>>//
    vector<vector<int> > ptcls_on_anode;
    array<array<scalar, NX>, NX> J;
    array<array<scalar, NX>, NX> numPtclsOnAnode;
    array<array<scalar, NX>, NX> Vr;
    array<array<scalar, NX>, NX> VrTimesRho;
    void UpdateAnodeCurrent(int anodeCurrentStep);
    void UpdateNumOfPtclsOnAnode(const Grid& grid, const Matrix& rho, int ptclIdx);
    void ZeroAnodeCurrent();
    void PrintAnodeCurrentDencity(int iter);
    void PrintAnodeCurrentParticles(const array<array<scalar, NX>, NX>& electronsInAnodeCells, const array<array<scalar, NX>, NX>& ionsInAnodeCells, int iter, int timeStep, scalar dt);
    array<array<scalar, NX>, NX> GetJ();

    void zero_ptcls_anode_info(const Grid& grid);
    //<<<{circle, left, right, up, down}>>>//
    bool ptcl_is_on_anode(int row, int col, const Grid& grid); //checks if the ptcl has to be included in a list of ptcls on anode
    void update_ptcls_on_anode(const Grid& grid); //updates the number of ptcls on the anode surface
    void update_ptcls_on_anode_leave(int row, int col);
    void print_ptcls_on_anode(const string& ptcl_type, int it, const Grid& grid);
    //<<<Anode current>>>//
    void update_anode_current(const Grid& grid);
    void print_current_on_anode(const string& ptcl_type, int it, const Grid& grid);

    void SetNtot(int NtotNew);
    void AddNtot(int dN);

    //Particles configuration log
    virtual void GetParticlesConfiguration();
    virtual void InitConfigurationFromFile();
};


#endif //CPP_2D_PIC_PARTICLES_H
