#include "Names.h"

ParticlesConstant::ParticlesConstant()
{
    electrons = "electrons";
    ions = "ions";
}

std::string ParticlesConstant::ElectronsTypeString()
{
    return electrons;
}

std::string ParticlesConstant::IonsTypeString()
{
    return ions;
}