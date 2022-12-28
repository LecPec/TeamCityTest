#include "Names.h"

ParticlesConstant::ParticlesConstant()
{
    electrons = "electrons";
    ions = "ions";
    velFileSuffix = "Velocities.txt";
    velCFileSuffix = "VelocitiesCenter.txt";
    coordFileSuffix = "Positions.txt";
    EFileSuffix = "ElectricField.txt";
    BFileSuffix = "MagneticField.txt";
    baseFileName = "BaseData.txt";
    initConfigurationFolderIn = "InitialConfiguration/"; //for ifstream
    initConfigurationFolderOut = "Tmp/"; //for ofstream
}

std::string ParticlesConstant::ElectronsTypeString()
{
    return electrons;
}

std::string ParticlesConstant::IonsTypeString()
{
    return ions;
}

std::string ParticlesConstant::VelocityFileSuffix()
{
    return velFileSuffix;
}

std::string ParticlesConstant::CoordinatesFileSuffix()
{
    return coordFileSuffix;
}

std::string ParticlesConstant::VelocityCenterFileSuffix()
{
    return velCFileSuffix;
}

std::string ParticlesConstant::ElectricFieldFileSuffix()
{
    return EFileSuffix;
}

std::string ParticlesConstant::MagneticFieldFileSuffix()
{
    return BFileSuffix;
}

std::string ParticlesConstant::BaseFileName()
{
    return baseFileName;
}

std::string ParticlesConstant::InitialConfigurationFolderIn()
{
    return initConfigurationFolderIn;
}

std::string ParticlesConstant::InitialConfigurationFolderOut()
{
    return initConfigurationFolderOut;
}

SettingNames::SettingNames()
{
    numberOfThreadsPerCore = 10;
}

int SettingNames::GetNumberOfThreadsPerCore()
{
    return numberOfThreadsPerCore;
}