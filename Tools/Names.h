#include <iostream>

class ParticlesConstant
{
private:
    std::string electrons;
    std::string ions;
    std::string velFileSuffix;
    std::string coordFileSuffix;
    std::string BFileSuffix;
    std::string EFileSuffix;
    std::string velCFileSuffix;
    std::string baseFileName;
    std::string initConfigurationFolderIn;
    std::string initConfigurationFolderOut;

public:
    ParticlesConstant();
    std::string ElectronsTypeString();
    std::string IonsTypeString();
    std::string VelocityFileSuffix();
    std::string CoordinatesFileSuffix();
    std::string ElectricFieldFileSuffix();
    std::string MagneticFieldFileSuffix();
    std::string VelocityCenterFileSuffix();
    std::string BaseFileName();
    std::string InitialConfigurationFolderIn();
    std::string InitialConfigurationFolderOut();
};