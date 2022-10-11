#include <iostream>

class ParticlesConstant
{
private:
    std::string electrons;
    std::string ions;
public:
    ParticlesConstant();
    std::string ElectronsTypeString();
    std::string IonsTypeString();
};