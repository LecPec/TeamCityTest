#include <iostream>
#include <string>

using namespace std;

class Bug : exception
{
public:
    string error;
    const char *error_ = &error[0];

    Bug(string errorText);
    const char * what () const throw () { return error_; }
};