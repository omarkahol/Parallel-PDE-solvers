#ifndef FUNCTIONSMPI_H
#define FUNCTIONSMPI_H

#include "functions.h"
#include <mpi/mpi.h>
#include <vector>

void initProcessGrid(double **pMemory,const int &ID,const int &nPROCS,const int &nPOINTS, const double& DX, const double &LENGTH);
void mpiCommunicateResults(double **pMemory, const int &currentIT, const int &nPOINTS, const int &ID, const int &nPROCS, MPI_Status *status);

#endif