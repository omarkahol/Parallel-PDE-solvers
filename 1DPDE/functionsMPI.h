#ifndef FUNCTIONSMPI_H
#define FUNCTIONSMPI_H

#include "functions.h"
#include <mpi/mpi.h>
#include <vector>

void initProcessGrid(double **pMemory,const int &ID,const int &nProcs,const int &nPoints, const double& dx, const double &length);
void mpiCommunicateResults(double **pMemory, const int &currentIT, const int &nPoints, const int &ID, const int &nProcs, MPI_Status *status);
double **finalizeResults(double **pMemory, Processor1D *pProc, const int &nTime, const int &nProcs, MPI_Status *status);
double **allocateMemoryMPI(const int &nTime, const int &nPoints);
void freeMemoryMPI(double **pMemory, const int &nTime);
void evolvePDEMPI(double **pMemory, const int &t, Processor1D *pProc);
void writeDataMPI(double **pMemory, const int &nTime, const int &nPoints, const char *fileName);
#endif