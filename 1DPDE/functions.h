#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <fstream>
#include <cmath>
#include <iostream>

const double c = 10;
const double k = 0.1;

struct Processor1D{
int start;
int nPointsProc;
int nPointsTot;
double dx;
double dt;
};

void writeData(double **pMemory, const int &nTime, const int &nPoints, const char *fileName);
double **allocateMemory(const int &nTime, const int &nPoints);
void freeMemory(double **pMemory, const int &nTime);
void initGrid(double **pMemory, const int &nPoints, const double& dx, const double &length);
void evolvePDE(double **pMemory, const int &t, Processor1D *pProc);
void printResults(const char *fileName, const int &nPoints, const int &nThreads,const double &dt, const double &dx,const double &elapsedTime);
#endif