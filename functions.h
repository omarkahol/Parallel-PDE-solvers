#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <fstream>
#include <cmath>
#include <iostream>

const double C = 5;
const double K = .02;

void writeData(double **data, const int &nTIME, const int &nPOINTS, const char *fileName);
double **allocateMemory(const int &nTIME, const int &nPOINTS);
void freeMemory(double **pMemory, const int &nTIME);
void initGrid(double **pMemory, double (*pFunction)(int), const int &nPOINTS);
void evolvePDE(double **pMemory, const int &currentIT, const int &nPOINTS, const int &start, const int &end, const double &DX, const double &DT);

#endif