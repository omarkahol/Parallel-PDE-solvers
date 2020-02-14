#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>

const double length = 1;
const double cX = 1.0;
const double cY = 3.0;
const double k = 0.1;

struct Processor{
int startX;
int startY;
int nPointsX;
int nPointsY;
double dx;
double dy;
int nPointsTotX;
int nPointsTotY;
};

double ***allocateMemory(const int& nTime, const int &nX, const int &nY);
void initGrid(double ***pMemory, const int &nX, const int &nY);
void coolPrintGrid(double ***pGrid, const int &currentT, const int &nX, const int &nY);
void freeMemory (double ***pMemory, const int &nTime, const int &nY);
void evolvePDE(double ***uGrid, double ***vGrid,const int &t, const double &dt, Processor *pProc);
void writeFile(double ***uGrid, double ***vGrid, const int &nX, const int &nY, const int &nTime, const char *fileName);
void createProcessors(const int &nProcX, const int &nProcY, const int &nXproc, const int &nYproc, std::vector<Processor> &procPool);
#endif