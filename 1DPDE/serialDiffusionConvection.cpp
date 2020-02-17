#include <cstring>
#include <chrono>
#include "functions.h"

/*
COMPILA --> g++ serialDiffusionConvection.cpp functions.cpp -o out
ESEGUI --> ./out [numero di punti] [sigma] [mostra i risultati] [iterazioni nel tempo]
*/

int main(int argc, char *argv[]) {

    const int nPoints = std::atoi(argv[1]);
    const double sigma = std::atof(argv[2]);
    const bool showResults = (bool) std::atoi(argv[3]);
    const int nTime=std::atoi(argv[5]);

    const double length = 1;
    const double dx = length/(nPoints-1);
    const double dt = sigma*dx;

    double ***pGrid = allocateMemory(nTime,nPoints);
    initGrid(pGrid,nPoints, dx, length);

    Processor1D processor{0,nPoints+2,nPoints, dx, dt};

    auto start = std::chrono::high_resolution_clock::now();
    for(int t=0; t<nTime-1; t++) {
        evolvePDE(pGrid, t, &processor);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    writeData(pGrid,nTime,nPoints,"dataFile.csv");
    printResults("resultsFile.csv",nPoints,1,dt,dx,duration.count());
    freeMemory(pGrid, nTime, nPoints);
    if (showResults){
    std::system("python3 postProcessorDiffusionConvection.py");
    }
    return 0;
}
