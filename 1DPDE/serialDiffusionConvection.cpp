#include <cstring>
#include <chrono>
#include "functions.h"

/*
COMPILA --> g++ serialDiffusionConvection.cpp functions.cpp -o out
ESEGUI --> ./out [numero di punti] [sigma] [mostra i risultati]
*/

int main(int argc, char *argv[]) {

    int NPOINTS = std::atoi(argv[1]);
    double LENGTH = 1;
    double DX = LENGTH/(NPOINTS-1);
    double SIGMA = std::atof(argv[2]);
    double DT = SIGMA*DX;
    int ITERATIONS_TIME=1000;
    bool SHOW_RESULTS = (bool) std::atoi(argv[3]);

    double **pDATA_STORAGE = allocateMemory(ITERATIONS_TIME,NPOINTS);
    initGrid(pDATA_STORAGE,NPOINTS, DX, LENGTH);

    auto start = std::chrono::high_resolution_clock::now();
    for(int t=0; t<ITERATIONS_TIME-1; t++) {
        evolvePDE(pDATA_STORAGE, t, NPOINTS, -1, NPOINTS+2, DX, DT);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    writeData(pDATA_STORAGE,ITERATIONS_TIME,NPOINTS,"dataFile.csv");
    printResults("resultsFile.csv",NPOINTS,1,DT,DX,duration.count());
    freeMemory(pDATA_STORAGE, ITERATIONS_TIME);
    if (SHOW_RESULTS){
    std::system("python3 postProcessorDiffusionConvection.py");
    }
    return 0;
}
