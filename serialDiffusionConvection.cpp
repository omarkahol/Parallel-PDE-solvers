#include <cstring>
#include <chrono>
#include "functions.h"

const int NPOINTS = 70;
const double LENGTH = 1;
const double DX = LENGTH/(NPOINTS-1);
const double SIGMA = 0.1;
const double DT = SIGMA*DX;
const int ITERATIONS_TIME=1000;

int main() {

    double **pDATA_STORAGE = allocateMemory(ITERATIONS_TIME,NPOINTS);
    auto initialCondition = [DX, LENGTH](int i){
        return std::sin(2*M_PI*i*DX/LENGTH);
    };
    initGrid(pDATA_STORAGE, initialCondition, NPOINTS);

    auto start = std::chrono::high_resolution_clock::now();
    for(int t=0; t<ITERATIONS_TIME-1; t++) {
        evolvePDE(pDATA_STORAGE, t, NPOINTS, -1, NPOINTS+2, DX, DT);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Elapsed Time: " << duration.count()<< " microseconds" << std::endl;

    writeData(pDATA_STORAGE,ITERATIONS_TIME,NPOINTS,"dataFile.csv");
    freeMemory(pDATA_STORAGE, ITERATIONS_TIME);
    std::system("python3 postProcessorDiffusionConvection.py");
    return 0;
}
