#include "functions.h"
#include <omp.h>
#include <chrono>
/*
COMPILA --> g++ openmpDiffusionConvection.cpp functions.cpp -fopenmp -o out
ESEGUI --> ./out [numero di thread] [numero di punti] [sigma] [mostra i risultati] [iterazioni nel tempo]
*/

int main(int argc, char *argv[]) {

  const int requestThreads = std::atoi(argv[1]);
  const int nPoints = std::atoi(argv[2]);
  const double sigma = std::atof(argv[3]);
  const bool showResults = (bool) std::atoi(argv[4]);
  const int nTime=std::atoi(argv[5]);
  const double dx = 1.0/(nPoints-1.0);
  const double dt = sigma*dx;
  const double length = 1.0; 

  omp_set_num_threads(requestThreads);

  double ***pMemory = allocateMemory(nTime,nPoints);
  initGrid(pMemory,nPoints,dx,1);

  int nThreads;

  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
  {
    #pragma omp master
    nThreads = omp_get_num_threads();

    for(int t=0; t<nTime-1; t++){

      #pragma omp for schedule(auto)
      for(int i=1; i<nPoints+2; i++){
        pMemory[t+1][(i)%nPoints][0] = pMemory[t][(i)%nPoints][0]
          + dt*(  -c*((pMemory[t][(i+1)%nPoints][0]-pMemory[t][std::abs(i-1)%nPoints][0])/(2*dx))
            + k*(pMemory[t][(i+1)%nPoints][0]-2*pMemory[t][i%nPoints][0]
              + pMemory[t][std::abs(i-1)%nPoints][0])/std::pow(dx,2)  );
      }
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
  writeData(pMemory,nTime,nPoints,"dataFile.csv");
  printResults("resultsFile.csv",nPoints,nThreads,dt,dx,duration.count());
  freeMemory(pMemory, nTime, nPoints);
    
  if (showResults){
    std::system("python3 postProcessorDiffusionConvection.py");
  }
  return 0;
}
