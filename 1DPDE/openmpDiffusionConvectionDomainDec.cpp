#include <omp.h>
#include "functions.h"
#include <chrono>

/*
COMPILA --> g++ openmpDiffusionConvection.cpp functions.cpp -fopenmp -o out
ESEGUI --> ./out [numero di thread] [numero di punti] [sigma] [mostra i risultati] [iterazioni nel tempo]
*/

int main(int argc, char *argv[]) {

  const int requestThreads = std::atoi(argv[1]);
  const int nPointsThread = std::atoi(argv[2]);
  const double sigma = std::atof(argv[3]);
  const bool showResults = (bool) std::atoi(argv[4]);
  const int nTime=std::atoi(argv[5]);
  int nThreads, nPoints;
  double dx, dt;
  const double length = 1; 

  omp_set_num_threads(requestThreads); //RICHIEDE I THREAD
  
  //VERIFICA IL NUMERO DI THREAD CHE È POSSIBILE RICHIEDERE E PROVA A CREARE I RIMANENTI
  #pragma omp parallel 
  {
    int id = omp_get_thread_num(); 
    if (id == 0) {
      nThreads = omp_get_num_threads();
      nPoints = nThreads*nPointsThread - 2*nThreads;
      dx = length/(nPoints-1);
      dt = sigma*dx;
    }
  }
  
  Processor1D processorPool[nThreads];
  for (int i=0; i<nThreads; i++) {
    processorPool[i] = Processor1D{i*(nPointsThread-2),nPointsThread,nPoints, dx, dt};
  }

  double ***pGrid = allocateMemory(nTime, nPoints);
  initGrid(pGrid, nPoints, dx, length);

  //OGNI THREAD EVOLVE LA SUA PORZIONE DI ARRAY
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel 
  {
    int id = omp_get_thread_num();
    for (int t=0; t<nTime-1; t++) {
      evolvePDE(pGrid, t, &(processorPool[id]));
      #pragma omp barrier    
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
  writeData(pGrid,nTime,nPoints,"dataFile.csv");
  printResults("resultsFile.csv",nPoints,nThreads,dt,dx,duration.count());
  freeMemory(pGrid, nTime, nPoints);
    
  if (showResults){
    std::system("python3 postProcessorDiffusionConvection.py");
  }
  return 0;
}