#include <omp.h>
#include "functions.h"
#include <chrono>

/*
COMPILA --> g++ openmpDiffusionConvection.cpp functions.cpp -fopenmp -o out
ESEGUI --> ./out [numero di thread] [numero di punti] [sigma] [mostra i risultati]
*/

int REQUEST_THREADS; //NUMERO DI THREADS DA RICHIEDERE
int NTHREADS; //NUMERO DI THREDS CONCESSI
int NPOINTS_THREAD; //PUNTI SU CUI LAVORA OGNI THREAD
int NPOINTS; //NUMERO DI PUNTI TOTALI IN CUI È STAT DIVISA LA GRIGLIA
const double LENGTH = 1; 
double DX;
const int ITERATIONS_TIME=1000;
double SIGMA;
double DT;

int main(int argc, char *argv[]) {

  const int REQUEST_THREADS = std::atoi(argv[1]);
  const int NPOINTS_THREAD = std::atoi(argv[2]);
  const double SIGMA = std::atof(argv[3]);
  bool SHOW_RESULTS = (bool) std::atoi(argv[4]);

  omp_set_num_threads(REQUEST_THREADS); //RICHIEDE I THREAD
  
  //VERIFICA IL NUMERO DI THREAD CHE È POSSIBILE RICHIEDERE E PROVA A CREARE I RIMANENTI
  #pragma omp parallel 
  {
    int ID = omp_get_thread_num(); 
    if (ID == 0) {
      NTHREADS = omp_get_num_threads();
      NPOINTS = NTHREADS*NPOINTS_THREAD - 2*NTHREADS;
      DX = LENGTH/(NPOINTS-1);
      DT = SIGMA*DX;
    }
  }

  double **pDATA_STORAGE = allocateMemory(ITERATIONS_TIME, NPOINTS);
  initGrid(pDATA_STORAGE, NPOINTS, DX, LENGTH);

  //OGNI THREAD EVOLVE LA SUA PORZIONE DI ARRAY
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel 
  {
    int ID = omp_get_thread_num();
    int start = ID*(NPOINTS_THREAD-2);
    int end = start + NPOINTS_THREAD;

    for (int t=0; t<ITERATIONS_TIME-1; t++) {
      evolvePDE(pDATA_STORAGE, t, NPOINTS, start, end, DX, DT);
      #pragma omp barrier    
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
  writeData(pDATA_STORAGE,ITERATIONS_TIME,NPOINTS,"dataFile.csv");
  printResults("resultsFile.csv",NPOINTS,NTHREADS,DT,DX,duration.count());
  freeMemory(pDATA_STORAGE, ITERATIONS_TIME);
    
  if (SHOW_RESULTS){
    std::system("python3 postProcessorDiffusionConvection.py");
  }
  return 0;
}