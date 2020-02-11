#include <omp.h>
#include "functions.h"

const int REQUEST_THREADS = 20; //NUMERO DI THREADS DA RICHIEDERE
int NTHREADS; //NUMERO DI THREDS CONCESSI
const int NPOINTS_THREAD = 5; //PUNTI SU CUI LAVORA OGNI THREAD
int NPOINTS; //NUMERO DI PUNTI TOTALI IN CUI È STAT DIVISA LA GRIGLIA
const double LENGTH = 1; 
double DX;
const int ITERATIONS_TIME=1000;
const double SIGMA = 0.1;
double DT;

int main() {
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

  auto initialCondition = [DX, LENGTH](int i){
    return std::sin(2*M_PI*i*DX/LENGTH);
  };

  initGrid(pDATA_STORAGE, initialCondition, NPOINTS);

  //OGNI THREAD EVOLVE LA SUA PORZIONE DI ARRAY
  #pragma omp parallel 
  {
    int ID = omp_get_thread_num();
    int start = ID*(NPOINTS_THREAD-2);
    int end = (ID+1)*(NPOINTS_THREAD +2);

    for (int t=0; t<ITERATIONS_TIME-1; t++) {
      evolvePDE(pDATA_STORAGE, t, NPOINTS, start, end, DX, DT);
      #pragma omp barrier    
    }
  }
  
  writeData(pDATA_STORAGE,ITERATIONS_TIME,NPOINTS,"dataFile.csv");
  freeMemory(pDATA_STORAGE, ITERATIONS_TIME);
  std::system("python3 postProcessorDiffusionConvection.py");
  return 0;
}