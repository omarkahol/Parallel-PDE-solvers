#include <iostream>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <vector>

const int REQUEST_THREADS = 20; //NUMERO DI THREADS DA RICHIEDERE
int NTHREADS; //NUMERO DI THREDS CONCESSI
const int NPOINTS_THREAD = 5; //PUNTI SU CUI LAVORA OGNI THREAD
int NPOINTS; //NUMERO DI PUNTI TOTALI IN CUI È STAT DIVISA LA GRIGLIA
const double LENGTH = 1; 
double DX;
const int ITERATIONS_TIME=1000;
const double C = 5;
const double K = .02;
const double SIGMA = 0.1;
double DT;

//FUNZIONE PER SALVARE I DATI IN UN FILE CSV
void writeData (double **storage, std::string&& name){
    std::fstream f;
    f.open(name,std::ios::out);

    for (int i=0; i<ITERATIONS_TIME; i++) {
        for (int j=0; j<NPOINTS; j++) {
            if (j != NPOINTS-1)
                f << storage[i][j] << ",";
            else
                f << storage[i][j]<<std::endl;
        }
    }
    f.close();
}

//LIBERA LA MEMORIA RICHIESTA
void freeMemory(double **storage) {
  for (int t=0; t<ITERATIONS_TIME; t++){
    delete[] storage[t];
  }
  delete[] storage;
}

int main() {
  omp_set_num_threads(REQUEST_THREADS); //RICHIEDE I THREAD
  bool executing[REQUEST_THREADS]{false}; //VARIABILE PER SALVARE LO STATO DEI THREAD (SONO IN ESECUZIONE O NO)

  /* MOLTO PERICOLOSO
  *  Richiede memoria nell'heap del sistema
  */
  double **pDATA_STORAGE = new double*[ITERATIONS_TIME];
  
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

  //RICHIEDE MEMORIA DINAMICA NELL'HEAP
  for (int t=0; t<ITERATIONS_TIME; t++) {
    pDATA_STORAGE[t] = new double[NPOINTS];
  }

  //CONDIZIONE INIZIALE
  for (int i=0; i<NPOINTS; i++){
    pDATA_STORAGE[0][i] = std::sin(2*M_PI*i*DX/LENGTH);
  }

  //OGNI THREAD EVOLVE LA SUA PORZIONE DI ARRAY
  #pragma omp parallel 
  {
    int ID = omp_get_thread_num();
    int IDbefore = (ID==0)?NTHREADS-1:ID-1;
    int IDafter = (ID == NTHREADS-1)?0:ID+1;
    int start = ID*(NPOINTS_THREAD-2);
    int end = (ID+1)*(NPOINTS_THREAD +2);

    for (int t=0; t<ITERATIONS_TIME-1; t++) {
      executing[ID] = true;
      for (int i=start+1; i<end-1; i++) {
        pDATA_STORAGE[t+1][(i)%NPOINTS] = pDATA_STORAGE[t][(i)%NPOINTS]
          + DT*(  -C*((pDATA_STORAGE[t][(i)%NPOINTS]-pDATA_STORAGE[t][(i-1)%NPOINTS])/DX)
            + K*(pDATA_STORAGE[t][(i+1)%NPOINTS]-2*pDATA_STORAGE[t][i%NPOINTS]
              + pDATA_STORAGE[t][(i-1)%NPOINTS])/std::pow(DX,2)  );
      }
      executing[ID] = false;
      while(executing[IDbefore] || executing[IDafter]) {}    
    }
  }
  
  writeData(pDATA_STORAGE,"dataFile.csv");
  freeMemory(pDATA_STORAGE);
  std::system("python3 postProcessorDiffusionConvection.py");
  return 0;
}