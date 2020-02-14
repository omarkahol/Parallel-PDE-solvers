#include <vector>
#include <thread>
#include "functions.h"

/*
COMPILA --> g++ threadDiffusionConvection.cpp functions.cpp -pthread -o out
ESEGUI --> ./out [numero di thread][numero di punti] [sigma] [mostra i risultati]
*/


int NPOINTS;
int NTHREADS;
double DX;
double DT;
const int ITERATIONS_TIME=1000;
bool *threadState;

void evolveThread(int ID, int start, int end,double **pDATA_STORAGE) { //SINGOLO THREAD

    int IDbefore = (ID==0)?NTHREADS-1:ID-1;
    int IDafter = (ID==NTHREADS-1)?0:ID+1;

    for(int t = 0; t<ITERATIONS_TIME-1; t++) {
        threadState[ID]=true;
        evolvePDE(pDATA_STORAGE, t, NPOINTS, start, end, DX, DT);
        threadState[ID] = false;
        while(threadState[IDbefore] || threadState[IDafter]) {}
    }
}

int main(int argc, char*argv[]) {
    //VARIABILI DEL SISTEMA
    NTHREADS = std::atoi(argv[1]); //NUMERO DI THREADS
    const int NPOINTS_THREAD = std::atoi(argv[2]); //NUMERO DI PUNTI PER OGNI THREAD
    NPOINTS = NTHREADS*NPOINTS_THREAD - 2*NTHREADS; //NUMERO DI PUNTI TOTALE
    const double LENGTH = 1;
    DX = LENGTH/(NPOINTS-1);
    const double SIGMA = std::atof(argv[3]);
    DT = SIGMA*DX;
    const bool SHOW_RESULTS = (bool) std::atoi(argv[4]);
    threadState = new bool[NTHREADS]; //True se il thread è in esecuzione False se non lo è

    //RICHIEDE MEMORIA E INIZIALIZZA L'ARRAY
    double **pDATA_STORAGE = allocateMemory(ITERATIONS_TIME, NPOINTS);
    initGrid(pDATA_STORAGE, NPOINTS, DX, LENGTH);

    //CREO UN NUMERO NTHREADS DI THREADS
    std::vector<std::thread> threadPool;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<NTHREADS; i++) {
        int start = i*(NPOINTS_THREAD-2); //OGNI THREAD LAVORA SOLO SU UNA FETTA DELL'ARRAY
        int end = (i+1)*(NPOINTS_THREAD+2);
        threadPool.push_back(std::thread(evolveThread,i,start,end,pDATA_STORAGE));
    }

    //ATTENDO CHE OGNI THREAD ABBIA TERMINATO
    for(int i=0; i<NTHREADS; i++) {
        threadPool.at(i).join();       //TERMINE DEL PROCESSO
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    delete[] threadState;
    
    writeData(pDATA_STORAGE,ITERATIONS_TIME,NPOINTS,"dataFile.csv");
    printResults("resultsFile.csv",NPOINTS,NTHREADS,DT,DX,duration.count());
    freeMemory(pDATA_STORAGE, ITERATIONS_TIME);
    
    if (SHOW_RESULTS){
        std::system("python3 postProcessorDiffusionConvection.py");
    }
    return 0;
}