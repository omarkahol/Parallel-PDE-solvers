#include <vector>
#include <thread>
#include "functions.h"

//VARIABILI DEL SISTEMA
const int NTHREADS = 15; //NUMERO DI THREADS
const int NPOINTS_THREAD = 6; //NUMERO DI PUNTI PER OGNI THREAD
const int NPOINTS = NTHREADS*NPOINTS_THREAD - 2*NTHREADS; //NUMERO DI PUNTI TOTALE
const double LENGTH = 1;
const double DX = LENGTH/(NPOINTS-1);
const int ITERATIONS_TIME=1000;
const double SIGMA = 0.1;
const double DT = SIGMA*DX;
bool threadState[NTHREADS]; //True se il thread è in esecuzione False se non lo è


/*
THREAD
Ogni thread lavora e aggiorna lo stesso file: pDATA_STORAGE. Tuttavia ogni thread si occupa di evolvere solo 
i punti interni al suo dominio assegnato tramite [start, end]. Alla fine modifica la variabile indicatrice del suo stato.
Prima di cominciare la nuova iterzione, attende che i due thread confinanti abbiano terminato le loro.
*/
void evolveThread(int ID, int start, int end, double **pDATA_STORAGE) {

    int IDbefore = (ID==0)?NTHREADS-1:ID-1;
    int IDafter = (ID==NTHREADS-1)?0:ID+1;

    for(int t = 0; t<ITERATIONS_TIME-1; t++) {
        threadState[ID]=true;
        evolvePDE(pDATA_STORAGE, t, NPOINTS, start, end, DX, DT);
        threadState[ID] = false;
        while(threadState[IDbefore] || threadState[IDafter]) {}
    }
}

int main() {

    //RICHIEDE MEMORIA E INIZIALIZZA L'ARRAY
    double **pDATA_STORAGE = allocateMemory(ITERATIONS_TIME, NPOINTS);
    auto initialCondition = [DX, LENGTH](int i){
        return std::sin(2*M_PI*i*DX/LENGTH);
    };
    initGrid(pDATA_STORAGE, initialCondition, NPOINTS);

    //CREO UN NUMERO NTHREADS DI THREADS 
    std::vector<std::thread> threadPool;
    for (int i=0; i<NTHREADS; i++) {
        int start = i*(NPOINTS_THREAD-2); //OGNI THREAD LAVORA SOLO SU UNA FETTA DELL'ARRAY
        int end = (i+1)*(NPOINTS_THREAD+2);
        threadPool.push_back(std::thread(evolveThread,i, start, end, pDATA_STORAGE)); //INIZIO DEL PROCESSO
    }

    //ATTENDO CHE OGNI THREAD ABBIA TERMINATO
    for(int i=0; i<NTHREADS; i++) {
        threadPool.at(i).join();       //TERMINE DEL PROCESSO
    }

    //POSTPROCESSORE
    writeData(pDATA_STORAGE,ITERATIONS_TIME,NPOINTS,"dataFile.csv");
    freeMemory(pDATA_STORAGE, ITERATIONS_TIME);
    std::system("python3 postProcessorDiffusionConvection.py");
    return 0;
}