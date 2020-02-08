#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <fstream>
#include <cstring>

//VARIABILI DEL SISTEMA
const int NTHREADS = 4; //NUMERO DI THREADS
const int NPOINTS_THREAD = 21; //NUMERO DI PUNTI PER OGNI THREAD
const int NPOINTS = NTHREADS*NPOINTS_THREAD - 2*NTHREADS; //NUMERO DI PUNTI TOTALE
const double LENGTH = 1;
const double DX = LENGTH/(NPOINTS-1);
const int ITERATIONS_TIME=1000;
const double C = 5;
const double K = .02;
const double SIGMA = 0.1;
const double DT = SIGMA*DX;

//VARIABILE PER SALVARE IL NUMERO DI ITERAZIONI EFFETTUATE
long numberExecution = 0;
std::mutex modifyNumExec;

/*
THREAD
Ogni thread lavora e aggiorna lo stesso file: dataStorage. Tuttavia ogni thread si occupa di evolvere solo 
i punti interni al suo dominio assegnato tramite [start, end]. Alla fine modifica la variabile numberExecution
per avviasare che l'iterazione è stata completata con successo.
*/
void evolveThread(int start, int end, std::vector<std::vector<double>>& dataStorage) {

    for(int t = 0; t<ITERATIONS_TIME-1; t++) {

        //ASPETTA CHE TUTTI I THREAD ABBIANO TERMINATO L'ITERAZIONE PRECEDENTE
        while (true) {
            if ( numberExecution >= t*NTHREADS) {
                break;
            }
        }

        //EVOLVE UOLD E SALVA I RISULTATI IN UNEW
        for (int i=start+1; i<end-1; i++) {
            dataStorage[t+1][(i)%NPOINTS] = dataStorage[t][(i)%NPOINTS]
                + DT*(  -C*((dataStorage[t][(i)%NPOINTS]-dataStorage[t][(i-1)%NPOINTS])/DX)
                    + K*(dataStorage[t][(i+1)%NPOINTS]-2*dataStorage[t][i%NPOINTS]+dataStorage[t][(i-1)%NPOINTS])/std::pow(DX,2)  );
        }

        modifyNumExec.lock();
        numberExecution += 1; //AVVISA DI AVER EFFETTUATO UNA ITERAZIONE
        modifyNumExec.unlock();

    }
}

//FUNZIONE PER SALVARE I DATI IN UN FILE CSV
void writeData (std::vector<std::vector<double>>& storage, std::string&& name){
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

//METODO MAIN
int main() {

    //CREAZIONE DELL'ARRAY DI MEMORIA CONDIVISA IN CUI LAVORERANNO I THREADS
    std::vector<std::vector<double>> dataStorage(ITERATIONS_TIME, std::vector<double>(NPOINTS));
    for (int i=0; i<NPOINTS; i++) {
        dataStorage[0][i]=std::sin(2*M_PI*i*DX/LENGTH);
    }

    //CREO UN NUMERO NTHREADS DI THREADS 
    std::vector<std::thread> threadPool;
    for (int i=0; i<NTHREADS; i++) {
        int start = i*(NPOINTS_THREAD-2); //OGNI THREAD LAVORA SOLO SU UNA FETTA DELL'ARRAY
        int end = (i+1)*(NPOINTS_THREAD+2);
        threadPool.push_back(std::thread(evolveThread,start,end,std::ref(dataStorage))); //INIZIO DEL PROCESSO
    }

    for(int i=0; i<NTHREADS; i++) {
        threadPool.at(i).join();       //TERMINE DEL PROCESSO
    }

    //DUMP TO FILE
    writeData(dataStorage,"dataFile.csv");

    //POST PROCESSING
    std::system("python3 postProcessorDiffusionConvection.py");
    return 0;
}