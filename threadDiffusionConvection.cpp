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
const int NPOINTS_THREAD = 20; //NUMERO DI PUNTI PER OGNI THREAD
const int NPOINTS = NTHREADS*NPOINTS_THREAD - 2*NTHREADS; //NUMERO DI PUNTI TOTALE
const double LENGTH = 1;
const double DX = LENGTH/(NPOINTS-1);
const int ITERATIONS_TIME=500;
const double C = 5;
const double K = .02;
const double SIGMA = 0.1;
const double DT = SIGMA*DX;

//VARIABILI PER I THREADS
std::mutex lockCounter;
int counterThread = 0;
bool canExecute[NTHREADS];

/*THREAD PER EVOLVERE UOLD E SALVARE I RISULTATI IN UNEW --> ne verranno creati NTHREADS
Si occupa di evolvere solo della fetta di array [start, end]
*/
void evolveThread(int ID, int start, int end, double* uNew, const double* uOld) {

    for(int t = 0; t<ITERATIONS_TIME; t++) {

        // ASPETTA CHE IL THREAD PRINCIPALE ABBIA SALVATO I RISULTATI DELLA PRECEDENTE ITERAZIONE
        while (true) {
            if (canExecute[ID]) {
                break;
            }
        }

        //EVOLVE UOLD E SALVA I RISULTATI IN UNEW
        for (int i=start+1; i<end-1; i++) {
            uNew[(i)%NPOINTS] = uOld[(i)%NPOINTS] + DT*(  -C*((uOld[(i)%NPOINTS]-uOld[(i-1)%NPOINTS])/DX) + K*(uOld[(i+1)%NPOINTS]-2*uOld[i%NPOINTS]+uOld[(i-1)%NPOINTS])/std::pow(DX,2)  );
        }

        // AVVISA IL THREAD PRINCIPALE CHE HA FINITO L'ITERAZIONE MODIFICANDO UNA VARIABILE counter
        lockCounter.lock();
        counterThread += 1;
        canExecute[ID] = false; //STOPPA L'ESECUZIONE FINO A CHE IL THREAD PRINCIPALE NON AVRA' AGGIORNATO I RISULTATI
        lockCounter.unlock();
    }
}

/* QUESTO E' IL THREAD PRINCIPALE --> NE VIENE CREATO SOLO 1
Il suo scopo è quello di aspettare che tutti i thread abbiano finito di lavorare sulla loro parte di array
Quando hanno finito (cioè quando counter == NTHREADS) sostituisce uOld con uNew in modo che l'iterazione successiva possa avvenire
Nel mentre si occupa di risvegliare ogni thread settando canExecute[ID] = true e salva i risultati dell'iterazione in uno storage
*/
void dumperThread(double* uOld, const double* uNew, std::vector<double>* storage) {

    for (int t=0; t<ITERATIONS_TIME; t++) {

        // Aspetta che tutti i thread abbiano finito di lavorare su uOld
        while (true) {
            if (counterThread == NTHREADS) {
                break;
            }
        }

        // Resetta il contatore
        lockCounter.lock();
        counterThread = 0;
        lockCounter.unlock();

        //Sostituisce unew con uold e salva i risultati nello storage
        for (int i=0; i<NPOINTS; i++) {
            uOld[i] = uNew[i];
            storage[t].push_back(uNew[i]);
        }

        // RISVEGLIA OGNI THREAD
        for (int i=0; i<NTHREADS; i++) {
            canExecute[i] = true;
        }

    }
}

//FUNZIONE PER SALVARE I DATI IN UN FILE CSV
void writeData (const std::vector<double>* storage, std::string&& name){
    std::fstream f;
    f.open(name,std::ios::out);

    for (int i=0; i<ITERATIONS_TIME; i++) {
        for (int j=0; j<NPOINTS; j++) {
            if (j != NPOINTS-1)
                f << storage[i].at(j) << ",";
            else
                f << storage[i].at(j)<<std::endl;
        }
    }
    f.close();
}

//METODO MAIN
int main() {

    //INIZIALIZZO LE VARIABILI
    double grid[NPOINTS];
    double uOld[NPOINTS];
    double uNew[NPOINTS];

    //RIEMPIO UOLD E UNEW
    for (int i=0; i<NPOINTS; i++) {
        grid[i] = i*DX;
        uOld[i]=std::sin(2*M_PI*grid[i]/LENGTH);
        uNew[i]=uOld[i];
    }

    // CREO IL THREAD PRINCIPALE
    std::cout << "INIZIO INTEGRAZIONE" << std::endl;
    auto* storage = new std::vector<double>[ITERATIONS_TIME];
    std::thread dump(dumperThread, uOld, uNew, storage);

    //CREO UN NUMERO NTHREADS DI THREADS "LAVORATORI"
    std::vector<std::thread> threadPool;
    for (int i=0; i<NTHREADS; i++) {
        canExecute[i] = true;
        int start = i*(NPOINTS_THREAD-2); //OGNI THREAD LAVORA SOLO SU UNA FETTA DELL'ARRAY
        int end = (i+1)*(NPOINTS_THREAD+2);
        threadPool.push_back(std::thread(evolveThread,i,start,end,uNew,uOld));
    }

    for(int i=0; i<NTHREADS; i++) {
        threadPool.at(i).join();       //ASPETTIAMO LA FINE DEL PROCESSO
    }

    dump.join(); //ASPETTIAMO CHE ANCHE IL THREAD PRINCIPALE ABBIA FINITO
    std::cout << "FINE INTEGRAZIONE" << std::endl;
    //DUMP TO FILE
    writeData(storage,"dataFile.csv");
    delete[] storage;

    //POST PROCESSING
    std::system("python3 postProcessorDiffusionConvection.py");
    return 0;
}