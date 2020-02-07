#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include <mpi/mpi.h>
#include <chrono>

// VARIABILI DEL PROBLEMA
const int N_POINTS_PROCESS =20;
const double LENGTH = 1;
const int ITERATIONS_TIME=1000;
const double C = 5;
const double K = 0.001;

void writeData (int npoints, const std::vector<double>* storage, std::string&& name){
    std::fstream f;
    f.open(name,std::ios::out);

    for (int i=0; i<ITERATIONS_TIME; i++) {
        for (int j=0; j<npoints; j++) {
            if (j != npoints-1)
                f << storage[i].at(j) << ",";
            else
                f << storage[i].at(j)<<std::endl;
        }
    }
    f.close();
}

int main(int argc, char* argv[]) {

    //INIZIALIZZO MPI
    int ierr, procID, numProcs;
    ierr = MPI_Init(&argc, &argv); // INIZIALIZZA LE VARIABILI DI MPI
    ierr = MPI_Comm_rank(MPI_COMM_WORLD,&procID); //DETERMINA L'ID DEL PROCESSO
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numProcs); // DETERMINA IL NUMERO DI PROCESSI
    MPI_Status status;

    const int N_POINTS_TOTAL = numProcs*N_POINTS_PROCESS - 2*numProcs;
    const double DX = LENGTH/(N_POINTS_TOTAL-1);
    const double sigma = 0.1;
    const double DT = sigma*DX;

    if (procID==0) {
        std::cout << "Solving for nPOINTS = " << N_POINTS_TOTAL << std::endl;
    }

    double processGrid[N_POINTS_PROCESS];
    double phase = procID*(N_POINTS_PROCESS-2)*DX;

    for (int i=0; i<N_POINTS_PROCESS; i++){
        processGrid[i] = phase + i*DX;
    }

    if (procID == numProcs -1) {
        processGrid[N_POINTS_PROCESS-2] = 0;
        processGrid[N_POINTS_PROCESS-1] = DX;
    }

    //IMPONIAMO LA CONDIZIONE INIZIALE
    double uOld[N_POINTS_PROCESS];
    double uNew[N_POINTS_PROCESS];
    for (int i=0; i<N_POINTS_PROCESS; i++) {
        uOld[i]=std::sin(2*M_PI*processGrid[i]/LENGTH);
        uNew[i]=std::sin(2*M_PI*processGrid[i]/LENGTH);
    }

    int it = 0;

    double storage[ITERATIONS_TIME][N_POINTS_PROCESS];

    auto start = std::chrono::high_resolution_clock::now();
    //EVOLVIAMO NEL TEMPO
    while (it++<ITERATIONS_TIME) {

        //EVOLVO NEI PUNTI DEL DOMINIO
        for (int i = 1; i<N_POINTS_PROCESS-1; i++){
            // EVOLVO LA FUNZIONE ATTRAVERSO Ut + cUx = kUxx
            uNew[i] = uOld[i] + DT*(  -C*((uOld[i]-uOld[i-1])/DX) + K*(uOld[i+1]-2*uOld[i]+uOld[i-1])/std::pow(DX,2)  );
        }

        //COMUNICO I RISULTATI CON GLI ALTRI PROCESSI
        // Ricevo il primo nodo, spedisco il secondo e il penultimo e ricevo l'ultimo
        if (procID == 0) {
            MPI_Send(&uNew[1],1,MPI_DOUBLE,numProcs-1,0,MPI_COMM_WORLD);
            MPI_Recv(&uNew[0],1,MPI_DOUBLE,numProcs-1,0,MPI_COMM_WORLD, &status);
            MPI_Send(&uNew[N_POINTS_PROCESS-2],1,MPI_DOUBLE,procID+1,0,MPI_COMM_WORLD);
            MPI_Recv(&uNew[N_POINTS_PROCESS-1],1,MPI_DOUBLE,procID+1,0,MPI_COMM_WORLD, &status);
        } else if (procID == numProcs-1) {
            MPI_Recv(&uNew[N_POINTS_PROCESS-1],1,MPI_DOUBLE,0,0,MPI_COMM_WORLD, &status);
            MPI_Send(&uNew[N_POINTS_PROCESS-2],1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
            MPI_Recv(&uNew[0],1,MPI_DOUBLE,procID-1,0,MPI_COMM_WORLD, &status);
            MPI_Send(&uNew[1],1,MPI_DOUBLE,procID-1,0,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&uNew[0],1,MPI_DOUBLE,procID-1,0,MPI_COMM_WORLD, &status);
            MPI_Send(&uNew[1],1,MPI_DOUBLE,procID-1,0,MPI_COMM_WORLD);
            MPI_Send(&uNew[N_POINTS_PROCESS-2],1,MPI_DOUBLE,procID+1,0,MPI_COMM_WORLD);
            MPI_Recv(&uNew[N_POINTS_PROCESS-1],1,MPI_DOUBLE,procID+1,0,MPI_COMM_WORLD, &status);
        }
        for (int i=0; i<N_POINTS_PROCESS; i++) {
            uOld[i] = uNew[i];
            storage[it-1][i] = uNew[i];
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    //PRINT ELAPSED TIME
    if (procID == 0) {
        std::cout << "Elapsed Time: " << duration.count()<< " microseconds" << std::endl;
    }

    if (procID == 0) {

        auto* toDump = new std::vector<double>[ITERATIONS_TIME];

        //START FILLING STORAGE
        for(int i=0; i<ITERATIONS_TIME; i++) {
            for (int j=0; j<N_POINTS_PROCESS-2; j++) {
                toDump[i].push_back(storage[i][j]);
            }
        }

        for (int sender = 1; sender<numProcs; sender++) {
            double toReceive[ITERATIONS_TIME][N_POINTS_PROCESS];
            MPI_Recv(&toReceive,ITERATIONS_TIME*N_POINTS_PROCESS,MPI_DOUBLE,sender,0,MPI_COMM_WORLD, &status);
            for(int i=0; i<ITERATIONS_TIME; i++) {
                for (int j=0; j<N_POINTS_PROCESS-2; j++) {
                    toDump[i].push_back(toReceive[i][j]);
                }
            }
        }

        //WRITE DATA AND CALL POSTPROCESSOR
        writeData(N_POINTS_TOTAL , toDump,"dataFile.csv");
        delete[] toDump;
        std::system("python3 postProcessorDiffusionConvection.py");
    } else {
        MPI_Send(&storage,N_POINTS_PROCESS*ITERATIONS_TIME,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    }


    ierr = MPI_Finalize();
    return 0;
}