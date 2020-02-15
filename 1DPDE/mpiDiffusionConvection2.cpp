#include "functionsMPI.h"
#include <chrono>

int main(int argc, char *argv[]) {
  //INIZIALIZZO MPI
  int ierr, id, nProcs;
  ierr = MPI_Init(0, 0); // INIZIALIZZA LE VARIABILI DI MPI
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&id); //DETERMINA L'id DEL PROCESSO
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // DETERMINA IL NUMERO DI PROCESSI
  MPI_Status status; // STATO DEL PROCESSO

  //VARIABILI
  const int nPointsProc = std::atoi(argv[1]);
  const double sigma = std::atof(argv[2]);
  const bool showResults = bool(std::atoi(argv[3]));
  const int nTime=5;
  const int nPointsTotal = nProcs*nPointsProc - 2*nProcs;
  const double length=1;
  const double dx = length/(nPointsTotal-1);
  const double dt = sigma*dx;
  
  double **pGrid = allocateMemory(nTime,nPointsProc);
  initProcessGrid(pGrid,id,nProcs,nPointsProc,dx,length);

  Processor1D processor{0,nPointsProc,nPointsTotal,dx,dt};

  auto start = std::chrono::high_resolution_clock::now();
  for(int t=0; t<nTime-1; t++) {
    evolvePDE(pGrid, t, &processor);
    mpiCommunicateResults(pGrid, t+1, nPointsProc, id, nProcs, &status);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  if (id==0) {
    double **dumpMemory = finalizeResults(pGrid,&processor,nTime, nProcs,&status);
    writeData(dumpMemory,nTime,nPointsTotal,"dataFile.csv");
    printResults("resultsFile.csv",nPointsTotal,nProcs,dt,dx,duration.count());
    freeMemory(dumpMemory, nTime);
    if (showResults){
      std::system("python3 postProcessorDiffusionConvection.py");
    }
  } else {
    for (int t=0; t<nTime; t++) {
      for(int i=0; i<nPointsProc-2; i++){
        MPI_Send(&(pGrid[t][i]),1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
      }
    }
    freeMemory(pGrid, nTime);
  }

  ierr = MPI_Finalize();
  return 0;
}