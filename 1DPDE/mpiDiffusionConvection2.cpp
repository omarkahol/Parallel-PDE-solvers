#include "functionsMPI.h"
#include <chrono>

const int ITERATIONS_TIME=500;
int NPOINTS_PROCESS;
const double LENGTH=1;
int NPOINTS_TOTAL;
double DX;
double DT;
double SIGMA;
bool SHOW_RESULTS;

int main(int argc, char *argv[]) {
  //INIZIALIZZO MPI
  int ierr, ID, numProcs;
  ierr = MPI_Init(0, 0); // INIZIALIZZA LE VARIABILI DI MPI
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&ID); //DETERMINA L'ID DEL PROCESSO
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numProcs); // DETERMINA IL NUMERO DI PROCESSI
  MPI_Status status; // STATO DEL PROCESSO

  //VARIABILI
  NPOINTS_PROCESS = std::atoi(argv[1]);
  SIGMA = std::atof(argv[2]);
  SHOW_RESULTS = bool(std::atoi(argv[3]));
  NPOINTS_TOTAL = numProcs*NPOINTS_PROCESS - 2*numProcs;
  DX = LENGTH/(NPOINTS_TOTAL-1);
  DT = SIGMA*DX;
  
  double **PROCESSGRID = allocateMemory(ITERATIONS_TIME,NPOINTS_PROCESS);
  initProcessGrid(PROCESSGRID,ID,numProcs,NPOINTS_PROCESS,DX,LENGTH);

  auto start = std::chrono::high_resolution_clock::now();
  for(int t=0; t<ITERATIONS_TIME-1; t++) {
    evolvePDE(PROCESSGRID, t, NPOINTS_PROCESS, -1, NPOINTS_PROCESS+2, DX,DT);
    mpiCommunicateResults(PROCESSGRID, t+1, NPOINTS_PROCESS, ID, numProcs, &status);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  if (ID==1){
  writeData(PROCESSGRID,ITERATIONS_TIME,NPOINTS_PROCESS,"dataFile.csv");
  printResults("resultsFile.csv",NPOINTS_TOTAL,numProcs,DT,DX,duration.count());
  freeMemory(PROCESSGRID, ITERATIONS_TIME);

  if (SHOW_RESULTS) {
    std::system("python3 postProcessorDiffusionConvection.py");
  }

  } else {
    freeMemory(PROCESSGRID, ITERATIONS_TIME);
  }
  ierr = MPI_Finalize();
  return 0;
}