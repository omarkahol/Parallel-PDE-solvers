#include "functionsMPI.h"

void initProcessGrid(double **pMemory,const int &id,const int &nProcs,const int &nPoints, const double& dx, const double &length) {
  double phase = id*(nPoints-2)*dx;
  for (int i=0; i<nPoints; i++){
        pMemory[0][i] = std::sin(2*M_PI*(phase+i*dx)/length); //RIEMPIO LA GRIGLIA
  }
  //CONDIZIONE AL CONTORNO PERIODICA, L'ULTIMO PROCESSO LAVORA ANCHE SUI DUE NODI INIZIALI
  if (id == nProcs -1) { 
    pMemory[0][nPoints-2] = std::sin(2*M_PI*0/length);
    pMemory[0][nPoints-1] = std::sin(2*M_PI*dx/length);
  }
}

void mpiCommunicateResults(double **pMemory, const int &currentIT, const int &nPoints, const int &id, const int &nProcs,MPI_Status *status) {
  if (id==0) {
    MPI_Send(&(pMemory[currentIT][1]),1,MPI_DOUBLE,nProcs-1,0,MPI_COMM_WORLD);
    MPI_Recv(&(pMemory[currentIT][0]),1,MPI_DOUBLE,nProcs-1,0,MPI_COMM_WORLD, status);
    MPI_Send(&(pMemory[currentIT][nPoints-2]),1,MPI_DOUBLE,id+1,0,MPI_COMM_WORLD);
    MPI_Recv(&(pMemory[currentIT][nPoints-1]),1,MPI_DOUBLE,id+1,0,MPI_COMM_WORLD, status);
  } else if (id == nProcs-1) {
    MPI_Recv(&(pMemory[currentIT][nPoints-1]),1,MPI_DOUBLE,0,0,MPI_COMM_WORLD, status);
    MPI_Send(&(pMemory[currentIT][nPoints-2]),1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    MPI_Recv(&(pMemory[currentIT][0]),1,MPI_DOUBLE,id-1,0,MPI_COMM_WORLD, status);
    MPI_Send(&(pMemory[currentIT][1]),1,MPI_DOUBLE,id-1,0,MPI_COMM_WORLD);
  } else {
    MPI_Recv(&(pMemory[currentIT][0]),1,MPI_DOUBLE,id-1,0,MPI_COMM_WORLD, status);
    MPI_Send(&(pMemory[currentIT][1]),1,MPI_DOUBLE,id-1,0,MPI_COMM_WORLD);
    MPI_Send(&(pMemory[currentIT][nPoints-2]),1,MPI_DOUBLE,id+1,0,MPI_COMM_WORLD);
    MPI_Recv(&(pMemory[currentIT][nPoints-1]),1,MPI_DOUBLE,id+1,0,MPI_COMM_WORLD, status);
  }
}

double **finalizeResults(double **pMemory, Processor1D *pProc, const int &nTime, const int &nProcs, MPI_Status *status){
  double **returnMemory = allocateMemory(nTime, pProc->nPointsTot);
  for (int t=0; t<nTime; t++) {
    for(int i=0; i<pProc->nPointsProc-2; i++) {
      returnMemory[t][i] = pMemory[t][i];
    }
  }
  freeMemory(pMemory, nTime);
  for(int sender = 1; sender<nProcs; sender++) {
    for (int t=0; t<nTime; t++) {
      for(int i=0; i<pProc->nPointsProc-2; i++) {
        MPI_Recv((&returnMemory[t][sender*(pProc->nPointsProc-2)+i]),1,MPI_DOUBLE,sender,0,MPI_COMM_WORLD, status);
      }
    }
  }
  return returnMemory;
}
