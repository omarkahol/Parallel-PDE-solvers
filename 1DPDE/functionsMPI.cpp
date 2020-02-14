#include "functionsMPI.h"

void initProcessGrid(double **pMemory,const int &ID,const int &nPROCS,const int &nPOINTS, const double& DX, const double &LENGTH) {
  double phase = ID*(nPOINTS-2)*DX;
  for (int i=0; i<nPOINTS; i++){
        pMemory[0][i] = std::sin(2*M_PI*(phase+i*DX)/LENGTH); //RIEMPIO LA GRIGLIA
  }
  //CONDIZIONE AL CONTORNO PERIODICA, L'ULTIMO PROCESSO LAVORA ANCHE SUI DUE NODI INIZIALI
  if (ID == nPROCS -1) { 
    pMemory[0][nPOINTS-2] = std::sin(2*M_PI*0/LENGTH);
    pMemory[0][nPOINTS-1] = std::sin(2*M_PI*DX/LENGTH);
  }
}

void mpiCommunicateResults(double **pMemory, const int &currentIT, const int &nPOINTS, const int &ID, const int &nPROCS,MPI_Status *status) {
  if (ID==0) {
    MPI_Send(&(pMemory[currentIT][1]),1,MPI_DOUBLE,nPROCS-1,0,MPI_COMM_WORLD);
    MPI_Recv(&(pMemory[currentIT][0]),1,MPI_DOUBLE,nPROCS-1,0,MPI_COMM_WORLD, status);
    MPI_Send(&(pMemory[currentIT][nPOINTS-2]),1,MPI_DOUBLE,ID+1,0,MPI_COMM_WORLD);
    MPI_Recv(&(pMemory[currentIT][nPOINTS-1]),1,MPI_DOUBLE,ID+1,0,MPI_COMM_WORLD, status);
  } else if (ID == nPROCS-1) {
    MPI_Recv(&(pMemory[currentIT][nPOINTS-1]),1,MPI_DOUBLE,0,0,MPI_COMM_WORLD, status);
    MPI_Send(&(pMemory[currentIT][nPOINTS-2]),1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    MPI_Recv(&(pMemory[currentIT][0]),1,MPI_DOUBLE,ID-1,0,MPI_COMM_WORLD, status);
    MPI_Send(&(pMemory[currentIT][1]),1,MPI_DOUBLE,ID-1,0,MPI_COMM_WORLD);
  } else {
    MPI_Recv(&(pMemory[currentIT][0]),1,MPI_DOUBLE,ID-1,0,MPI_COMM_WORLD, status);
    MPI_Send(&(pMemory[currentIT][1]),1,MPI_DOUBLE,ID-1,0,MPI_COMM_WORLD);
    MPI_Send(&(pMemory[currentIT][nPOINTS-2]),1,MPI_DOUBLE,ID+1,0,MPI_COMM_WORLD);
    MPI_Recv(&(pMemory[currentIT][nPOINTS-1]),1,MPI_DOUBLE,ID+1,0,MPI_COMM_WORLD, status);
  }
}
