#include <fstream>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iomanip>
#include <algorithm>

#define NX 501
#define NY_MAX 501
#define DX 1.0/double(NX-1)
#define NT 1000
#define CX 10.0
#define CY 10.0
#define K 0.1
#define SIGMA 0.1

//DISCRETIZE THE OUTER FOR LOOP IN THE Y DIRECTION

int main(int argc, char **argv) {
  bool show = (bool) std::atoi(argv[1]);
  
  //INITIALIZING MPI
  int ierr, id, numProcs;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &id);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  int idPrev = (id==0)?numProcs-1:id-1;
  int idAfter = (id==numProcs-1)?0:id+1;

  const int numYproc = 2+(NY_MAX / numProcs);
  const int NY = numYproc*numProcs - 2*numProcs;
  const double DY = 1.0/(NY-1);
  const double DT = SIGMA*DX*DY;

  double *solution = new double[NX*numYproc*NT];
  double x, y;
  for (int i = 0; i < numYproc; i++) {
    y = id*(numYproc-2)*DY + i*DY;
    for (int j = 0; j < NX; j++) {
      x=j*DX;
      if (std::sqrt(std::pow(x-0.5,2) + std::pow(y-0.5,2))<0.2){
        solution[j + NX * (i + numYproc * 0)] = 5.0;
      } else {
        solution[j + NX * (i + numYproc * 0)] = 1.0;
      }
    }
  }


  int jBefore, jAfter;
  double dUdx, dUdy, d2Udx2, d2Udy2;
  double sendBefore[NX], receiveBefore[NX], sendAfter[NX], receiveAfter[NX];
  MPI_Status status[4];
  MPI_Request request[4];
  double start = MPI_Wtime();
  for (int t=0; t<NT-1; t++) {
    
    //SOLVE THE PDE IN THE DOMAIN
    for(int i=1; i<numYproc-1; i++) {
      for(int j=0; j<NX; j++) {
        jBefore = (j==0)?NX-2:j-1;
        jAfter = (j==NX-1)?1:j+1;

        dUdx = (solution[jAfter + NX * (i + numYproc * t)] - solution[jBefore + NX * (i + numYproc * t)])/(2*DX);
        dUdy = (solution[j + NX * (i+1 + numYproc * t)] - solution[j + NX * (i-1 + numYproc * t)])/(2*DY);
        d2Udx2 = (solution[jAfter + NX * (i + numYproc * t)] - 2*solution[j + NX * (i + numYproc * t)]+ solution[jBefore + NX * (i + numYproc * t)])/(DX*DX);
        d2Udy2 = (solution[j + NX * (i+1 + numYproc * t)] - 2*solution[j + NX * (i + numYproc * t)]+ solution[j + NX * (i-1 + numYproc * t)])/(DY*DY);

        solution[j + NX * (i + numYproc * (t+1))]=solution[j + NX * (i + numYproc * t)] + DT*(-CX*dUdx-CY*dUdy+K*(d2Udx2+d2Udy2));
      }
    }

    //COMMUNICATE RESULTS
    for(int j=0; j<NX; j++){
      sendBefore[j] = solution[j + NX * (1 + numYproc * (t+1))];
      sendAfter[j] = solution[j + NX * (numYproc-2 + numYproc * (t+1))];
    }

    MPI_Isend(sendBefore, NX, MPI_DOUBLE, idPrev, 1, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(sendAfter, NX, MPI_DOUBLE, idAfter, 0, MPI_COMM_WORLD,&request[1]);
    MPI_Irecv(receiveAfter, NX, MPI_DOUBLE,idAfter,1,MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(receiveBefore, NX, MPI_DOUBLE,idPrev,0,MPI_COMM_WORLD, &request[3]);

    MPI_Waitall(4, request, status);

    for(int j=0; j<NX; j++){
      solution[j + NX * (0 + numYproc * (t+1))] = receiveBefore[j];
      solution[j + NX * (numYproc-1 + numYproc * (t+1))] = receiveAfter[j];
    }
  }
  double end = MPI_Wtime();

  if (id==0) {
    std::cout << "numProcs," << numProcs << std::endl;
    std::cout << std::setprecision(5)<<"elapsedTime," << end-start << std::endl;
    std::cout << std::endl;
  }

  if (show && id==0) {
    std::fstream f;
    f.open("iterations.csv",std::ios::out);

    for(int t=0; t<NT; t++){
      for(int i=0; i<numYproc; i++){
        for(int j=0; j<NX; j++){
          f << solution[j + NX * (i + numYproc * t)] << ((j==NX-1)?"":",");
        }
        f << ((i==numYproc-1)?"":";");
      }
      f<<std::endl;
    }
    f.close();
    std::system("python3 postProcessor.py");
  }

  delete [] solution;
  ierr = MPI_Finalize();
  return 0;
}