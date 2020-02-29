#include <fstream>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iomanip>
#include <algorithm>

#define NXMAX 10001
#define NT 1000
#define C 10.0
#define K 0.1
#define SIGMA 0.1

int main(int argc, char **argv) {

  bool show = (bool) std::atoi(argv[1]);
  
  //INITIALIZING MPI
  int ierr, id, numProcs;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &id);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  int idPrev = (id==0)?numProcs-1:id-1;
  int idAfter = (id==numProcs-1)?0:id+1;

  const int nPoints = 2+(NXMAX / numProcs);
  const int nPointsTot = nPoints*numProcs - 2*numProcs;
  const double DX = 1.0/(nPointsTot-1);
  const double DT = SIGMA*DX*DX;

  //PREPARE GRID
  double *solution = new double[nPoints*NT];
  double error[NT], x, phase, analyticSolution, dUdx, d2Udx2;

  MPI_Status status[4];
  MPI_Request request[4];

  phase = id*(nPoints-2)*DX;
  for(int i=0; i<nPoints; i++) {
    x = phase + i*DX;
    solution[i] = std::sin(2*M_PI*x);
  }
  if (id ==numProcs-1) {
    solution[nPoints-2] = std::sin(2*M_PI*0);
    solution[nPoints-1] = std::sin(2*M_PI*DX);
  }

  double start = MPI_Wtime();
  for(int t=0; t<NT-1; t++){
    for(int i=1; i<nPoints-1; i++){
      dUdx = (solution[t*nPoints+i+1]-solution[t*nPoints+i-1])/(2*DX);
      d2Udx2 = (solution[t*nPoints+i+1]-2*solution[t*nPoints+i]+solution[t*nPoints+i-1])/(DX*DX);
      solution[(t+1)*nPoints+i] = solution[t*nPoints+i] + DT*(-C*dUdx+K*d2Udx2);
    }

    MPI_Isend(&solution[(t+1)*nPoints+1], 1, MPI_DOUBLE, idPrev, 1, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(&solution[(t+1)*nPoints+nPoints-2], 1, MPI_DOUBLE, idAfter, 0, MPI_COMM_WORLD,&request[1]);
    MPI_Irecv(&solution[(t+1)*nPoints+nPoints-1], 1, MPI_DOUBLE,idAfter,1,MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(&solution[(t+1)*nPoints], 1, MPI_DOUBLE,idPrev,0,MPI_COMM_WORLD, &request[3]);

    for(int i=0; i<nPoints-2; i++) {
      analyticSolution = std::sin(2*M_PI*(phase + i*DX-C*t*DT))*std::exp(-t*K*DT*std::pow((2*M_PI),2));
      error[t] += std::pow(solution[t*nPoints+i]-analyticSolution,2);
    }
    error[t] = std::sqrt(error[t]);
    MPI_Waitall(4,request,status);
  }
  double end = MPI_Wtime();

  //PRINT RESULTS
  double elapsedTime=end-start;
  double meanTime=0.0;
  MPI_Reduce(&elapsedTime,&meanTime,1,MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);

  double meanError[NT];
  MPI_Reduce(error,meanError,NT,MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
  
  if (id==0) {
    std::cout << "numPoints," <<nPointsTot<<std::endl;
    std::cout << "numProcs," << numProcs << std::endl;
    std::cout << std::setprecision(5)<<"dx," << DX << std::endl;
    std::cout << "numTime," << NT<<std::endl;
    std::cout << std::setprecision(5) <<"dt,"<<DT<<std::endl;
    std::cout << std::setprecision(5)<<"elapsedTime," << meanTime/numProcs << std::endl;
    std::cout << "maxError," << (*std::max_element(meanError, meanError+NT))/numProcs << std::endl;
    std::cout << std::endl;
  }
  if (show){
    if(id==numProcs-1) {
    std::fstream f;
    f.open("iterations.csv",std::ios::out);
    for(int t=0; t<NT; t++) {
      for(int i=0; i<nPoints; i++){
        f << solution[t*nPoints+i];
        if (i != nPoints-1){
          f << ",";
        }
      }
      f<<std::endl;
    }
    f.close();
    std::system("python3 postProcessor.py");
    }
  }

  delete [] solution;
  ierr = MPI_Finalize();
  return ierr;
}
