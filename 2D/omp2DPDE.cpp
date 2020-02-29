#include <fstream>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <vector>

#define NX 11
#define NY 11
#define NT 500
#define DX 1.0/double(NX-1)
#define DY 1.0/double(NY-1)
#define DT 0.1*DX*DY
#define CX 10.0
#define CY 10.0
#define K 0.1

int main(int argc, char **argv) {

  omp_set_num_threads(std::atoi(argv[1]));
  bool show = (bool) std::atoi(argv[2]);

  //INIT MEMORY
  std::vector<double> solution(NT*NX*NY);
  double x, y;
  int i, j, t, numThreads;
  for (i = 0; i < NY; i++) {
    y = i*DY;
    for (j = 0; j < NX; j++) {
      x=j*DX;
      if (std::sqrt(std::pow(x-0.5,2) + std::pow(y-0.5,2))<0.2){
        solution[j + NX * (i + NY * 0)] = 5.0;
      } else {
        solution[j + NX * (i + NY * 0)] = 1.0;
      }
    }
  }

  //SOLVE PDE
  double dUdx, dUdy, d2Udx2, d2Udy2;
  int iAfter, iBefore, jAfter, jBefore;
  double wtime = omp_get_wtime();
  #pragma omp parallel default(none) shared(solution, numThreads) private(i,j,t,dUdx, dUdy, d2Udx2, d2Udy2,iAfter, iBefore, jAfter, jBefore)
  {
    #pragma omp master
    numThreads = omp_get_num_threads();

    for (t=0; t<NT-1; t++) {

      #pragma omp for schedule(static)
      for (i = 0; i < NY; i++) {
        for (j = 0; j < NX; j++) {
          iBefore = (i==0)?NY-2:i-1;
          iAfter = (i==NY-1)?1:i+1; 
          jBefore = (j==0)?NX-2:j-1;
          jAfter = (j==NX-1)?1:j+1;
          dUdx = (solution[jAfter + NX * (i + NY * t)] - solution[jBefore + NX * (i + NY * t)])/(2*DX);
          dUdy = (solution[j + NX * (iAfter + NY * t)] - solution[j + NX * (iBefore + NY * t)])/(2*DY);
          d2Udx2 = (solution[jAfter + NX * (i + NY * t)] - 2*solution[j + NX * (i + NY * t)]+ solution[jBefore + NX * (i + NY * t)])/(DX*DX);
          d2Udy2 = (solution[j + NX * (iAfter + NY * t)] - 2*solution[j + NX * (i + NY * t)]+ solution[j + NX * (iBefore + NY * t)])/(DY*DY);

          solution[j + NX * (i + NY * (t+1))]=solution[j + NX * (i + NY * t)] + DT*(-CX*dUdx-CY*dUdy+K*(d2Udx2+d2Udy2));
          }
      }
    }
  }
  wtime = omp_get_wtime()-wtime;
  std::cout << "numThreads: " <<numThreads << ". Elapsed Time: "<<wtime*1000 << std::endl;

  if (show) {
    std::fstream f;
    f.open("iterations.csv",std::ios::out);

    for(int t=0; t<NT; t++){
      for(int i=0; i<NY; i++){
        for(int j=0; j<NX; j++){
          f << solution[j + NX * (i + NY * t)] << ((j==NX-1)?"":",");
        }
        f << ((i==NY-1)?"":";");
      }
      f<<std::endl;
    }
    f.close();
    std::system("python3 postProcessor.py");
  }
  return 0;
}
