#include <fstream>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <algorithm>

//COMPILER MACROS
#define NX 10001 //DOMAIN DISCRETIZATION
#define DX 1.0/((double)NX+1.0) //STEP IN SPACE
#define DT 0.1*DX*DX //STEP IN TIME
#define NT 1000 //MAX TIME ITERATIONS
#define C 10.0 //CONVECTION VELOCITY
#define K 0.1 //DIFFUSION COEFFICIENT

int main(int argc, char **argv) {
  omp_set_num_threads(std::atoi(argv[1]));  //SET THE REQUIRED NUMBER OF THREADS
  bool show = (bool) std::atoi(argv[2]); //CALL POSTPROCESSOR

  //INTIALING MEMORY --> USING A FLAT DOUBLE*
  double *solution = new double[2*NX*NT];

  for (int i=0; i<NX; i++){
    solution[i] = std::sin(i*DX*2*M_PI); //INITIAL CONDITION
  }

  int numThreads, i, t, iBefore, iAfter;
  double energy[NT]{0.0}; //ENERGY of the solution --> e(t)= integral from 0 to 1 of ||u(x,t)||^2 dx
  double error[NT]{0.0};

  //SOLVE THE PDE ON A RING
  double start = omp_get_wtime();
  double dUdx, d2Udx2, analyticSolution;
  #pragma omp parallel default(none) shared(solution, energy, numThreads, error) private(iBefore, iAfter, i, t, dUdx, d2Udx2, analyticSolution)
  {
    #pragma omp master
    numThreads = omp_get_num_threads();

    for(t=0; t<NT-1; t++){

      #pragma omp for schedule(static) reduction(+:energy[t]) reduction(+:error[t])
      for(i=0; i<NX; i++){
        iBefore = (i==0)?NX-2:i-1;
        iAfter = (i==NX-1)?1:i+1;
        dUdx = (solution[2*t*NX+iAfter]-solution[2*t*NX+iBefore])/(2*DX);
        d2Udx2 = (solution[2*t*NX+iAfter]-2*solution[2*t*NX+i]+solution[2*t*NX+iBefore])/(DX*DX);
        solution[2*(t+1)*NX+i] = solution[2*t*NX+i] + DT*(-C*dUdx+K*d2Udx2);
        energy[t] += DX*solution[2*t*NX+i]*solution[2*t*NX+i];
        analyticSolution = std::sin(2*M_PI*(i*DX-C*t*DT))*std::exp(-t*DT*K*std::pow((2*M_PI),2));
        error[t] += std::pow(solution[2*t*NX+i]-analyticSolution,2);
      }
    }
  }
  double end = omp_get_wtime();
  
  std::cout << "numPoints," <<NX<<std::endl;
  std::cout << "numProcs," << numThreads << std::endl;
  std::cout << std::setprecision(5)<<"dx," << DX << std::endl;
  std::cout << "numTime," << NT<<std::endl;
  std::cout << std::setprecision(5) <<"dt,"<<DT<<std::endl;
  std::cout << std::setprecision(5)<<"elapsedTime," << end-start << std::endl;
  std::cout << std::setprecision(10)<< "maxError," << std::sqrt(*std::max_element(error, error+NT)) << std::endl;
  std::cout << std::endl;

  if(show) {
    std::fstream f;
    f.open("iterations.csv",std::ios::out);
    for(t=0; t<NT; t++) {
      for(i=0; i<NX; i++){
        f << solution[2*t*NX+i];
        if (i != NX-1){
          f << ",";
        }
      }
      f<<std::endl;
    }
    f.close();

    f.open("energy_error.csv",std::ios::out);
    for(int t=0; t<NT; t++){
      f << energy[t] << ","<<error[t] << std::endl;
    }
    f.close();

    std::system("python3 postProcessor.py");
  }

  delete [] solution;
  return 0;
}