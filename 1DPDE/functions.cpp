#include "functions.h"

void writeData(double ***pMemory, const int &nTime, const int &nPoints, const char *fileName) {
  std::fstream f;
  f.open(fileName,std::ios::out);

  for (int i=0; i<nTime; i++) {
    for (int j=0; j<nPoints; j++) {
      if (j != nPoints-1)
        f << pMemory[i][j][0] << ",";
      else
        f << pMemory[i][j][0]<<std::endl;
      }
    }
    f.close();
}

double ***allocateMemory(const int &nTime, const int &nPoints){
  double ***pMemory = new double**[nTime];
  for (int t=0; t<nTime; t++){
    pMemory[t] = new double*[nPoints];
    for (int i=0; i<nPoints; i++) {
      pMemory[t][i]= new double[pad];
    }
  }
  return pMemory;
}

void freeMemory(double ***pMemory, const int &nTime, const int &nPoints){
  for (int t=0; t<nTime; t++){
    for (int i=0; i<nPoints; i++){
      delete[] pMemory[t][i];
    }
    delete[] pMemory[t];
  }
  delete[] pMemory;
}

void initGrid(double ***pMemory, const int &nPoints, const double& dx, const double &length){
  for (int i=0; i<nPoints; i++) {
    pMemory[0][i][0] = std::sin(2*M_PI*i*dx/length);
  }
}

void evolvePDE(double ***pMemory, const int &t, Processor1D *pProc) {
  int nPoints = pProc -> nPointsTot;
  int end =pProc->start + pProc->nPointsProc;
  for (int i=pProc->start +1 ; i<end; i++) {
        pMemory[t+1][(i)%nPoints][0] = pMemory[t][(i)%nPoints][0]
          + pProc->dt*(  -c*((pMemory[t][(i+1)%nPoints][0]-pMemory[t][std::abs(i-1)%nPoints][0])/(2*pProc->dx))
            + k*(pMemory[t][(i+1)%nPoints][0]-2*pMemory[t][i%nPoints][0]
              + pMemory[t][std::abs(i-1)%nPoints][0])/std::pow(pProc->dx,2)  );
      }
}

void printResults(const char *fileName, const int &nPoints, const int &nThreads,const double &dt, const double &dx,const double &elapsedTime){
  std::fstream f;
  f.open(fileName, std::ios::out);
  f << "Numero di punti," << nPoints << std::endl;
  f << "Numero di threads, " << nThreads << std::endl;
  f << "dt, " << dt << std::endl;
  f << "dx, " << dx << std::endl;
  f << "Tempo richiesto microsecondi, " << elapsedTime << std::endl;
  f.close();
}