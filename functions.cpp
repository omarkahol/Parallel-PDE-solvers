#include "functions.h"

void writeData(double **data, const int &nTIME, const int &nPOINTS, const char *fileName) {
  std::fstream f;
  f.open(fileName,std::ios::out);

  for (int i=0; i<nTIME; i++) {
    for (int j=0; j<nPOINTS; j++) {
      if (j != nPOINTS-1)
        f << data[i][j] << ",";
      else
        f << data[i][j]<<std::endl;
      }
    }
    f.close();
}

double **allocateMemory(const int &nTIME, const int &nPOINTS){
  double **pMemory = new double*[nTIME];
  for (int i=0; i<nTIME; i++){
    pMemory[i] = new double[nPOINTS];
  }
  return pMemory;
}

void freeMemory(double **pMemory, const int &nTIME){
  for (int t=0; t<nTIME; t++){
    delete[] pMemory[t];
  }
  delete[] pMemory;
}

void initGrid(double **pMemory, double (*pFunction)(int), const int &nPOINTS) {
  for (int i=0; i<nPOINTS; i++) {
    pMemory[0][i] = pFunction(i);
  }
}

void evolvePDE(double **pMemory, const int &t, const int &nPOINTS, const int &start, const int &end, const double &DX, const double &DT) {
  for (int i=start+1; i<end-1; i++) {
        pMemory[t+1][(i)%nPOINTS] = pMemory[t][(i)%nPOINTS]
          + DT*(  -C*((pMemory[t][(i)%nPOINTS]-pMemory[t][(i-1)%nPOINTS])/DX)
            + K*(pMemory[t][(i+1)%nPOINTS]-2*pMemory[t][i%nPOINTS]
              + pMemory[t][(i-1)%nPOINTS])/std::pow(DX,2)  );
      }
}