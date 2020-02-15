#include "functions.h"

double ***allocateMemory(const int& nTime, const int &nX, const int &nY){
  double ***pMemory = new double**[nTime];
  for (int t=0; t<nTime; t++) {
    pMemory[t] = new double*[nY];
    for (int i=0; i<nY; i++) {
      pMemory[t][i] = new double[nX];
    }
  }
  return pMemory;
}
void freeMemory (double ***pMemory, const int &nTime, const int &nY) {
  for (int t=0; t<nTime; t++) {
    for (int i=0; i<nY; i++) {
      delete[] pMemory[t][i];
    }
    delete[] pMemory[t];
  }
  delete[] pMemory;
}

void initGrid(double ***pMemory, const int &nX, const int &nY) {
  double dx = length/(nX-1);
  double dy = length/(nY-1);
  for (int i=0; i<nY; i++) {
    for (int j=0; j<nX; j++) {
      if (std::sqrt(std::pow(i*dy-0.5,2)+std::pow(j*dx-0.5,2))<0.2) {
      pMemory[0][i][j] = 1;
      } else {
        pMemory[0][i][j] = 0;
      }
    }
  }
}

void coolPrintGrid(double ***pGrid, const int &currentT, const int &nX, const int &nY) {
  for (int i=0; i<nY; i++) {
    for (int j=0; j<nX; j++) {
      std::cout << std::scientific <<std::setprecision(5)<< pGrid[currentT][i][j] << " " ;
    }
    std::cout << std::endl;
  }
}

void evolvePDE(double ***uGrid, double ***vGrid, const int &t, const double &dt, Processor *pProc) {
  for (int i=pProc->startY+1; i<pProc->startY+pProc->nPointsY;i++) {
    for (int j=pProc->startX+1; j<pProc->startX+pProc->nPointsX;j++) {
      int iC = i%pProc->nPointsTotY;
      int iB = std::abs(i-1)%pProc->nPointsTotY;
      int iA = (i+1)%pProc->nPointsTotY;
      int jC = j%pProc->nPointsTotX;
      int jB = std::abs(j-1)%pProc->nPointsTotX;
      int jA = (j+1)%pProc->nPointsTotX;

      double dUdx = (uGrid[t][iC][jA] - uGrid[t][iC][jB])/(2*pProc->dx);
      double dUdy = (uGrid[t][iA][jC] - uGrid[t][iB][jC])/(2*pProc->dy);
      double dVdx = (vGrid[t][iC][jA] - vGrid[t][iC][jB])/(2*pProc->dx);
      double dVdy = (vGrid[t][iA][jC] - vGrid[t][iB][jC])/(2*pProc->dy);  
      double d2Udx2 = (uGrid[t][iC][jA] - 2*uGrid[t][iC][jC] + uGrid[t][iC][jB])/std::pow(pProc->dx,2);
      double d2Udy2 = (uGrid[t][iA][jC] - 2*uGrid[t][iC][jC] + uGrid[t][iB][jC])/std::pow(pProc->dy,2);
      double d2Vdx2 = (vGrid[t][iC][jA] - 2*vGrid[t][iC][jC] + vGrid[t][iC][jB])/std::pow(pProc->dx,2);
      double d2Vdy2 = (vGrid[t][iA][jC] - 2*vGrid[t][iC][jC] + vGrid[t][iB][jC])/std::pow(pProc->dy,2);

      //double cX = uGrid[t][iC][jC];
      //double cY = vGrid[t][iC][jC];

      uGrid[t+1][iC][jC] = uGrid[t][iC][jC] + dt*(-cX*dUdx - cY*dUdy + k*(d2Udx2 + d2Udy2));
      vGrid[t+1][iC][jC] = vGrid[t][iC][jC] + dt*(-cX*dVdx - cY*dVdy + k*(d2Vdx2 + d2Vdy2));
    }
  }
}

void writeFile(double ***uGrid, double ***vGrid, const int &nX, const int &nY, const int &nTime, const char *fileName) {
  std::fstream f;
  f.open(fileName, std::ios::out);

  for (int t=0; t<nTime; t++) {
    for (int i=0; i<nY; i++) {
      for (int j=0; j<nX; j++) {
        f << std::sqrt(std::pow(uGrid[t][i][j],2)+std::pow(vGrid[t][i][j],2));
        if (j != nX-1) {
          f <<",";
        }
      }
      if (i != nY -1){
        f << ";";
      }
    }
    f << std::endl;
  }

  f.close();
}

void createProcessors(const int &nProcX, const int &nProcY, const int &nXproc, const int &nYproc, std::vector<Processor> &procPool) {
  int nXtot = nXproc*nProcX-2*nProcX;
  int nYtot = nYproc*nProcY-2*nProcY;
  double dx = length/(nXtot-1);
  double dy = length/(nYtot-1);
  for (int i=0; i<nProcY; i++) {
    for(int j=0; j<nProcX; j++) {
      int xstart = j*(nXproc-2);
      int ystart = i*(nYproc-2);
      procPool.push_back(Processor{xstart,ystart,nXproc, nYproc, dx, dy, nXtot, nYtot});
    }
  }
}