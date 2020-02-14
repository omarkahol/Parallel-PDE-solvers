#include "functions.h"
#include <omp.h>

int main(int arc, char *argv[]) {
  const int nProcX = std::atoi(argv[1]);
  const int nProcY = std::atoi(argv[2]);
  const int nXproc = std::atoi(argv[3]);
  const int nYproc = std::atoi(argv[4]);
  const int nTime = std::atoi(argv[5]);
  const int showResults = (bool) std::atoi(argv[6]);

  const int nXtot = nXproc*nProcX-2*nProcX;
  const int nYtot = nYproc*nProcY-2*nProcY;
  const double dt = 0.1*(length/(nXtot-1))*(length/(nYtot-1));
  
  omp_set_num_threads(nProcX*nProcY);
  std::vector<Processor> processorPool;
  createProcessors(nProcX, nProcY, nXproc, nYproc, processorPool);

  double ***uGrid = allocateMemory(nTime, nXtot, nYtot);
  double ***vGrid = allocateMemory(nTime, nXtot, nYtot);
  
  initGrid(uGrid, nXtot, nYtot);
  initGrid(vGrid, nXtot, nYtot);

  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
  {
    int ID = omp_get_thread_num();
    for(int t=0; t<nTime-1; t++) {
      evolvePDE(uGrid, vGrid, t, dt, &(processorPool[ID]));
      #pragma omp barrier
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Integrazione terminata dopo " << duration.count() << " microsecondi." << std::endl;

  writeFile(uGrid, vGrid, nXtot, nYtot, nTime, "resultsFile.csv");

  freeMemory(uGrid, nTime, nYtot);
  freeMemory(vGrid, nTime, nYtot);

  if (showResults) {
    std::system("python3 postProcessor.py");
  }

}