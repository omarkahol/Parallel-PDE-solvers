#include "functions.h"

int main(int argc, char *argv[]) {
  const int nX = std::atoi(argv[1]);
  const int nY = std::atoi(argv[2]);
  const int nTime = std::atoi(argv[3]);
  const int showResults = (bool) std::atoi(argv[4]);
  const double dx = length/(nX-1);
  const double dy = length/(nY-1);
  const double dt = dx*dy;
  
  double ***uGrid = allocateMemory(nTime, nX, nY);
  double ***vGrid = allocateMemory(nTime, nX, nY);
  
  initGrid(uGrid, nX, nY);
  initGrid(vGrid, nX, nY);

  Processor serialProcessor{0,0,nX+2, nY+2, dx, dy, nX, nY};
  auto start = std::chrono::high_resolution_clock::now();
  for (int t=0; t<nTime-1; t++) {
    evolvePDE(uGrid, vGrid, t, dt, &serialProcessor);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Integrazione terminata dopo " << duration.count() << " microsecondi." << std::endl;

  writeFile(uGrid, vGrid, nX, nY, nTime, "resultsFile.csv");
  
  freeMemory(uGrid, nTime, nY);
  freeMemory(vGrid, nTime, nY);

  if (showResults) {
    std::system("python3 postProcessor.py");
  }
  return 0;
}