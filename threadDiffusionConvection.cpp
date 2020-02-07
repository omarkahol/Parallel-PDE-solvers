#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <fstream>
#include <cstring>

//VARIABILI DEL SISTEMA
const int NTHREADS = 10;
const int NPOINTS_THREAD = 20;
const int NPOINTS = NTHREADS*NPOINTS_THREAD - 2*NTHREADS;
const double LENGTH = 1;
const double DX = LENGTH/(NPOINTS-1);
const int ITERATIONS_TIME=1000;
const double C = 5;
const double K = 0.001;
const double SIGMA = 0.1;
const double DT = SIGMA*DX;

//VARIABILI PER I THREADS
std::mutex lockCounter;
int counterThread = 0;

std::condition_variable cv;
std::mutex executionMutex[NTHREADS];
bool canExecute[NTHREADS]{true};
std::mutex mainThreadMutex;

void evolveThread(int ID, int start, int end, double* uNew, const double* uOld) {

  std::unique_lock<std::mutex> ul(executionMutex[ID]);

  for(int t = 0; t<ITERATIONS_TIME; t++) {

    cv.wait(ul, [ID](){return canExecute[ID];});

    for (int i=start; i<end; i++) {
      uNew[(i)%NPOINTS] = uOld[(i)%NPOINTS] + DT*(  -C*((uOld[(i)%NPOINTS]-uOld[(i-1)%NPOINTS])/DX) + K*(uOld[(i+1)%NPOINTS]-2*uOld[i%NPOINTS]+uOld[(i-1)%NPOINTS])/std::pow(DX,2)  );
    }

    lockCounter.lock();
    counterThread += 1;
    canExecute[ID] = false;
    lockCounter.unlock();
  }
}

void dumperThread(double* uOld, double* uNew) {
  for (int t=0; t<ITERATIONS_TIME; t++) {

    std::unique_lock<std::mutex> ul(mainThreadMutex);
    cv.wait(ul, [](){return counterThread == NTHREADS;});

    counterThread = 0;

    for (int i=0; i<NPOINTS; i++) {
      uOld[i] = uNew[i];
      std::cout<<uNew[i]<<";";
    }

    std::cout << std::endl;
    for (int i=0; i<NTHREADS; i++) {
      canExecute[i] = true;
    }

  }
}

int main() {

  //INIZIALIZZO LE VARIABILI
  double grid[NPOINTS];
  double uOld[NPOINTS];
  double uNew[NPOINTS];
  for (int i=0; i<NPOINTS; i++) {
    grid[i] = i*DX;
    uOld[i]=std::sin(2*M_PI*grid[i]/LENGTH);
    uNew[i]={0};
  }

  std::thread dump(dumperThread, uOld, uNew);
  dump.join();

  for (int i=0; i<NTHREADS; i++) {
    std::thread t(evolveThread, i, 0, 5, uNew, uOld);
    t.join();
  }


}