#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>

const int N_POINTS = 72;
const double LENGTH = 1;
const double DX = LENGTH/(N_POINTS-1);
const double SIGMA = 0.1;
const double DT = SIGMA*DX;
const int ITERATIONS_TIME=1000;
const double C = 5;
const double K = .001;

std::vector<double> getGrid(){
    std::vector<double> grid(N_POINTS);
    double start = 0;
    for (int i=0; i<N_POINTS; i++){
        grid.at(i)=start;
        start += DX;
    }
    return grid;
}

std::vector<double> initialCondition(const std::vector<double>& grid) {
    std::vector<double> initialCond(N_POINTS);
    for (int i=0; i<N_POINTS; i++){
        initialCond.at(i)=std::sin(2*M_PI*grid.at(i));
    }
    return initialCond;
}

void writeData (const std::vector<double>* storage, std::string&& name){
    std::fstream f;
    f.open(name,std::ios::out);

    for (int i=0; i<ITERATIONS_TIME; i++) {
        for (int j=0; j<N_POINTS; j++) {
            if (j != N_POINTS-1)
                f << storage[i].at(j) << ",";
            else
                f << storage[i].at(j)<<std::endl;
        }
    }
    f.close();
}

int main() {

    // GENERARE UNA GRIGLIA E LA CONDIZIONE INIZIALE
    auto grid=getGrid();
    auto uOld = initialCondition(grid);

    auto uNew = initialCondition(grid);
    int it = 0;

    //ALLOCATE SOME MEMORY
    auto* storage = new std::vector<double>[ITERATIONS_TIME];
    auto start = std::chrono::high_resolution_clock::now();
    while (it++<ITERATIONS_TIME) {
        for (int i = 1; i<N_POINTS-1; i++){
            
            // EVOLVO LA FUNZIONE ATTRAVERSO Ut + cUx = kUxx
            uNew.at(i) = uOld.at(i) + DT*(  -C*((uOld.at(i)-uOld.at(i-1))/DX) + K*(uOld.at(i+1)-2*uOld.at(i)+uOld.at(i-1))/std::pow(DX,2)  );
        }

        //CONDIZIONE AL CONTORNO PERIODICHE Ux(0,t) = Ut(0,t)
        uNew.at(N_POINTS-1) = 0.5*(uNew.at(1)+uNew.at(N_POINTS-2));

        //CONDIZIONE AL CONTORNO SULLA FUNZIONE U(0,t) = U(1,t)
        uNew.at(0) = uNew.at(N_POINTS-1);

        storage[it-1] = uNew;

        uOld = uNew;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Elapsed Time: " << duration.count()<< " microseconds" << std::endl;

    //DUMP TO FILE
    writeData(storage,"dataFile.csv");

    //POST PROCESSING
    std::system("python3 postProcessorDiffusionConvection.py");
    delete[] storage;

    return 0;
}
