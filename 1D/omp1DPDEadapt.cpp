#include <iostream>
#include <set>
#include <cmath>
#include <fstream>
#include <omp.h>

#define NX 25
#define NT 10000
#define C 10.0
#define K 0.01
#define DX 1.0/NX
#define DXMIN 0.4*DX
#define DXMAX 1.0*DX
#define SIGMA 1e-4
#define BIASREFINE 1.5
#define BIASCOARSEN 1.5

#pragma pack(push, 1)
struct Node {
    double x{0.0};
    double y{0.0};
    double dudx{0.0};
    double d2udx2{0.0};
    double yExact{0.0};
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Cell {
    double x{0.0};
    mutable double y{0.0};

    Cell(const double &x, const double &y): x(x), y(y){};
    Cell() = default;

    bool operator<(const Cell &other) const {
        return x < other.x;
    }
};
#pragma pack(pop)

int main(int argc, char **argv) {

    omp_set_num_threads(std::atoi(argv[1]));

    Node *uOld = new Node[NX];
    Node *uNew = new Node[NX];
    std::set<Cell> mesh;

    double *error = new double[NT];
    double *energy = new double[NT];
    double *meshSizes = new double[NT];

    double dxStart{1.0/(NX)};
    double DT = SIGMA*dxStart*dxStart;
    double time{0.0};

    for(int i=0; i<NX; i++){
        uOld[i].x=i*dxStart;
        uOld[i].y=std::sin(2*M_PI*i*dxStart);
        uNew[i].x=uOld[i].x;
        mesh.insert(Cell(uOld[i].x, uOld[i].y));
    }

    std::fstream output;
    output.open("output.csv", std::ios::out);

    int meshSize{NX};
    double start = omp_get_wtime();
#pragma omp parallel default(shared)
{
    for (int t=0; t<NT; t++) {

#pragma omp for schedule(static) reduction(+:energy[t]) reduction(+:error[t])
        for (int i=0; i<meshSize; i++) {
            int iBefore = (i==0)?meshSize-1:i-1;
            int iAfter = (i==meshSize-1)?0:i+1;
            double dxAfter = (i==meshSize-1)?std::abs(uOld[iAfter].x + 1 - uOld[i].x):std::abs(uOld[iAfter].x - uOld[i].x);
            double dxBefore = (i==0)?std::abs(uOld[i].x + 1 - uOld[iBefore].x):std::abs(uOld[i].x - uOld[iBefore].x);            
            double f = uOld[i].y;
            double fBefore = uOld[iBefore].y;
            double fAfter = uOld[iAfter].y;
            uOld[i].dudx = (fAfter-(dxAfter/dxBefore)*(dxAfter/dxBefore)*fBefore-(1-(dxAfter/dxBefore)*(dxAfter/dxBefore))*f)/(dxAfter*(1+dxAfter/dxBefore));
            uOld[i].d2udx2 = 2*(fAfter+(dxAfter/dxBefore)*fBefore-(1+(dxAfter/dxBefore))*f)/(dxAfter*dxBefore*(1+dxAfter/dxBefore));
            uNew[i].y = uOld[i].y + DT*(-C*uOld[i].dudx + K*uOld[i].d2udx2);
            uNew[i].yExact = std::sin(2*M_PI*(uNew[i].x-C*time))*std::exp(-time*K*std::pow(2*M_PI,2));
            energy[t] += std::abs(uOld[i].x-uOld[iBefore].x)*uNew[i].y*uNew[i].y;
            error[t] += std::abs(uNew[i].yExact - uNew[i].y)/meshSize;

#pragma omp critical
{
            output << uOld[i].x << "," << uOld[i].y << ";";
}
        }
#pragma omp single
{       
        output << std::endl;

        //REFINEMENT PROCESS
        if ((t+1)%101==0) {

            //WE NEED TWO ITERATORS POINTING TO THE PREVIOUS AND CURRENT VERTEX
            std::set<Cell>::iterator cellBefore = mesh.begin();
            std::set<Cell>::iterator thisCell = ++mesh.begin();
            int count = 1; //INTEGER COUNTER TO REFERENCE THE ARRAYS

            //STARTING THE LOOP
            cellBefore->y = uNew[0].y;
            while (thisCell != mesh.end()) {
                thisCell->y=uOld[count].y;

                //CAN REFINE?
                if (std::abs(thisCell->x-cellBefore->x)>DXMIN) {
                    if (std::abs(uNew[count].y-uNew[count].yExact)>BIASREFINE*error[t]) {

                        double dx = thisCell->x - 0.5*(cellBefore->x+thisCell->x);
                        double f = uOld[count].y - dx*uOld[count].dudx + 0.5*dx*dx*uOld[count].d2udx2;

                        mesh.insert( Cell(0.5*(cellBefore->x+thisCell->x), f));

                        cellBefore++;
                        cellBefore++;
                        thisCell++;
                        count++;
                        continue;
                    }
                }

                // CAN COARSEN?
                if (std::abs(thisCell->x-cellBefore->x)<DXMAX && thisCell != --mesh.end()) {
                    if (std::abs(uNew[count].y-uNew[count].yExact)<error[t]/BIASCOARSEN) {
                        thisCell=mesh.erase(thisCell);
                        count ++;
                        continue;
                    }
                }

                //INCREASE ITERATORS AND COUNTER
                cellBefore++;
                thisCell++;
                count++;
            }

            //REINIT VALUES
            delete [] uOld;
            delete [] uNew;
            meshSize = mesh.size();
            dxStart = 1.0/meshSize;
            DT = SIGMA*dxStart*dxStart;
            uNew = new Node[meshSize];
            uOld = new Node[meshSize];

            //FILL THE ARRAY
            count = 0;
            for (auto value: mesh) {
                uOld[count].x=value.x;
                uOld[count].y=value.y;
                uNew[count].x=value.x;
                uNew[count].y=value.y;
                count ++;
            }
}

        }
#pragma omp single
{
        time += DT;
        std::swap(uOld, uNew);
        meshSizes[t]=meshSize;
}
    }
}
    double end = omp_get_wtime();

    output.close();
    std::cout << "Time: " << end-start << std::endl;
    std::cout << "Error: " << error[NT-1] << std::endl;

    std::fstream outError;
    std::fstream outMeshSize;

    outError.open("error.csv", std::ios::out);
    outMeshSize.open("mesh.csv", std::ios::out);

    for (int t=0; t<NT; t++) {
        outMeshSize << meshSizes[t] << std::endl;
        outError << error[t] << std::endl;
    }

    for (auto v: mesh) {
        std::cout << v.x << " ";;
    }
    std::cout << std::endl;
    
    delete [] uOld;
    delete [] uNew;
    delete [] error;
    delete [] energy;
    delete [] meshSizes;
    return 0;
}
