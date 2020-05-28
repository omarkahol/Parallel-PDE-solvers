#include <iostream>
#include <set>
#include <cmath>
#include <fstream>
#include <mpi.h>

#define NX 25
#define NT 10000
#define C 1.0
#define K 0.01

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

int main(int argc, char** argv) {

    int ierr, id, numProcs;
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &id);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int idPrev = (id==0)?numProcs-1:id-1;
    int idAfter = (id==numProcs-1)?0:id+1;

    const int activePoints = numProcs*(NX-2);
    const double dx = 1.0/activePoints;

    

    ierr = MPI_Finalize();
    return ierr;

}