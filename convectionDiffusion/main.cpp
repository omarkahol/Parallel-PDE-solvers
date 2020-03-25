#include "pdeSolver.h"

int main(int argc, char **argv) {

    dealii::Utilities::MPI::MPI_InitFinalize init(argc, argv, 1);
    pdeSolver<2> solver;
    solver.run();
}
