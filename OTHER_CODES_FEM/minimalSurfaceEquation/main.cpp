#include <iostream>
#include "solver.h"

int main(int argc, char **argv)
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    solver<2> solv;
    solv.run();
    return 0;
}
