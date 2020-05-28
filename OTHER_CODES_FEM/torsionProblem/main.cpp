#include <iostream>
#include "solver.h"

int main(int argc, char **argv)
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    solver sol;
    sol.run();

    return 0;
}
