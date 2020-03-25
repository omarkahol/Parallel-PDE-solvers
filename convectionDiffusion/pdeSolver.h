#ifndef PDESOLVER_H
#define PDESOLVER_H

#include "include_deal.h"
#include <sstream>

template <int dim>
class pdeSolver {
public:
    pdeSolver();
    void run();
private:
    void makeGrid();
    void setupSystem(const bool&, const bool&);
    void assembleSystem();
    int solve();
    void refineMesh(const double&, const double &,const unsigned int&);
    void executeTransferAndRefinement(const bool&, const bool&);
    void outputResults();
private:
    dealii::Triangulation<dim> mesh;
    dealii::DoFHandler<dim> dofHandler;
    dealii::FE_Q<dim> finiteElement;
    dealii::AffineConstraints<double> constraints;

    dealii::PETScWrappers::MPI::SparseMatrix systemMatrix;
    dealii::PETScWrappers::MPI::SparseMatrix systemRhsMatrix;

    dealii::PETScWrappers::MPI::Vector systemRhs;
    dealii::PETScWrappers::MPI::Vector newSolution;

    dealii::Tensor<1,dim> C;
    double K{0.01};
    double time{0.0};
    double timeStep{1.0/120.0};
    unsigned int timeStepNumber{0};
    const double theta{0.53};

    unsigned int mpiProcs;
    unsigned int mpiProcId;
    MPI_Comm communicator;
};

template<int dim>
class InitialSolution: public dealii::Function<dim> {
public:
    double value(const dealii::Point<dim> &point, const unsigned int component = 0) const override {
        double checkValue{0.0};
        for(int i=0; i<dim; i++) {
            checkValue += point(i)*point(i);
        }
        if (std::sqrt(checkValue)<0.5) {
            return 1;
        } else {
            return 0;
        }
    }
};


template <int dim>
pdeSolver<dim>::pdeSolver(): finiteElement(1),dofHandler(mesh) {
    communicator=MPI_COMM_WORLD;
    mpiProcs=dealii::Utilities::MPI::n_mpi_processes(communicator);
    mpiProcId=dealii::Utilities::MPI::this_mpi_process(communicator);

    for (int i=0; i<dim; i++)
        C[i] = 2;
};

template<int dim>
void pdeSolver<dim>::makeGrid() {
    dealii::GridGenerator::hyper_cube(mesh,-2,2,true);
    mesh.refine_global(5);
}

template <int dim>
void pdeSolver<dim>::setupSystem(const bool &keepConstraints, const bool &zeroBoundary) {
    dealii::GridTools::partition_triangulation(mpiProcs, mesh);
    dofHandler.distribute_dofs(finiteElement);
    dealii::DoFRenumbering::subdomain_wise(dofHandler);


    constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dofHandler, constraints);
    if (zeroBoundary) {
        dealii::VectorTools::interpolate_boundary_values(dofHandler,0,dealii::Functions::ZeroFunction<dim>(), constraints);
        dealii::VectorTools::interpolate_boundary_values(dofHandler,1,dealii::Functions::ZeroFunction<dim>(), constraints);
        dealii::VectorTools::interpolate_boundary_values(dofHandler,2,dealii::Functions::ZeroFunction<dim>(), constraints);
        dealii::VectorTools::interpolate_boundary_values(dofHandler,3,dealii::Functions::ZeroFunction<dim>(), constraints);
    } else {
        dealii::DoFTools::make_periodicity_constraints(dofHandler,0,1,0,constraints);
        dealii::DoFTools::make_periodicity_constraints(dofHandler,2,3,1,constraints);
        if (dim==3) {
            dealii::DoFTools::make_periodicity_constraints(dofHandler,4,5,2,constraints);
        }
    }
    constraints.close();

    dealii::DynamicSparsityPattern dsp(dofHandler.n_dofs(), dofHandler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dofHandler, dsp, constraints, keepConstraints);

    const std::vector<dealii::IndexSet> procDofs = dealii::DoFTools::locally_owned_dofs_per_subdomain(dofHandler);
    const dealii::IndexSet localDofs = procDofs[mpiProcId];

    systemMatrix.reinit(localDofs,localDofs,dsp,communicator);
    systemRhsMatrix.reinit(localDofs,localDofs,dsp,communicator);
    systemRhs.reinit(localDofs,communicator);
}

template <int dim>
int pdeSolver<dim>::solve() {
    dealii::SolverControl solverControl(1000, 1e-8*systemRhs.l2_norm());
    dealii::PETScWrappers::SolverCG solver(solverControl, communicator);

    dealii::PETScWrappers::PreconditionBlockJacobi precond(systemMatrix);

    solver.solve(systemMatrix, newSolution, systemRhs, precond);

    dealii::Vector<double> localizedSolution(newSolution);
    constraints.distribute(localizedSolution);
    newSolution = localizedSolution;
    return solverControl.last_step();
}

template <int dim>
void pdeSolver<dim>::outputResults() {
    const dealii::Vector<double> localizedSolution(newSolution);
    if (mpiProcId==0) {

    dealii::DataOut<dim> dataOut;
    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(localizedSolution, "U");
    dataOut.build_patches();

    std::stringstream fileName;
    fileName << "solution_" << timeStepNumber << ".vtk";

    std::ofstream output(fileName.str());

    dataOut.write_vtk(output);
    }
}

template <int dim>
void pdeSolver<dim>::refineMesh(const double &coarsen, const double &refine, const unsigned int &max) {

    const dealii::Vector<double> localizedSolution(newSolution);

    dealii::Vector<float> estimatedErrorPerCell(mesh.n_active_cells());
    dealii::KellyErrorEstimator<dim>::estimate(
                dofHandler,
                dealii::QGauss<dim - 1>(finiteElement.degree + 1),
                {},
                localizedSolution,
                estimatedErrorPerCell,
                dealii::ComponentMask(),
                nullptr,
                dealii::MultithreadInfo::n_threads(),
                mpiProcId);
    const unsigned int nLocalCells = dealii::GridTools::count_cells_with_subdomain_association(mesh, mpiProcId);
    dealii::PETScWrappers::MPI::Vector distributedAllErrors(communicator, mesh.n_active_cells(),nLocalCells );

    for(unsigned int i=0; i<estimatedErrorPerCell.size(); ++i) {
        if(estimatedErrorPerCell(i) != 0)
            distributedAllErrors(i)=estimatedErrorPerCell(i);
    }
    distributedAllErrors.compress(dealii::VectorOperation::insert);
    const dealii::Vector<float> localizeAllErrors(distributedAllErrors);
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(mesh,localizeAllErrors,refine,coarsen);
    if (mesh.n_levels() > max)
      for (const auto &cell : mesh.active_cell_iterators_on_level(max))
        cell->clear_refine_flag();
}

template <int dim>
void pdeSolver<dim>::executeTransferAndRefinement(const bool &keepConstraints, const bool &zeroBoundary) {
    dealii::SolutionTransfer<dim> solutionTransfer(dofHandler);
    dealii::Vector<double> previousSolution(newSolution);
    previousSolution = newSolution;
    mesh.prepare_coarsening_and_refinement();
    solutionTransfer.prepare_for_coarsening_and_refinement(previousSolution);
    mesh.execute_coarsening_and_refinement();

    this -> setupSystem(keepConstraints, zeroBoundary);

    const std::vector<dealii::IndexSet> procDofs = dealii::DoFTools::locally_owned_dofs_per_subdomain(dofHandler);
    const dealii::IndexSet localDofs = procDofs[mpiProcId];
    newSolution.reinit(localDofs, communicator);
    dealii::Vector<double> localizedSolution(newSolution);
    solutionTransfer.interpolate(previousSolution,localizedSolution);
    newSolution=localizedSolution;
}

template <int dim>
void pdeSolver<dim>::assembleSystem() {
    const dealii::QGauss<dim> quad(finiteElement.degree + 1);
    dealii::FEValues<dim> feVal(finiteElement, quad, dealii::update_values | dealii::update_gradients |
                                dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofsCell = finiteElement.dofs_per_cell;
    const unsigned int nQuad = quad.size();

    dealii::FullMatrix<double> cellMatrix(dofsCell, dofsCell);
    dealii::FullMatrix<double> cellRhsMatrix(dofsCell, dofsCell);

    std::vector<dealii::types::global_dof_index> localDofIndices(dofsCell);

    for(const auto &cell : dofHandler.active_cell_iterators()) {
        if (cell->subdomain_id() == mpiProcId) {
        cellMatrix = 0;
        cellRhsMatrix = 0;
        feVal.reinit(cell);
        for (unsigned int q = 0; q<nQuad; q++) {
            for (unsigned int i=0; i<dofsCell; i++) {
                for (unsigned int j=0; j<dofsCell; j++) {
                    cellMatrix(i,j) += (feVal.shape_value(i,q)*feVal.shape_value(j,q) + timeStep*theta*K*feVal.shape_grad(i,q)*feVal.shape_grad(j,q)+timeStep*theta*feVal.shape_value(i,q)*(C*feVal.shape_grad(j,q)))*feVal.JxW(q);
                    cellRhsMatrix(i,j) += (feVal.shape_value(i,q)*feVal.shape_value(j,q) - timeStep*(1-theta)*K*feVal.shape_grad(i,q)*feVal.shape_grad(j,q) -timeStep*(1-theta)*feVal.shape_value(i,q)*(C*feVal.shape_grad(j,q)))*feVal.JxW(q);
                }
            }
        }
        cell -> get_dof_indices(localDofIndices);
        constraints.distribute_local_to_global(cellMatrix,localDofIndices,systemMatrix);
        constraints.distribute_local_to_global(cellRhsMatrix,localDofIndices,systemRhsMatrix);
        }
    }

    systemMatrix.compress(dealii::VectorOperation::add);
    systemRhsMatrix.compress(dealii::VectorOperation::add);
    systemRhsMatrix.vmult(systemRhs,newSolution);
}

template <int dim>
void pdeSolver<dim> :: run() {
    this -> makeGrid();

    for(;time <=5; time += timeStep) {
        std::cout << "Time step: " << ++timeStepNumber << " at t="<<time << std::endl;
        if (timeStepNumber==1){
            this -> setupSystem(false, false);
            const std::vector<dealii::IndexSet> procDofs = dealii::DoFTools::locally_owned_dofs_per_subdomain(dofHandler);
            const dealii::IndexSet localDofs = procDofs[mpiProcId];
             newSolution.reinit(localDofs, communicator);

                    dealii::Vector<double> localizedSolution(dofHandler.n_dofs());
                    dealii::VectorTools::project(dofHandler, constraints, dealii::QGauss<dim>(finiteElement.degree +1), InitialSolution<dim>(), newSolution);
                }
        this -> refineMesh(0.4,0.6,6);
        this -> executeTransferAndRefinement(false, false);
        this -> assembleSystem();
        this -> solve();
        this -> outputResults();
        }
}

#endif // PDESOLVER_H
