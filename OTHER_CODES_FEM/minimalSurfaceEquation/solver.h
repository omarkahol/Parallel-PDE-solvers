#ifndef SOLVER_H
#define SOLVER_H

#include "include_deal.h"
#include <sstream>
using namespace dealii;

template<int dim>
class solver {
public:
    solver();
    ~solver(){dofHandler.clear(); mesh.clear(); systemMatrix.clear();};
    void run() {

        double tol=1e-5;
        const int maxRef=8;

        this -> makeGrid();
        this -> setupSystem(true);
        this -> interpolateBoundaryValues();

        unsigned int refinement = 0;
        bool compute = true;
        while(compute) {
            ++refinement;
            pcout << std::endl << "****************************************************************" << std::endl;
            pcout << "\t\tREFINED MESH: " << refinement << std::endl;
            pcout << "****************************************************************" << std::endl;
            pcout << "\tActive cells: " << mesh.n_active_cells() << std::endl;
            pcout << "\tNumber DoF: " << dofHandler.n_dofs() << std::endl;
            this -> refineMesh();
            this -> interpolateBoundaryValues();

            for (unsigned int inner_iteration = 0; inner_iteration < 5; ++inner_iteration) {
                this -> setupSystem(false);
                this ->assembleSystem();
                this -> solve();
                this -> interpolateBoundaryValues();
                this -> computeSetLength();
                pcout << " \t Residual: " << residualValue << ", Step Length: " << currentStepLength << std::endl;
                if (residualValue < tol) {
                    break;
                }
            }
            this -> outputResults(refinement);

            timer.print_summary();
            pcout << std::endl;

            compute = (residualValue > tol) || (refinement < maxRef);
       }
 }
private:
    void makeGrid();
    void setupSystem(const bool &reinitData);
    void assembleSystem();
    void solve();
    void interpolateBoundaryValues();
    void refineMesh();
    double residual();
    void computeSetLength();
    void outputResults(const unsigned int &cycle);
private:

    MPI_Comm communicator;
    IndexSet ownedDofs;
    IndexSet relevantDofs;
    ConditionalOStream pcout;
    TimerOutput timer;

    parallel::distributed::Triangulation<dim> mesh;
    DoFHandler<dim> dofHandler;
    FE_Q<dim> fe;
    AffineConstraints<double> constraints;
    SparsityPattern sparsityPattern;

    dealii::LinearAlgebraPETSc::MPI::SparseMatrix systemMatrix;
    dealii::LinearAlgebraPETSc::MPI::Vector systemRhs;

    dealii::LinearAlgebraPETSc::MPI::Vector ownedSolution;
    dealii::LinearAlgebraPETSc::MPI::Vector ownedNewtonUpdate;
    dealii::LinearAlgebraPETSc::MPI::Vector relevantSolution;


    double currentStepLength{0.125};
    double residualValue{1};
};

template <int dim>
class BoundaryValues: public Function<dim> {
public:
    BoundaryValues(): Function<dim>(){}
    virtual double value(const Point<dim> &p, const unsigned int component=0) const override {
        return std::sin(3*std::atan2(p[1],p[0]));
    }
};

template <int dim>
class RhsForcingTerm: public Function<dim> {
public:
    RhsForcingTerm(): Function<dim>(){}
    virtual double value(const Point<dim> &p, const unsigned int component=0) const override {
        return 0.5;
    }
};

template <int dim>
solver<dim>::solver():
communicator(MPI_COMM_WORLD),
mesh(communicator, typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement | Triangulation<dim>::smoothing_on_coarsening)),
fe(2),
dofHandler(mesh),
pcout(std::cout, Utilities::MPI::this_mpi_process(communicator)==0),
timer(communicator, pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times_grouped){}

template<int dim>
void solver<dim>::makeGrid(){
    GridGenerator::hyper_ball(mesh);
    mesh.refine_global(2);
}

template<int dim>
void solver<dim>::setupSystem(const bool &reinitData){
    TimerOutput::Scope t(timer, "setup system");
    if (reinitData) {
        //INITIALIZING DOF HANDLER WITH FINITE ELEMENT AND THE SOLUTION VECTOR
        dofHandler.distribute_dofs(fe);

        ownedDofs = dofHandler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dofHandler, relevantDofs);

        relevantSolution.reinit(ownedDofs, relevantDofs, communicator);
        ownedSolution.reinit(ownedDofs, communicator);

        //MAKING CONSTRAINTS FOR THE NEWTON UPDATE
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dofHandler, constraints);
        DoFTools::make_zero_boundary_constraints(dofHandler,constraints);
        constraints.close();

        DynamicSparsityPattern dsp(relevantDofs);
        DoFTools::make_sparsity_pattern(dofHandler, dsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern(dsp, dofHandler.n_locally_owned_dofs_per_processor(), communicator, relevantDofs);
        sparsityPattern.copy_from(dsp);
        this -> currentStepLength=0.125;
    }

    //REINIT LINEAR SYSTEM
    ownedNewtonUpdate.reinit(ownedDofs, communicator);

    systemRhs.reinit(ownedDofs, communicator);
    systemMatrix.reinit(ownedDofs, ownedDofs, sparsityPattern, communicator);
}

template <int dim>
void solver<dim>::assembleSystem() {
    TimerOutput::Scope t(timer, "assemble");

    //ASSEMBLE LINEAR SYSTEM
    const QGauss<dim> quad(fe.degree+1);
    FEValues<dim> feVal(fe, quad, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofsPerCell = fe.dofs_per_cell;
    const unsigned int numQuadPoints = quad.size();

    //BUILD CELL MATRICES
    FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
    Vector<double> cellRhs(dofsPerCell);
    std::vector<Tensor<1,dim>> solutionGradients(numQuadPoints);

    std::vector<double> rhsForcingTermVal(numQuadPoints,0);
    RhsForcingTerm<dim> rhsForcingTerm;

    std::vector<types::global_dof_index> localDofIndices(dofsPerCell);
    for (const auto &cell : dofHandler.active_cell_iterators()) {
        if (cell -> is_locally_owned()) {
            cellMatrix=0.0;
            cellRhs=0.0;
            feVal.reinit(cell);
            feVal.get_function_gradients(relevantSolution, solutionGradients);
            rhsForcingTerm.value_list(feVal.get_quadrature_points(), rhsForcingTermVal);

            for (unsigned int q=0; q<numQuadPoints; q++) {
                const double coeff = 1.0/std::sqrt(1+solutionGradients[q]*solutionGradients[q]);

                for (unsigned int i=0; i<dofsPerCell; i++) {
                    for(unsigned int j=0; j<dofsPerCell; j++) {
                        cellMatrix(i,j) +=(((feVal.shape_grad(i, q)*coeff * feVal.shape_grad(j, q)) -
                                            (feVal.shape_grad(i, q)
                                             * coeff * coeff * coeff
                                             * (feVal.shape_grad(j, q)
                                             * solutionGradients[q])
                                             * solutionGradients[q]))
                                             * feVal.JxW(q));
                    }
                    cellRhs(i) += (rhsForcingTermVal[q]*feVal.shape_value(i,q) - feVal.shape_grad(i, q) * coeff * solutionGradients[q]) * feVal.JxW(q);
                }
            }
            cell -> get_dof_indices(localDofIndices);
            constraints.distribute_local_to_global(cellMatrix, cellRhs, localDofIndices, systemMatrix, systemRhs);
        }
    }
    systemMatrix.compress(VectorOperation::add);
    systemRhs.compress(VectorOperation::add);
}

template <int dim>
void solver<dim>::solve() {
  TimerOutput::Scope t(timer, "solve");

  SolverControl solver_control(dofHandler.n_dofs(),1e-5);
  dealii::LinearAlgebraPETSc::SolverCG solver(solver_control);

  dealii::LinearAlgebraPETSc::MPI::PreconditionJacobi preconditioner;
  preconditioner.initialize(systemMatrix);

  solver.solve(systemMatrix, ownedNewtonUpdate, systemRhs, preconditioner);
  constraints.distribute(ownedNewtonUpdate);
  ownedSolution.add(this -> currentStepLength, ownedNewtonUpdate);
}

template <int dim>
void solver<dim>::refineMesh() {
    TimerOutput::Scope t(timer, "refinement");

    Vector<float> estimatedError(mesh.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dofHandler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      relevantSolution,
      estimatedError);
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(mesh,estimatedError,0.3,0.03);

    mesh.prepare_coarsening_and_refinement();

    parallel::distributed::SolutionTransfer<dim, PETScWrappers::MPI::Vector> transfer(dofHandler);
    transfer.prepare_for_coarsening_and_refinement(relevantSolution);

    mesh.execute_coarsening_and_refinement();
    this -> setupSystem(true);
    transfer.interpolate(ownedSolution);
}

template <int dim>
void solver<dim>::interpolateBoundaryValues() {
    TimerOutput::Scope t(timer, "boundary values");
    std::map<dealii::types::global_dof_index, double> boundaryValues;

    VectorTools::interpolate_boundary_values(dofHandler,0,BoundaryValues<dim>(), boundaryValues);
    for (const auto &bv: boundaryValues) {
        ownedSolution[bv.first]=bv.second;
    }
    ownedSolution.compress(VectorOperation::insert);
    relevantSolution=ownedSolution;
    relevantSolution.compress(VectorOperation::insert);
}

template <int dim>
double solver<dim>::residual() {
    TimerOutput::Scope t(timer, "residual");
    dealii::LinearAlgebraPETSc::MPI::Vector residual(ownedDofs, communicator);

    const QGauss<dim> quad(fe.degree + 1);
    FEValues<dim> feVal(fe, quad, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    const unsigned int dofsPerCell = fe.dofs_per_cell;
    const unsigned int numQuadPoints = quad.size();

    Vector<double> cellResidual(dofsPerCell);
    std::vector<Tensor<1, dim>> solutionGradients(numQuadPoints);

    std::vector<types::global_dof_index> localDofIndices(dofsPerCell);

    std::vector<double> rhsForcingTermVal(numQuadPoints);
    RhsForcingTerm<dim> rhsForcingTerm;

    for (const auto &cell : dofHandler.active_cell_iterators()) {
        if (cell -> is_locally_owned()) {
            cellResidual = 0.0;
            feVal.reinit(cell);
            feVal.get_function_gradients(relevantSolution, solutionGradients);
            rhsForcingTerm.value_list(feVal.get_quadrature_points(), rhsForcingTermVal);

            for (unsigned int q = 0; q < numQuadPoints; ++q) {
                const double coeff = 1 / std::sqrt(1 + solutionGradients[q] * solutionGradients[q]);
                for (unsigned int i = 0; i < dofsPerCell; ++i) {
                    cellResidual(i) += (rhsForcingTermVal[q]*feVal.shape_value(i,q) - feVal.shape_grad(i, q) * coeff * solutionGradients[q]) * feVal.JxW(q);
                }
            }
            cell -> get_dof_indices(localDofIndices);
            constraints.distribute_local_to_global(cellResidual, localDofIndices, residual);
        }
    }
    residual.compress(VectorOperation::add);
    return residual.l2_norm();
}

template <int dim>
void solver<dim>::computeSetLength() {
    TimerOutput::Scope t(timer, "step length");
    double previousResidual = systemRhs.l2_norm();
    this -> residualValue = this -> residual();
    if (residualValue < previousResidual) {
        if (std::abs(currentStepLength-1.0)<1e-5) {
            return;
        } else {
            this -> currentStepLength *= 2.0;
            return;
        }
    } else {
        currentStepLength *= 1.0/2.0;
        return;
    }
}

template <int dim>
void solver<dim>::outputResults(const unsigned int &cycle) {
    TimerOutput::Scope t(timer, "output");

    DataOut<dim> dataOut;
    dataOut.attach_dof_handler(dofHandler);

    dataOut.add_data_vector(relevantSolution, "solution");

    dataOut.build_patches();
    std::ofstream outstream("solution-" + Utilities::int_to_string(cycle, 2) + "." +Utilities::int_to_string(mesh.locally_owned_subdomain(), 4));
    dataOut.write_vtu(outstream);

    if (Utilities::MPI::this_mpi_process(communicator)==0) {
        std::vector<std::string> filenames;

        for(int i=0; i<Utilities::MPI::n_mpi_processes(communicator); i++) {
            filenames.push_back("solution-" + Utilities::int_to_string(cycle, 2) + "." +Utilities::int_to_string(i, 4));
        }

        std::ofstream masterOutput("master_"+Utilities::int_to_string(cycle, 2)+".pvtu");
        dataOut.write_pvtu_record(masterOutput,filenames);
    }
}
#endif // SOLVER_H
