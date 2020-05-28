#ifndef SOLVER_H
#define SOLVER_H

#include "include_deal.h"
using namespace dealii;

template <int dim>
class solver {
public:
    solver();
    void run();
private:
    void makeGrid(const int&);
    void buildInitialSolution(const int&);
    void setupSystem(const bool&);
    void assembleSystem();
    void assembleLaplaceProblem();
    int solve();
    void refineGrid(const double&, const double&, const int&, const int&, const bool&);
    void outputResults(const unsigned int &);
    void printRelevantInfo(const int&);
private:
    MPI_Comm communicator;
    parallel::distributed::Triangulation<dim> mesh;
    FE_Q<dim> fe;
    DoFHandler<dim> dofHandler;

    IndexSet ownedDofs;
    IndexSet relevantDofs;

    AffineConstraints<double> constraints;
    SparsityPattern sparsityPattern;
    LinearAlgebraPETSc::MPI::SparseMatrix systemMatrix;
    LinearAlgebraPETSc::MPI::Vector localSolution;
    LinearAlgebraPETSc::MPI::Vector systemRhs;

    ConditionalOStream pcout;
    TimerOutput timer;

    const double K{0.8};
    Tensor<1,dim> C;
    double timeStep{1.0/1000};
    int timeStepNumber{0};
    double time{0.0};
};

template<int dim>
class LaplaceForce: public dealii::Function<dim> {
public:
    double value(const dealii::Point<dim> &point, const unsigned int component = 0) const override {
        if(point.norm()<0.5) {
            return 1;
        } else {
            return 0;
        }
    }
};

template<int dim>
class LaplaceCoefficient: public dealii::Function<dim> {
public:
    double value(const dealii::Point<dim> &point, const unsigned int component = 0) const override {
        if (point.norm() < 0.5) {
            return 1;
        } else {
            return 20;
        }
    }
};

template<int dim>
solver<dim>::solver():
    communicator(MPI_COMM_WORLD),
    mesh(communicator, typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement |Triangulation<dim>::smoothing_on_coarsening)),
    fe(1),
    dofHandler(mesh),
    pcout(std::cout, Utilities::MPI::this_mpi_process(communicator)==0),
    timer(communicator, pcout, TimerOutput::summary,TimerOutput::wall_times){

    for(int i=0; i<dim; i++) {
        C[i]=10.0;
    }
}

template <int dim>
void solver<dim>::makeGrid(const int &globalLevel) {
    std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>> periodicityVector;
    GridGenerator::hyper_cube(mesh,-2,2, true);
    GridTools::collect_periodic_faces(mesh, 0,1,0,periodicityVector);
    GridTools::collect_periodic_faces(mesh, 2,3,1,periodicityVector);
    if (dim==3) {
        GridTools::collect_periodic_faces(mesh, 4,5,2,periodicityVector);
    }
    mesh.add_periodicity(periodicityVector);
    mesh.refine_global(globalLevel);
}

template<int dim>
void solver<dim>::setupSystem(const bool &periodicBC) {
    TimerOutput::Scope t(timer, "setup system");
    dofHandler.distribute_dofs(fe);

    ownedDofs = dofHandler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dofHandler, relevantDofs);

    localSolution.reinit(ownedDofs,relevantDofs,communicator);
    systemRhs.reinit(ownedDofs, communicator);

    constraints.clear();
    constraints.reinit(relevantDofs);
    DoFTools::make_hanging_node_constraints(dofHandler, constraints);

    if (periodicBC) {
        std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>> periodicityVector;
        GridTools::collect_periodic_faces(dofHandler, 0,1,0,periodicityVector);
        GridTools::collect_periodic_faces(dofHandler, 2,3,1,periodicityVector);
        if (dim==3) {
            GridTools::collect_periodic_faces(dofHandler, 4,5,2,periodicityVector);
        }

        DoFTools::make_periodicity_constraints<DoFHandler<dim>>(periodicityVector,constraints);

    } else {
        for (int i=0; i<GeometryInfo<dim>::faces_per_cell; i++)
            DoFTools::make_zero_boundary_constraints(dofHandler,i,constraints);
    }
    constraints.close();

    DynamicSparsityPattern dsp(relevantDofs);

    DoFTools::make_sparsity_pattern(dofHandler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp, dofHandler.n_locally_owned_dofs_per_processor(),communicator, relevantDofs);
    sparsityPattern.copy_from(dsp);
    systemMatrix.reinit(ownedDofs,ownedDofs,sparsityPattern,communicator);
}

template <int dim>
void solver<dim>::buildInitialSolution(const int &level) {
    TimerOutput::Scope t(timer, "initial solution");
    for (int cycle = 0; cycle < 8; cycle ++) {
        pcout << "initial solution step: " << cycle << std::endl;
        if (cycle != 0)
            this -> refineGrid(0.3,0.03,level, level+2, false);
        else
            this -> setupSystem(false);
        this -> assembleLaplaceProblem();
        this -> solve();
    }
    this -> outputResults(0);
}

template <int dim>
void solver<dim>::assembleLaplaceProblem() {
    const dealii::QGauss<dim> quad(fe.degree + 1);
    dealii::FEValues<dim> feVal(fe, quad, dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofsCell = fe.dofs_per_cell;
    const unsigned int nQuad = quad.size();

    dealii::FullMatrix<double> cellMatrix(dofsCell, dofsCell);
    dealii::Vector<double> cellRhs(dofsCell);
    LaplaceCoefficient<dim> coeff;
    LaplaceForce<dim> rhs;
    std::vector<dealii::types::global_dof_index> localDofIndices(dofsCell);

    for(const auto &cell : dofHandler.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            cell -> get_dof_indices(localDofIndices);
            cellMatrix = 0;
            cellRhs = 0;
            feVal.reinit(cell);
            const auto &quadPoints=feVal.get_quadrature_points();

            for (unsigned int q = 0; q<nQuad; q++) {
                for (unsigned int i=0; i<dofsCell; i++) {
                    cellRhs(i) += rhs.value(quadPoints[q])*feVal.JxW(q)*feVal.shape_value(i,q);
                    for (unsigned int j=0; j<dofsCell; j++) {
                        cellMatrix(i,j) += feVal.JxW(q)*coeff.value(quadPoints[q])*(feVal.shape_grad(i,q)*feVal.shape_grad(j,q));
                    }
                 }
            }
            constraints.distribute_local_to_global(cellMatrix, cellRhs, localDofIndices, systemMatrix, systemRhs);
        }
    }
    systemMatrix.compress(dealii::VectorOperation::add);
    systemRhs.compress(dealii::VectorOperation::add);
}

template <int dim>
void solver<dim>::assembleSystem() {
    TimerOutput::Scope t(timer, "assembly");
    const dealii::QGauss<dim> quad(fe.degree + 1);
    dealii::FEValues<dim> feVal(fe, quad, dealii::update_values | dealii::update_gradients |
                                dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofsCell = fe.dofs_per_cell;
    const unsigned int nQuad = quad.size();

    dealii::FullMatrix<double> cellMatrix(dofsCell, dofsCell);
    dealii::Vector<double> cellRhs(dofsCell);
    std::vector<dealii::types::global_dof_index> localDofIndices(dofsCell);
    pcout << dofsCell << std::endl;

    for(const auto &cell : dofHandler.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            cell -> get_dof_indices(localDofIndices);
            cellMatrix = 0;
            cellRhs = 0;
            feVal.reinit(cell);
            for (unsigned int q = 0; q<nQuad; q++) {
                for (unsigned int i=0; i<dofsCell; i++) {
                    for (unsigned int j=0; j<dofsCell; j++) {
                        cellMatrix(i,j) += feVal.JxW(q)*(feVal.shape_value(i,q)*feVal.shape_value(j,q)+timeStep*K*feVal.shape_grad(i,q)*feVal.shape_grad(j,q));
                        cellRhs(i) += feVal.JxW(q)*localSolution(localDofIndices[j])*(feVal.shape_value(i,q)*feVal.shape_value(j,q)-timeStep*feVal.shape_value(i,q)*(feVal.shape_grad(j,q)*C));
                    }
                }
            }
            constraints.distribute_local_to_global(cellMatrix, cellRhs, localDofIndices, systemMatrix, systemRhs);
        }
    }
    systemMatrix.compress(dealii::VectorOperation::add);
    systemRhs.compress(dealii::VectorOperation::add);
}

template <int dim>
int solver<dim>::solve() {
    TimerOutput::Scope t(timer, "solve");
    LinearAlgebraPETSc::MPI::Vector distributedSolution(ownedDofs, communicator);

    SolverControl control(10*dofHandler.n_dofs(), 1e-5);
    LinearAlgebraPETSc::SolverCG solver(control, communicator);
    LinearAlgebraPETSc::MPI::PreconditionJacobi precond;
   // LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data;
   // data.symmetric_operator=false;

    precond.initialize(systemMatrix);

    solver.solve(systemMatrix, distributedSolution, systemRhs, precond);
    constraints.distribute(distributedSolution);
    localSolution=distributedSolution;
    return control.last_step();
}

template <int dim>
void solver<dim>::refineGrid(const double &refine, const double &coarsen, const int &minLevel, const int &maxLevel, const bool &periodicBc)
{
  TimerOutput::Scope t(timer, "refine");
  Vector<float> estimatedError(mesh.n_active_cells());

  KellyErrorEstimator<dim>::estimate(
    dofHandler,
    QGauss<dim - 1>(fe.degree + 1),
    std::map<types::boundary_id, const Function<dim> *>(),
    localSolution,
    estimatedError);

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(mesh, estimatedError, refine, coarsen);

  if (mesh.n_levels() > maxLevel) {
      for(const auto &cell : mesh.active_cell_iterators_on_level(maxLevel)) {
          if (cell -> is_locally_owned())
            cell -> clear_refine_flag();
      }
  }
  for (const auto &cell : mesh.active_cell_iterators_on_level(minLevel)) {
      if (cell -> is_locally_owned())
        cell -> clear_coarsen_flag();
  }

  mesh.prepare_coarsening_and_refinement();
  parallel::distributed::SolutionTransfer<dim, PETScWrappers::MPI::Vector> transfer(dofHandler);

  transfer.prepare_for_coarsening_and_refinement(localSolution);
  mesh.execute_coarsening_and_refinement();
  this -> setupSystem(periodicBc);
  LinearAlgebraPETSc::MPI::Vector interpolatedSolution(ownedDofs, communicator);
  transfer.interpolate(interpolatedSolution);
  constraints.distribute(interpolatedSolution);
  localSolution=interpolatedSolution;
}

template <int dim>
void solver<dim>::outputResults(const unsigned int &cycle) {
    TimerOutput::Scope t(timer, "output");
    DataOut<dim> dataOut;
    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(localSolution, "u");
    Vector<float> subdomain(mesh.n_active_cells());

    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = mesh.locally_owned_subdomain();

    dataOut.add_data_vector(subdomain, "subdomain");
    dataOut.build_patches();

    const std::string filename = ("solution-" + Utilities::int_to_string(cycle, 2) + "." +Utilities::int_to_string(mesh.locally_owned_subdomain(), 4));
    std::ofstream output(filename + ".vtu");
    dataOut.write_vtu(output);
    if (Utilities::MPI::this_mpi_process(communicator) == 0)
      {
        std::vector<std::string> filenames;

        for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(communicator); ++i)
          filenames.push_back("solution-" + Utilities::int_to_string(cycle, 2) + "." + Utilities::int_to_string(i, 4) + ".vtu");

        std::ofstream masterOutput("master-" + Utilities::int_to_string(cycle, 2) + ".pvtu");
        dataOut.write_pvtu_record(masterOutput, filenames);
      }
}

template<int dim>
void solver<dim>::printRelevantInfo(const int &iter) {
    pcout << std::endl << "****************************************" <<std::endl;
    pcout << "TIME STEP: " <<timeStepNumber << " at TIME: "<<time << std::endl;
    pcout << "Number of active cells: "
          << mesh.n_global_active_cells() << std::endl
          << "Number of degrees of freedom: " << dofHandler.n_dofs()
          << std::endl;
    timer.print_summary();
}

template<int dim>
void solver<dim>::run() {

    int minLevel{5};
    this -> makeGrid(minLevel);
    this -> buildInitialSolution(minLevel);
    int iter{0};
    while(timeStepNumber++ < 200) {
         if (timeStepNumber % 11 == 0) {
             for(int i=0; i<3; i++){
                this -> refineGrid(0.6,0.4,minLevel,minLevel+2,true);
                this -> assembleSystem();
                this -> solve();
             }
         } else {
            systemMatrix.reinit(ownedDofs,ownedDofs,sparsityPattern,communicator);
            systemRhs.reinit(ownedDofs, communicator);
            this->assembleSystem();
            iter = this->solve();
        }
         this->outputResults(timeStepNumber);
         this -> printRelevantInfo(iter);
         time += timeStep;
    }
}



#endif // SOLVER_H
