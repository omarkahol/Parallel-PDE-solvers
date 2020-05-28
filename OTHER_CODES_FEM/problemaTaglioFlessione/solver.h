#ifndef SOLVER_H
#define SOLVER_H
#include "include_deal.h"
#include "deal.II/fe/fe_nedelec.h"

using namespace dealii;

class solver {
public:
    solver();
    void run();
private:
    MPI_Comm communicator;
    parallel::distributed::Triangulation<2> mesh;

    dealii::FE_Q<2> fe;
    DoFHandler<2> dofHandler;

    AffineConstraints<double> constraints;
    LinearAlgebraPETSc::MPI::SparseMatrix systemMatrix;

    LinearAlgebraPETSc::MPI::Vector relevantSolution;
    LinearAlgebraPETSc::MPI::Vector systemRhs;

    IndexSet ownedDofs;
    IndexSet relevantDofs;

    ConditionalOStream pcout;
    TimerOutput timer;
public:
    constexpr static double poisson{0.5};

private:
    void makeGrid();
    void setupSystem();
    void assembleSystem();
    void solve();
    void refineGrid();
    void outputResults(const unsigned int cycle);
    double getInertia();
};


solver::solver():
    communicator(MPI_COMM_WORLD),
    mesh(communicator, typename Triangulation<2>::MeshSmoothing(Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening)),
    fe(3),
    dofHandler(mesh),
    pcout(std::cout, Utilities::MPI::this_mpi_process(communicator)==0),
    timer(communicator, pcout, TimerOutput::summary, TimerOutput::wall_times){}

void solver::run(){
    this -> makeGrid();

    for (unsigned int cycle=0; cycle<6; cycle++) {

        if (cycle != 0)
            this -> refineGrid();

        this -> setupSystem();
        this -> assembleSystem();
        this -> solve();
        this -> outputResults(cycle);

        pcout << "*************************************************************************************"<<std::endl;
        pcout << "\tCycle: " << cycle << std::endl;
        pcout << "\tActive cells: " << mesh.n_active_cells() << std::endl;
        pcout << "\tDoFs: " << dofHandler.n_dofs() << std::endl;
        pcout << "\tInertia: " << this ->getInertia() << std::endl;
        timer.print_summary();
        pcout << "*************************************************************************************"<<std::endl;
    }
}

void solver::setupSystem() {
    TimerOutput::Scope t(timer, "setup");

    dofHandler.distribute_dofs(fe);
    ownedDofs = dofHandler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dofHandler,relevantDofs);

    relevantSolution.reinit(ownedDofs, relevantDofs, communicator);
    systemRhs.reinit(ownedDofs, communicator);

    constraints.clear();
    constraints.reinit(relevantDofs);
    DoFTools::make_hanging_node_constraints(dofHandler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(relevantDofs);
    DoFTools::make_sparsity_pattern(dofHandler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,dofHandler.n_locally_owned_dofs_per_processor(), communicator, relevantDofs);
    systemMatrix.reinit(ownedDofs, ownedDofs, dsp, communicator);

}

void solver::assembleSystem() {
    TimerOutput::Scope t(timer, "assemble");

    const QGauss<2> quadratureFormula(fe.degree + 1);
    const QGauss<1> faceQuadratureFormula(fe.degree + 1);

    FEValues<2> feValues(fe, quadratureFormula,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEFaceValues<2> feFaceValues(fe, faceQuadratureFormula,
            update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);

    const unsigned int dofsPerCell=fe.dofs_per_cell;
    const unsigned int nPointsQuad=quadratureFormula.size();
    const unsigned int nPointsFaceQuad=faceQuadratureFormula.size();


    FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
    Vector<double> cellRhs(dofsPerCell);

    std::vector<types::global_dof_index> indices(dofsPerCell);
    std::vector<Point<2>> faceQuadPoints(nPointsFaceQuad);
    std::vector<Point<2>> quadPoints(nPointsQuad);

    for (const auto &cell: dofHandler.active_cell_iterators()) {
        if (cell -> is_locally_owned()) {

            cellMatrix = 0.0;
            cellRhs = 0.0;
            feValues.reinit(cell);
            quadPoints = feValues.get_quadrature_points();

            for (unsigned int q=0; q<nPointsQuad; ++q) {
                for(unsigned int i=0; i<dofsPerCell; ++i){
                    cellRhs(i) += 1.0*feValues.shape_value(i,q)*feValues.JxW(q);
                    for(unsigned int j = 0; j<dofsPerCell; ++j){
                        cellMatrix(i,j) += feValues.shape_grad(i,q)*feValues.shape_grad(j,q)*feValues.JxW(q);
                    }
                }
            }

            for (unsigned int faceNumber=0; faceNumber<dealii::GeometryInfo<2>::faces_per_cell; ++faceNumber){
                if (cell->face(faceNumber)->at_boundary()) {
                    feFaceValues.reinit(cell, faceNumber);
                    faceQuadPoints=feFaceValues.get_quadrature_points();
                    for (unsigned int q=0; q<nPointsFaceQuad; ++q){
                        double x = faceQuadPoints[q][0];
                        double y = faceQuadPoints[q][1];
                        for(unsigned int i=0; i<dofsPerCell; ++i){
                            cellRhs(i) += -feFaceValues.shape_value(i,q)*feFaceValues.JxW(q)*feFaceValues.normal_vector(q)[1]*
                                    (y*y - x*x*poisson/(1+poisson));

                        }
                    }

                }
            }

            cell -> get_dof_indices(indices);
            constraints.distribute_local_to_global(cellMatrix, cellRhs, indices, systemMatrix, systemRhs);
        }
    }
    systemMatrix.compress(VectorOperation::add);
    systemRhs.compress(VectorOperation::add);
}

double solver::getInertia() {
    TimerOutput::Scope t(timer, "assemble");

    const QGauss<2> quadratureFormula(fe.degree + 1);

    FEValues<2> feValues(fe, quadratureFormula, update_quadrature_points | update_JxW_values);

    const unsigned int nPointsQuad=quadratureFormula.size();

    std::vector<Point<2>> quadPoints(nPointsQuad);
    double inertia{0.0};

    for (const auto &cell: dofHandler.active_cell_iterators()) {
        if (cell -> is_locally_owned()) {
            feValues.reinit(cell);
            quadPoints = feValues.get_quadrature_points();
            for (unsigned int q=0; q<nPointsQuad; ++q){
                inertia += quadPoints[q][1]*quadPoints[q][1]*feValues.JxW(q);
            }
        }
    }
    return Utilities::MPI::sum(inertia, communicator);
}

void solver::solve() {
    TimerOutput::Scope t(timer, "solve");

    LinearAlgebraPETSc::MPI::Vector distributedSolution(ownedDofs, communicator);

    SolverControl solverControl(dofHandler.n_dofs(), 1e-5);
    LinearAlgebraPETSc::SolverCG solver(solverControl, communicator);

    LinearAlgebraPETSc::MPI::PreconditionJacobi precond;
    precond.initialize(systemMatrix);

    solver.solve(systemMatrix, distributedSolution, systemRhs, precond);

    constraints.distribute(distributedSolution);
    relevantSolution=distributedSolution;
}

void solver::refineGrid() {
    TimerOutput::Scope t(timer, "refine");

    Vector<float> estimateErrorPerCell(mesh.n_active_cells());
    KellyErrorEstimator<2>::estimate(dofHandler, dealii::QGauss<1>(fe.degree+1), std::map<dealii::types::boundary_id,
                                       const dealii::Function<2> *>(), relevantSolution, estimateErrorPerCell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(mesh, estimateErrorPerCell, 1.0/3.0, 1.0/30.0);
    mesh.execute_coarsening_and_refinement();
}

void solver::makeGrid() {
    GridGenerator::hyper_cube(mesh);
    mesh.refine_global(4);
}

class TauPostProcessor : public DataPostprocessorVector<2>
{
public:
    double inertia;
    TauPostProcessor (const double &inertia): DataPostprocessorVector<2> ("tau", update_gradients | update_quadrature_points), inertia(inertia){}

    virtual void evaluate_scalar_field(const dealii::DataPostprocessorInputs::Scalar<2> &inputData, std::vector<dealii::Vector<double> > &computedQuantities) const {
        for (unsigned int p=0; p<inputData.solution_gradients.size(); ++p) {
        computedQuantities[p][0] = (0.5/inertia)*inputData.solution_gradients[p][0];
        computedQuantities[p][1] = (0.5/inertia)*(inputData.solution_gradients[p][1]+(solver::poisson/(solver::poisson+1))*inputData.evaluation_points[p][0]*inputData.evaluation_points[p][0]-
                inputData.evaluation_points[p][1]*inputData.evaluation_points[p][1]);
        }
    }
};

void solver::outputResults(const unsigned int cycle){
    TimerOutput::Scope t(timer, "output");

    DataOut<2> dataOut;
    dataOut.attach_dof_handler(dofHandler);

    TauPostProcessor tau(this->getInertia());

    dataOut.add_data_vector(relevantSolution, "solution");
    dataOut.add_data_vector(relevantSolution, tau);

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
