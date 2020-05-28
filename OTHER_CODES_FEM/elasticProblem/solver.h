#ifndef SOLVER_H
#define SOLVER_H

#include "include_dealii.h"

template <int dim>
class solver {
public:
    solver();
    ~solver() {dofHandler.clear();};
    void run() {
        this -> makeGrid();
        for(int cycle=0; cycle<5;cycle++) {
            std::cout << "   Number of active cells:       "
                      << mesh.n_active_cells() << std::endl;

            this -> setupSystem();
            this -> assembleSystem();
            this -> solve();
            if (cycle != 4)
                this -> refineGrid();

        }
        this -> outputResults();
    };
private:
    void makeGrid();
    void setupSystem();
    void assembleSystem();
    void solve();
    void refineGrid();
    void outputResults();
    double getStrain(const dealii::Tensor<2, dim> &grad, const dealii::Tensor<2, dim> &symmDisp);

private:
    dealii::Triangulation<dim> mesh;
    dealii::DoFHandler<dim> dofHandler;
    dealii::FESystem<dim> fe;
    dealii::AffineConstraints<double> constraints;

    dealii::SparsityPattern sparsityPattern;
    dealii::SparseMatrix<double> systemMatrix;
    dealii::Vector<double> systemRhs;
    dealii::Vector<double> solution;

    dealii::Tensor<4, dim> C;
};

template<int dim>
solver<dim>::solver():dofHandler(mesh), fe(dealii::FE_Q<dim>(2), dim){
    double lambda{1.0};
    double mu{1.0};
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          for (unsigned int k = 0; k < dim; ++k)
            for (unsigned int l = 0; l < dim; ++l)
              C[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                                 ((i == l) && (j == k) ? mu : 0.0) +
                                 ((i == j) && (k == l) ? lambda : 0.0));
}

template<int dim>
double solver<dim>::getStrain(const dealii::Tensor<2, dim> &grad, const dealii::Tensor<2, dim> &symmDisp){
    double temp{0.0};
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
              temp += grad[i][j]*C[i][j][k][l]*symmDisp[k][l];
    return temp;
}

template <int dim>
void solver<dim>::makeGrid() {
    dealii::Point<dim> p1{0,0,0};
    dealii::Point<dim> p2{1,0.25, 0.25};

    dealii::GridGenerator::hyper_rectangle(mesh, p1, p2, true);
    mesh.refine_global(2);
}

template <int dim>
void solver<dim>::setupSystem() {
    dofHandler.distribute_dofs(fe);

    constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dofHandler, constraints);
    dealii::DoFTools::make_zero_boundary_constraints(dofHandler,0,constraints);
    constraints.close();

    dealii::DynamicSparsityPattern dsp(dofHandler.n_dofs(), dofHandler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dofHandler,
                                            dsp,
                                            constraints,
                                            false);
    sparsityPattern.copy_from(dsp);
    systemMatrix.reinit(sparsityPattern);
    solution.reinit(dofHandler.n_dofs());
    systemRhs.reinit(dofHandler.n_dofs());
}
template <int dim>
void solver<dim>::assembleSystem() {
    dealii::QGauss<dim> quad(fe.degree + 1);
    dealii::QGauss<dim-1> faceQuad(fe.degree+1);

    dealii::FEValues<dim> feVal(fe, quad,
                                dealii::update_values |
                                dealii::update_quadrature_points |
                                dealii::update_gradients |
                                dealii::update_JxW_values);
    dealii::FEFaceValues<dim> feFaceValues(fe, faceQuad,
            dealii::update_values | dealii::update_normal_vectors | dealii::update_quadrature_points | dealii::update_JxW_values);

    const int dofsCell = fe.dofs_per_cell;
    const int numQuad = quad.size();
    const int numFaceQuad = faceQuad.size();

    dealii::FullMatrix<double> cellMatrix(dofsCell,dofsCell);
    dealii::Vector<double> cellRhs(dofsCell);

    std::vector<dealii::types::global_dof_index> localDofIndices(dofsCell);

    const dealii::FEValuesExtractors::Vector displacements(0);

    for(const auto &cell : dofHandler.active_cell_iterators()) {
        cellMatrix=0.0;
        cellRhs=0.0;

        feVal.reinit(cell);        

        for (int q=0; q<numQuad; q++) {
            for(int i=0; i<dofsCell; i++) {
                for(int j=0; j<dofsCell; j++) {
                    cellMatrix(i,j)+=feVal.JxW(q)*getStrain(feVal[displacements].gradient(i,q), feVal[displacements].symmetric_gradient(j,q));

                }
            }
        }

        for (unsigned int faceNumber=0; faceNumber<dealii::GeometryInfo<2>::faces_per_cell; ++faceNumber){
            if (cell->face(faceNumber)->at_boundary() && cell->face(faceNumber)->boundary_id()==1) {
                feFaceValues.reinit(cell, faceNumber);
                for (unsigned int q=0; q<numFaceQuad; ++q){
                    for(unsigned int i=0; i<dofsCell; ++i){
                        cellRhs(i) += 3.0*std::sin(M_PI*feVal.quadrature_point(q)[1]/0.25)*feFaceValues.JxW(q)*feFaceValues[displacements].value(i,q)[0];
                    }
                }

            }
        }

        cell -> get_dof_indices(localDofIndices);
        constraints.distribute_local_to_global(cellMatrix,cellRhs, localDofIndices, systemMatrix, systemRhs);
    }

}

template <int dim>
void solver<dim>::solve() {
    dealii::SolverControl solverControl(1000,1e-7);
    dealii::SolverCG<> cg(solverControl);
    dealii::PreconditionSSOR<> precond;
    precond.initialize(systemMatrix, 1.2);
    cg.solve(systemMatrix, solution, systemRhs, precond);
    constraints.distribute(solution);
}

template <int dim>
void solver<dim>::refineGrid() {
    dealii::Vector<float> estimatedError(mesh.n_active_cells());
    dealii::KellyErrorEstimator<dim>::estimate(
      dofHandler,
      dealii::QGauss<dim - 1>(fe.degree + 1),
      std::map<dealii::types::boundary_id, const dealii::Function<dim> *>(),
      solution,
      estimatedError);
    dealii::GridRefinement::refine_and_coarsen_fixed_number(mesh,estimatedError,0.3, 0.03);
    mesh.execute_coarsening_and_refinement();
}

template<int dim>
class StressPostProcessor: public dealii::DataPostprocessorScalar<dim> {
public:
    StressPostProcessor () : dealii::DataPostprocessorScalar<dim> ("strain", dealii::update_gradients){}
    virtual void evaluate_vector_field( const dealii::DataPostprocessorInputs::Vector<dim> &input_data, std::vector<dealii::Vector<double>> &computed_quantities) const {
    for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p) {
        dealii::Tensor<2,dim> strain;
        for (unsigned int d=0; d<dim; ++d)
          for (unsigned int e=0; e<dim; ++e)
            strain[d][e] = (input_data.solution_gradients[p][d][e] + input_data.solution_gradients[p][e][d]) / 2;
        computed_quantities[p](0)=strain.norm();
      }
  }

};

template <int dim>
void solver<dim>::outputResults() {
    std::vector<std::string> solutionNames(dim,"u");
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>interpretation(dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
    StressPostProcessor<dim> stress;

    dealii::DataOut<dim> dataOut;
    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(dofHandler, solution,solutionNames,interpretation);
    dataOut.add_data_vector(solution, stress);
    dataOut.build_patches();
    std::ofstream output("solution.vtu");
    dataOut.write_vtu(output);
}





#endif // SOLVER_H
