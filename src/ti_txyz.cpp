#include <random>
#include <iostream>
#include <cmath>

#include <fstream>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "Circuit.hpp"

#include "operators.hpp"
#include "Optimizers/OptimizerFactory.hpp"

#include <tbb/tbb.h>

Eigen::SparseMatrix<double> get_xyz(double a, double b)
{
	Eigen::SparseMatrix<double> m(4,4);

	m.insert(0, 0) = -1;
	m.insert(1, 1) = 1;
	m.insert(2, 2) = 1;
	m.insert(3, 3) = -1;

	m.insert(0, 3) = b-a;
	m.insert(1, 2) = -a-b;
	m.insert(2, 1) = -a-b;
	m.insert(3, 0) = b-a;

	m.makeCompressed();
	return m;
}

Eigen::SparseMatrix<qunn::cx_double> txyz_ham(uint32_t N, double a, double b)
{
	using namespace qunn;

	edp::LocalHamiltonian<double> lh(N, 2);

	for(uint32_t n = 0; n < N; ++n)
	{
		lh.addTwoSiteTerm(std::make_pair(n, (n+1)%N), get_xyz(a, b));
	}
	for(uint32_t n = 0; n < N; ++n)
	{
		lh.addTwoSiteTerm(std::make_pair(n, (n+2)%N), get_xyz(b, a));
	}

	return edp::constructSparseMat<cx_double>(1<<N, lh);
}

int get_num_threads()
{
	const char* p = getenv("TBB_NUM_THREADS");
	if(!p)
		return tbb::this_task_arena::max_concurrency();
	return atoi(p);
}

qunn::SumPauliString xx_term(const uint32_t N, int j)
{
	using namespace qunn;
	std::vector<std::map<uint32_t, Pauli>> paulis;

	for(uint32_t k = 0; k < N; ++k)
	{
		std::map<uint32_t, Pauli> term;
		term[k] = Pauli('X');
		term[(k+j)%N] = Pauli('X');

		paulis.emplace_back(std::move(term));
	}
	return qunn::SumPauliString(N, paulis);
}

qunn::SumPauliString yy_term(const uint32_t N, int j)
{
	using namespace qunn;
	std::vector<std::map<uint32_t, Pauli>> paulis;
	for(uint32_t k = 0; k < N; ++k)
	{
		std::map<uint32_t, Pauli> term;
		term[k] = Pauli('Y');
		term[(k+j)%N] = Pauli('Y');

		paulis.emplace_back(std::move(term));
	}

	return qunn::SumPauliString(N, paulis);
}

qunn::DiagonalOperator zz_term(const uint32_t N, int j)
{
	using namespace qunn;
	Eigen::VectorXd zz(1<<N);

	for(uint32_t n = 0; n < (1u << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; ++k)
		{
			int z0 = 1-2*((n >> k) & 1);
			int z1 = 1-2*((n >> ((k+j)%N)) & 1);
			elt += z0*z1;
		}
		zz(n) = elt;
	}

	return DiagonalOperator(zz);
}

int main(int argc, char *argv[])
{
    using namespace qunn;
    using std::sqrt;
	const uint32_t total_epochs = 2000;
	const double a = 0.4;
	const double b = 3.0;

	nlohmann::json param_in;
	nlohmann::json param_out;
	if(argc != 2)
	{
		printf("Usage: %s [param_in.json]\n", argv[0]);
		return 1;
	}
	{
		std::ifstream fin(argv[1]);
		fin >> param_in;
	}
    const uint32_t N = param_in.at("N").get<uint32_t>();
    const uint32_t depth = param_in.at("depth").get<uint32_t>();
	const double sigma = param_in.at("sigma").get<double>();
	const double learning_rate = param_in.value("learning_rate", 1.0e-2);

	param_out["parameters"] = nlohmann::json({
		{"N", N},
		{"depth", depth},
		{"sigma", sigma},
		{"learning_rate", learning_rate},
	});


	const int num_threads = get_num_threads();
	std::cerr << "Processing using " << num_threads << " threads." << std::endl;
	tbb::global_control c(tbb::global_control::max_allowed_parallelism, 
			num_threads);


    std::random_device rd;
    std::default_random_engine re{rd()};

    Circuit circ(1 << N);

	auto xx_nn_ham = xx_term(N, 1);
	auto xx_nnn_ham = xx_term(N, 2);
	auto yy_nn_ham = yy_term(N, 1);
	auto yy_nnn_ham = yy_term(N, 2);
	auto zz_nn_ham = zz_term(N, 1);
	auto zz_nnn_ham = zz_term(N, 2);
	auto x_all_ham = qunn::SumLocalHam(N, qunn::pauli_x().cast<cx_double>());

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<qunn::SumPauliStringHamEvol>(xx_nn_ham));
		circ.add_op_right(std::make_unique<qunn::SumPauliStringHamEvol>(yy_nn_ham));
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(zz_nn_ham));

		circ.add_op_right(std::make_unique<qunn::SumPauliStringHamEvol>(xx_nnn_ham));
		circ.add_op_right(std::make_unique<qunn::SumPauliStringHamEvol>(yy_nnn_ham));
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(zz_nnn_ham));

		circ.add_op_right(std::make_unique<qunn::ProductHamEvol>(x_all_ham));
	}

	param_out["circuit"] = circ.desc();
	{
		std::ofstream fout("param_out.json");
		fout << param_out << std::endl;
	}

    auto parameters = circ.parameters();

    std::normal_distribution<double> ndist(0., sigma);
    for(auto& p: parameters)
    {
        p = ndist(re);
    }

    Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(1 << N);
    ini /= sqrt(1 << N);
    const auto ham = txyz_ham(N, a, b);

	std::cout.precision(10);

    circ.set_input(ini);

    for(uint32_t epoch = 0; epoch < total_epochs; ++epoch)
    {
        circ.clear_evaluated();
        Eigen::VectorXcd output = *circ.output();
    	for(auto& p: parameters)
		{
			p.zero_grad();
		}

        circ.derivs();

        Eigen::MatrixXcd grads(1 << N, parameters.size());

        for(uint32_t k = 0; k < parameters.size(); ++k)
        {
            grads.col(k) = *parameters[k].grad();
        }

        Eigen::VectorXd egrad = (output.adjoint()*ham*grads).real();
        double energy = real(cx_double(output.adjoint()*ham*output));
		
		Eigen::VectorXd opt;
		{
			Eigen::MatrixXd fisher = (grads.adjoint()*grads).real();
			double lambda = std::max(100.0*std::pow(0.9, epoch), 1e-3);
			fisher += lambda*Eigen::MatrixXd::Identity(parameters.size(), parameters.size());
			Eigen::LLT<Eigen::MatrixXd> llt_fisher(fisher);
			opt = -learning_rate*llt_fisher.solve(egrad);
		}

        std::cout << energy << "\t" << egrad.norm() << "\t" << output.norm() << std::endl;


        for(uint32_t k = 0; k < parameters.size(); ++k)
        {
            parameters[k] += opt(k);
        }
    }

    return 0;
}
