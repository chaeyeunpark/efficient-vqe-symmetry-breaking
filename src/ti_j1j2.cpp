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

Eigen::SparseMatrix<double> get_SS()
{
	Eigen::SparseMatrix<double> m(4,4);

	m.insert(0, 0) = 1;
	m.insert(1, 1) = -1;
	m.insert(2, 2) = -1;
	m.insert(3, 3) = 1;

	m.insert(1, 2) = 2;
	m.insert(2, 1) = 2;

	m.makeCompressed();
	return m;
}

Eigen::SparseMatrix<qunn::cx_double> j1j2_ham(uint32_t N, double j2)
{
	using namespace qunn;

	edp::LocalHamiltonian<double> lh(N, 2);

	for(uint32_t n = 0; n < N; ++n)
	{
		lh.addTwoSiteTerm(std::make_pair(n, (n+1)%N), get_SS());
	}
	for(uint32_t n = 0; n < N; ++n)
	{
		lh.addTwoSiteTerm(std::make_pair(n, (n+2)%N), j2*get_SS());
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

qunn::SumPauliString xx_yy_nn(const uint32_t N)
{
	using namespace qunn;
	std::vector<std::map<uint32_t, Pauli>> paulis;

	for(uint32_t k = 0; k < N; ++k)
	{
		std::map<uint32_t, Pauli> term;
		term[k] = Pauli('X');
		term[(k+1)%N] = Pauli('X');

		paulis.emplace_back(std::move(term));
	}
	for(uint32_t k = 0; k < N; ++k)
	{
		std::map<uint32_t, Pauli> term;
		term[k] = Pauli('Y');
		term[(k+1)%N] = Pauli('Y');

		paulis.emplace_back(std::move(term));
	}

	return qunn::SumPauliString(N, paulis);
}

qunn::SumPauliString xx_yy_nnn(const uint32_t N)
{
	using namespace qunn;
	std::vector<std::map<uint32_t, Pauli>> paulis;

	for(uint32_t k = 0; k < N; ++k)
	{
		std::map<uint32_t, Pauli> term;
		term[k] = Pauli('X');
		term[(k+2)%N] = Pauli('X');

		paulis.emplace_back(std::move(term));
	}
	for(uint32_t k = 0; k < N; ++k)
	{
		std::map<uint32_t, Pauli> term;
		term[k] = Pauli('Y');
		term[(k+2)%N] = Pauli('Y');

		paulis.emplace_back(std::move(term));
	}
	return qunn::SumPauliString(N, paulis);
}

qunn::DiagonalOperator zz_nn(const uint32_t N)
{
	using namespace qunn;
	Eigen::VectorXd zz(1<<N);

	for(uint32_t n = 0; n < (1u << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; ++k)
		{
			int z0 = 1-2*((n >> k) & 1);
			int z1 = 1-2*((n >> ((k+1)%N)) & 1);
			elt += z0*z1;
		}
		zz(n) = elt;
	}

	return DiagonalOperator(zz);
}

qunn::DiagonalOperator zz_nnn(const uint32_t N)
{
	using namespace qunn;
	Eigen::VectorXd zz(1<<N);

	for(uint32_t n = 0; n < (1u << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; ++k)
		{
			int z0 = 1-2*((n >> k) & 1);
			int z1 = 1-2*((n >> ((k+2)%N)) & 1);
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
	const double j2 = 0.0;

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
		{"j2", j2}
	});


	const int num_threads = get_num_threads();
	std::cerr << "Processing using " << num_threads << " threads." << std::endl;
	tbb::global_control c(tbb::global_control::max_allowed_parallelism, 
			num_threads);


    std::random_device rd;
    std::default_random_engine re{rd()};

    Circuit circ(1 << N);

	auto xx_yy_nn_ham = xx_yy_nn(N);
	auto zz_nn_ham = zz_nn(N);
	auto xx_yy_nnn_ham = xx_yy_nnn(N);
	auto zz_nnn_ham = zz_nnn(N);
	auto x_all_ham = qunn::SumLocalHam(N, qunn::pauli_x().cast<cx_double>());

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<qunn::SumPauliStringHamEvol>(xx_yy_nn_ham));
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(zz_nn_ham));

		circ.add_op_right(std::make_unique<qunn::SumPauliStringHamEvol>(xx_yy_nnn_ham));
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
    const auto ham = j1j2_ham(N, j2);

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

		Eigen::MatrixXd fisher = (grads.adjoint()*grads).real();
		double lambda = std::max(100.0*std::pow(0.9, epoch), 1e-3);
		fisher += lambda*Eigen::MatrixXd::Identity(parameters.size(), parameters.size());

        Eigen::VectorXd egrad = (output.adjoint()*ham*grads).real();
        double energy = real(cx_double(output.adjoint()*ham*output));

        std::cout << energy << "\t" << egrad.norm() << "\t" << output.norm() << std::endl;

		Eigen::VectorXd opt = -learning_rate*fisher.inverse()*egrad;

        for(uint32_t k = 0; k < parameters.size(); ++k)
        {
            parameters[k] += opt(k);
        }
    }

    return 0;
}
