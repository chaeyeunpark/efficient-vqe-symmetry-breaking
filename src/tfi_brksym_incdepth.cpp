#include <random>
#include <iostream>
#include <cmath>
#include <regex>
#include <fstream>

#include <tbb/tbb.h>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "Circuit.hpp"

#include "operators.hpp"
#include "Optimizers/OptimizerFactory.hpp"

Eigen::SparseMatrix<double> tfi_ham(const uint32_t N, double h)
{
    edp::LocalHamiltonian<double> ham_ct(N, 2);
    for(uint32_t k = 0; k < N; ++k)
    {
        ham_ct.addTwoSiteTerm(std::make_pair(k, (k+1) % N), qunn::pauli_zz());
        ham_ct.addOneSiteTerm(k, h*qunn::pauli_x());
    }
    return -edp::constructSparseMat<double>(1 << N, ham_ct);
}

int get_num_threads()
{
	const char* p = getenv("TBB_NUM_THREADS");
	if(!p)
		return tbb::this_task_arena::max_concurrency();
	return atoi(p);
}

int main(int argc, char *argv[])
{
    using namespace qunn;
    using std::sqrt;
	const uint32_t total_epochs = 2000;

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
	const bool centering = param_in.value("centering", true);
	const double learning_rate = param_in.value("learning_rate", 2.0e-2);
	const std::string ini_path = param_in.value("ini_path", "");

	if(ini_path.empty())
	{
		std::cerr << "You must provide ini_path" << std::endl;
		return 1;
	}

	{
		std::smatch depth_match;
		std::regex depth_search("d(\\d+)");
		std::regex_search(ini_path, depth_match, depth_search);
		int depth_ini_path = std::stoi(depth_match[1]);

		if((depth_ini_path + 1) != depth)
		{
			std::cerr << "depth must be larger one than that provided in ini_path" << std::endl;
			return 1;
		}
	}

	param_out["parameters"] = nlohmann::json({
		{"N", N},
		{"depth", depth},
		{"sigma", sigma},
		{"centering", centering},
		{"learning_rate", learning_rate},
		{"ini_path", ini_path}
	});


	const int num_threads = get_num_threads();
	std::cerr << "Processing using " << num_threads << " threads." << std::endl;
	tbb::global_control c(tbb::global_control::max_allowed_parallelism, 
			num_threads);


    std::random_device rd;
    std::default_random_engine re{rd()};

    Circuit circ(1 << N);

	Eigen::VectorXd zz_all(1<<N);
	Eigen::VectorXd z_all(1<<N);
	
	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; ++k)
		{
			int z0 = 1-2*((n >> k) & 1);
			int z1 = 1-2*((n >> ((k+1)%N)) & 1);
			elt += z0*z1;
		}
		zz_all(n) = elt;
	}

	
	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; ++k)
		{
			int z0 = 1-2*((n >> k) & 1);
			elt += z0;
		}
		z_all(n) = elt;
	}

	auto zz_all_ham = qunn::DiagonalOperator(zz_all, "zz all");
	auto x_all_ham = qunn::SumLocalHam(N, qunn::pauli_x().cast<cx_double>(), "x all");
	auto z_all_ham = qunn::DiagonalOperator(z_all, "z all");

	for(uint32_t p = 0; p < depth+1; ++p)
	{
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(zz_all_ham));
		circ.add_op_right(std::make_unique<qunn::ProductHamEvol>(x_all_ham));
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(z_all_ham));
	}

	param_out["circuit"] = circ.desc();
	{
		std::ofstream fout("param_out.json");
		fout << param_out << std::endl;
	}

    auto parameters = circ.parameters();

	{
		std::cerr << "load initial parameters from " << ini_path << std::endl;
		std::normal_distribution<double> ndist(0., sigma);
		for(auto& p: parameters)
		{
			p = ndist(re);
		}

		std::ifstream init_fin(ini_path);
		double v;
		for(uint32_t idx = 0; idx < depth/2; ++idx)
		{
			init_fin >> v;
			parameters[3*idx] += v;
			init_fin >> v;
			parameters[3*idx+1] += v;
			init_fin >> v;
			parameters[3*idx+2] += v;
		}
		for(uint32_t idx = depth/2 + 1; idx < depth; ++idx)
		{
			init_fin >> v;
			parameters[3*idx] += v;
			init_fin >> v;
			parameters[3*idx+1] += v;
			init_fin >> v;
			parameters[3*idx+2] += v;
		}
	}

	{
		std::ofstream initial_weight("initial_weight.dat");
		for(auto& p: parameters)
		{
			initial_weight << p.value() << "\t";
		}
		initial_weight.close();
	}

    const auto ham = tfi_ham(N, 0.5);

    Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(1 << N);
    ini /= sqrt(1 << N);
    circ.set_input(ini);

	std::cout.precision(10);
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

		std::cout << epoch << "\t" << energy << "\t" << 
			egrad.norm() << "\t" << output.norm() << std::endl;

		Eigen::MatrixXd fisher = (grads.adjoint()*grads).real();
		if (centering)
		{
			Eigen::RowVectorXcd o = (output.adjoint()*grads);
			fisher -= (o.adjoint()*o).real();
		}
		double lambda = std::max(100.0*std::pow(0.9, epoch), 1e-3);
		fisher += lambda*Eigen::MatrixXd::Identity(parameters.size(), parameters.size());

		Eigen::LLT<Eigen::MatrixXd> llt_fisher(fisher);
		Eigen::VectorXd opt_v = -learning_rate*llt_fisher.solve(egrad);

		for(uint32_t k = 0; k < parameters.size(); ++k)
		{
			parameters[k] += opt_v(k);
		}
	}

	{
		std::ofstream final_weight("final_weight.dat");
		for(const auto& p: parameters)
		{
			final_weight << p.value() << "\t";
		}
		final_weight.close();
	}

    return 0;
}
