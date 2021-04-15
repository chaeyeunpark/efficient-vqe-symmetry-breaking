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

Eigen::SparseMatrix<qunn::cx_double> single_pauli(const uint32_t N, const uint32_t idx, 
		const Eigen::SparseMatrix<qunn::cx_double>& m)
{
	edp::LocalHamiltonian<qunn::cx_double> lh(N, 2);
	lh.addOneSiteTerm(idx, m);
	return edp::constructSparseMat<qunn::cx_double>(1<<N, lh);
}

Eigen::SparseMatrix<qunn::cx_double> identity(const uint32_t N)
{
	std::vector<Eigen::Triplet<qunn::cx_double>> triplets;
	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		triplets.emplace_back(n, n, 1.0);
	}
	Eigen::SparseMatrix<qunn::cx_double> m(1<<N,1<<N);
	m.setFromTriplets(triplets.begin(), triplets.end());
	return m;
}


Eigen::SparseMatrix<qunn::cx_double> cluster_ham(uint32_t N, double h)
{
	using namespace qunn;
	Eigen::SparseMatrix<cx_double> ham(1<<N, 1<<N);
	for(uint32_t k = 0; k < N-2; k++)
	{
		Eigen::SparseMatrix<cx_double> term = identity(N);
		term = term*single_pauli(N, k, pauli_z().cast<cx_double>());
		term = term*single_pauli(N, (k+1)%N, pauli_x().cast<cx_double>());
		term = term*single_pauli(N, (k+2)%N, pauli_z().cast<cx_double>());

		ham += -term;
	}

	edp::LocalHamiltonian<double> lh(N, 2);
	for(uint32_t k = 0; k < N; k++)
	{
		lh.addOneSiteTerm(k, pauli_x());
	}
	ham -= h*edp::constructSparseMat<cx_double>(1<<N, lh);

	return ham;
}

Eigen::SparseMatrix<qunn::cx_double> x_even_op(uint32_t N)
{
	using namespace qunn;
	Eigen::SparseMatrix<cx_double> term = identity(N);
	for(uint32_t k = 0; k < N; k += 2)
		term = term*single_pauli(N, k, pauli_x().cast<cx_double>());
	return term;
}
Eigen::SparseMatrix<qunn::cx_double> x_odd_op(uint32_t N)
{
	using namespace qunn;
	Eigen::SparseMatrix<cx_double> term = identity(N);
	for(uint32_t k = 1; k < N; k += 2)
		term = term*single_pauli(N, k, pauli_x().cast<cx_double>());
	return term;
}


Eigen::SparseMatrix<qunn::cx_double> z_even_op(uint32_t N)
{
	using namespace qunn;
	Eigen::SparseMatrix<cx_double> term = identity(N);
	for(uint32_t k = 0; k < N; k += 2)
		term = term*single_pauli(N, k, pauli_z().cast<cx_double>());
	return term;
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
	const uint32_t total_epochs = 3000;
	const double h = 0.0;

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
	const double grad_clip = param_in.value("grad_clip", 1e+8);
	const double alpha = param_in.value("alpha", 0.8);
	const bool use_ngd = param_in.value("use_ngd", true);

	param_out["parameters"] = nlohmann::json({
		{"N", N},
		{"depth", depth},
		{"sigma", sigma},
		{"grad_clip", grad_clip},
		{"alpha", alpha},
		{"use_ngd", use_ngd},
		{"h", h}
	});


	const int num_threads = get_num_threads();
	std::cerr << "Processing using " << num_threads << " threads." << std::endl;
	tbb::global_control c(tbb::global_control::max_allowed_parallelism, 
			num_threads);


    std::random_device rd;
    std::default_random_engine re{rd()};

    Circuit circ(1 << N);

	std::vector<std::map<uint32_t, Pauli>> ti_zxz;

	for(uint32_t k = 0; k < N-2; ++k)
	{
		std::map<uint32_t, Pauli> term;
		term[k] = Pauli('Z');
		term[(k+1)%N] = Pauli('X');
		term[(k+2)%N] = Pauli('Z');

		ti_zxz.emplace_back(std::move(term));
	}

	Eigen::VectorXd z_even = Eigen::VectorXd::Zero(1u << N);
	for(uint32_t n = 0; n < (1u << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; k+=2)
		{
			elt += 1-2*((n >> k) & 1);
		}
		z_even(n) = elt;
	}

	Eigen::VectorXd z_odd = Eigen::VectorXd::Zero(1u << N);
	for(uint32_t n = 0; n < (1u << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 1; k < N; k+=2)
		{
			elt += 1-2*((n >> k) & 1);
		}
		z_odd(n) = elt;
	}

	auto zxz_ham = qunn::SumPauliString(N, ti_zxz);
	auto x_all_ham = qunn::SumLocalHam(N, qunn::pauli_x().cast<cx_double>());
	auto z_even_ham = qunn::DiagonalOperator(z_even);
	auto z_odd_ham = qunn::DiagonalOperator(z_odd);

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<qunn::SumPauliStringHamEvol>(zxz_ham));
		circ.add_op_right(std::make_unique<qunn::ProductHamEvol>(x_all_ham));
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(z_even_ham));
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(z_odd_ham));
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

	//parameters[3] += M_PI/2;

	for(uint32_t idx = 0; idx < parameters.size(); idx += 4)
	{
		parameters[idx + 2] += 2*M_PI/depth;
		parameters[idx + 3] += 2*M_PI/depth;
	}

    Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(1 << N);
    ini /= sqrt(1 << N);

    const auto ham = cluster_ham(N, h);
	const auto x_even = x_even_op(N);
	const auto x_odd = x_even_op(N);

	std::cout.precision(10);

    circ.set_input(ini);

	auto optimizer = OptimizerFactory::getInstance().createOptimizer(
			param_in["optimizer"]);

    for(uint32_t epoch = 0; epoch < total_epochs; ++epoch)
    {
		/*
		double alpha_1_t = 0.1*alpha_1;
		double alpha_2_t = 0.1*alpha_2;

		if(epoch > 800)
		{
			alpha_1_t += alpha_1*std::min(1.0, int((epoch-800)/200)*0.1);
			alpha_2_t += alpha_2*std::min(1.0, int((epoch-800)/200)*0.1);
		}
		*/

		double alpha_1_t = alpha;
		double alpha_2_t = alpha;


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


        Eigen::VectorXd egrad = (output.transpose()*ham*grads).real();
        Eigen::VectorXd x_even_grad = (output.transpose()*x_even*grads).real();
        Eigen::VectorXd x_odd_grad = (output.transpose()*x_odd*grads).real();
        double energy = real(cx_double(output.transpose()*ham*output));
		double x_even_expectation = real(cx_double(output.transpose()*x_even*output));
		double x_odd_expectation = real(cx_double(output.transpose()*x_even*output));

        std::cout << epoch << "\t" << energy << "\t" << x_even_expectation <<
			"\t" << x_odd_expectation << "\t" << egrad.norm() << "\t" << output.norm() << std::endl;

		Eigen::VectorXd total_grad = (egrad + alpha_1_t*x_even_grad + alpha_2_t*x_odd_grad);
		
		double grad_norm = total_grad.norm();
		if(grad_norm > grad_clip)
		{
			total_grad *= (grad_clip/grad_norm);
		}
		Eigen::VectorXd v;

		if(use_ngd)
		{
			Eigen::MatrixXd fisher = (grads.adjoint()*grads).real();
			double lambda = 1e-3;
			fisher += lambda*Eigen::MatrixXd::Identity(parameters.size(), parameters.size());

			Eigen::LLT<Eigen::MatrixXd> llt_fisher(fisher);
			v = llt_fisher.solve(total_grad);
		}
		else{
			v = total_grad;
		}
		Eigen::VectorXd opt = optimizer->getUpdate(v);

        for(uint32_t k = 0; k < parameters.size(); ++k)
        {
            parameters[k] += opt(k);
        }
    }
	{
		std::ofstream weight_out("final_weight.dat");
		for(auto& p: parameters)
		{
			weight_out << p.value() << "\t";
		}
	}

    return 0;
}