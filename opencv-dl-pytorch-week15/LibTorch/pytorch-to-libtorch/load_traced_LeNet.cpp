#include <ostream>
#include<torch/script.h>
#include<iostream>
#include<vector>

int main()
{
	torch::jit::script::Module LeNet = torch::jit::load("./LeNet.pt");
	auto input = torch::randn({3, 1, 28, 28});
	std::cout<<"Successfully loaded the traced model into Libtorch ."<<std::endl;
	std::vector<torch::jit::IValue> jit_input;
	jit_input.push_back(input);
	std::cout<<"Inferring on a random input "<<std::endl;
	auto output = LeNet.forward(jit_input).toTensor();

	std::cout<<"Output size is "<<output.sizes()<<std::endl;
	





return 0;
}
