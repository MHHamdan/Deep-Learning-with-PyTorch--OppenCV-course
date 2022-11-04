#pragma once
#include <torch/torch.h>
#include<iostream>

using namespace std;
using namespace torch::indexing;

// Create a struct named 'DemoNet' inhereting from 'torch::nn::Module'
struct DemoNetImpl : torch::nn::Module
{   
	// declare the required the layers
    torch::nn::Linear fc1 , fc2 ;
    torch::nn::BatchNorm2d bn1 ;
    torch::nn::Conv2d conv1, conv2 ;
    torch::nn::MaxPool2d mxpool1 ;

    //create the constructor
    DemoNetImpl(int64_t N): 
    	//initialize the attributes of the layers.
        conv1(torch::nn::Conv2dOptions(1, 32, 3).stride(1)),
        conv2(torch::nn::Conv2dOptions(32,64,3).stride(1)),
        fc1(torch::nn::Linear(9216, 128)),
        fc2(torch::nn::Linear(128, N)),
        bn1(torch::nn::BatchNorm2d(32)),
        mxpool1(torch::nn::MaxPool2dOptions(2).stride(2))
    {
    	// register the layers if we want their gradients
        register_module("ConvLayer1", conv1);
        register_module("ConvLayer2", conv2);
        register_module("DenseLayer1", fc1);
        register_module("DenseLayer2", fc2);
        register_module("MaxPoolLayer", mxpool1);
        register_module("BatchnormLayer", bn1);

    }

    //create the forward function to guide the flow of Tensors from input to output.
    torch::Tensor forward(torch::Tensor x)
    {
        x = conv1(x);
        x = bn1(x);
        x = conv2(x);
        x = mxpool1(x);
        x = torch::flatten(x, 1);
        x = fc1(x);
        x = fc2(x);
        return x;
    }

}; 
TORCH_MODULE(DemoNet);


// Defining a small sequential module 
torch::nn::Sequential simpleSequence(
torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1)),
torch::nn::BatchNorm2d(32),  
torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
);


int main()
{

	//////////////////////////////   1. Tensor Creation 

	//Create a Tensor of size [9,2,7,1]
	torch::Tensor a = torch::randn({9,2,7,1});

	//Create a Tensor with necessary datatype
	torch::Tensor b = torch::arange(13, 43,  torch::TensorOptions().dtype(torch::kInt32));

	//Reshape `b` to size-[2,5,3]
	torch::Tensor c = b.view({2,-1,3}); // or b.view({-1,5,3}) or b.view({2,5,-1})
	std::cout<<"Shape of Tensor c is "<<c.sizes()<<std::endl;
	
	////////////////////////   2. Tensor Indexing

	// 2.1 Getting a Tensor

	// let A be a random Tensor of size-[9,5,7,4]
	torch::Tensor A = torch::randn({9,5,7,4});

	// Then Libtorch equivalent of A[:,2:4,:,1:3] is shown below.
	torch::Tensor subA1 = A.index({Slice(), Slice(2,4), Slice(), Slice(1,3)}); 
	// subA1 Tensor size will be (9, 2, 7, 2)
	std::cout<<"Shape of Tensor subA1 is "<<subA1.sizes()<<std::endl;


	// 2.2 Setting a Tensor

	//Setting a subtensor to a scalar value
	A.index_put_({"...",3}, 77);

	// Let's see if the elements are all set to 77 or not. We can do this with `torch::allclose()`. 
	// If they are not the same, then there shall be an Exception raised.
	torch::allclose(A.index({"...", 3}), torch::tensor({77.}));

	// Setting a sub-tensor with another Tensor
	torch::Tensor X1 = torch::randn({4,5,6,7});
	torch::Tensor Y1 = torch::randn({3,2,2});

	//Pytorch equivalent is X1[1:, 2:4, 0, 2:4] = Y1
	X1.index_put_({Slice(1,None), Slice(2,4), 0, Slice(2,4)}, Y1); //setting a subtensor of size [3,2,2] with B's elements.
	
	// Let's see if the particular indices of X1 has been changed to Y1. Let's simply subtract the corresponding tensors
	// and sum the residuals. If both the tensors are same, then the sum should be zero.
	torch::Tensor residual1 = X1.index({Slice(1,None), Slice(2,4), 0, Slice(2,4)}) - Y1; 
	std::cout <<"Residual sum between X1's subarray and Y1 after assignment is "<< *residual1.sum().data_ptr<float>()<<std::endl;

	// Setting a sub-tensor with another sub-tensor
	torch::Tensor X2 = torch::randn({4,5,6,7});
	torch::Tensor Y2 = torch::randn({3,2,2});

	//Pytorch equivalent is X2[1:, 2:4, 0, 2] = Y2[:,:,0]
	X2.index_put_({Slice(1,None), Slice(2,4), 0, 2}, Y2.index({Slice(),Slice(),0}));//setting A's subtensor of size-[3,2] with B's subtensor.
	// Let's verify if these two-subarrays are identical by taking the difference of these two sub-arrays and then summing-up.
	torch::Tensor residual2 = X2.index({Slice(1,None), Slice(2,4), 0, 2}) -  Y2.index({Slice(),Slice(),0});
	std::cout<<"Residual sum between X2's subarray and Y2's subarray after assignment is "<< *residual2.sum().data_ptr<float>()<<std::endl;
	
	////////////////////////////  3. Neural networks.

	// We create the output number of logits 
	int64_t N = 10;
	//instantite the model object
	DemoNet net(N); //equivalent to `net = DemoNet(N)` in Pytorch

	//create a random input and forward it.
	torch::Tensor x = torch::randn({1,1,28,28});
	torch::Tensor y = net->forward(x); // equivalent to `y = net(x)` in Pytorch

	cout<<"Output shape from simple-CNN  is "<<y.sizes()<<endl;



	// Calling the forward method on the Sequential object
	torch::Tensor logits = simpleSequence->forward(torch::randn({3,1,28,28}));
	cout<<"Output from sequential model is of size- "<<logits.sizes()<<endl;


	return 0;

}
