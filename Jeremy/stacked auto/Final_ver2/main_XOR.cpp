

#include "MLP_Network.h"
#include "MLP_Global.h"
#include "MLP_Layer.h"
#include "MNIST_Parser_v5.h"


int main() {
    
    int num_input_size      = 2;        // node number of input layer
    int num_hidden_size     = 4;        // node number of hidden layer
    int num_output_size     = 1;        // node number of output layer
    
    float learning_rate     = 0.1;
    int num_hidden_Layer    = 1;        // number of hidden layer
    int num_mini_batch      = 4;        // possible to set a number of batch
                                        // 1 :pattern, 1 < n < training numbers : mini-batch, n: batch
    
    int num_training_set    = 4;        // number of traing sets
    int num_test_set        = 4;
    
    //Allocate
    float **inputs			= new float*[num_training_set];
    float **desired_outputs	= new float*[num_training_set];
    
    for(int i = 0;i < num_training_set;i++){
        inputs[i]			= new float[num_input_size];
        desired_outputs[i]	= new float[num_output_size];
    }
    
    
    desired_outputs[0][0] = 0;
    desired_outputs[1][0] = 1;
    desired_outputs[2][0] = 1;
    desired_outputs[3][0] = 0;
    
    inputs[0][0] = 0;	inputs[0][1] = 0;
    inputs[1][0] = 0;	inputs[1][1] = 1;
    inputs[2][0] = 1;	inputs[2][1] = 0;
    inputs[3][0] = 1;	inputs[3][1] = 1;

    
    //Create MLP Class
    MLP_Network* MLP = new MLP_Network(num_input_size,num_hidden_size, num_output_size,num_hidden_Layer,
                                        num_training_set, num_mini_batch,learning_rate, inputs, desired_outputs);
    MLP->Train();
    
    MLP->Train_Print_Result();
    
    /*
    //test
    desired_outputs[0][0] = 0;
    desired_outputs[1][0] = 0;
    desired_outputs[2][0] = 1;
    desired_outputs[3][0] = 1;
    
    inputs[0][0] = 1;	inputs[0][1] = 1;
    inputs[1][0] = 0;	inputs[1][1] = 0;
    inputs[2][0] = 1;	inputs[2][1] = 0;
    inputs[3][0] = 0;	inputs[3][1] = 1;
    
    
    //Test & Print Result
    MLP->Test_Print_Result(inputs, desired_outputs, num_test_set);
    
	*/

    //Deallocation
    delete MLP;
    
    for (int i = 0; i < num_training_set; i++)
    {
        desired_outputs[i]=NULL;
        inputs[i]=NULL;
        delete [] desired_outputs[i];
        delete [] inputs[i];
    }
    inputs=NULL;
    desired_outputs=NULL;
    
    delete[] inputs;
    delete[] desired_outputs;
    
 
    return 0;
}