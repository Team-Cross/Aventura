#include "SA_Network.h"
#include "SA_Global.h"
#include "SA_Layer.h"
#include "MNIST_Parser_v5.h"

int main()
{
    
    int num_input_size      = 784;        // units of input layer
    int num_hidden_size     = 500;        // units of hidden layer
    int num_output_size     = 10;         // units of output layer
    
    int num_hidden_Layer    = 1;          // hidden layers
    int num_mini_batch      = 1;         // 1 :pattern, 1 < n < training numbers : mini-batch, n: batch
    float learning_rate     = 0.1;
    
    int num_training_set    = 1000;      // training sets
    int num_test_set        = 1000;       // test sets
    
    //Allocate
    float **inputs			= new float*[num_training_set];
    float **desired_outputs	= new float*[num_training_set];
    
    for(int i = 0;i < num_training_set;i++){
        inputs[i]			= new float[num_input_size];
        desired_outputs[i]	= new float[num_output_size];
    }
    
    //Start clock
    clock_t start, finish;
    double elapsed_time;
    start = clock();

    
    MNIST_Parser m_parser;
    

    m_parser.ReadMNIST_Input("train-images-idx3-ubyte", num_training_set, inputs);
    m_parser.ReadMNIST_Label("train-labels-idx1-ubyte",num_training_set, desired_outputs);
    
    

    SA_Network* SA_MNIST = new SA_Network(num_input_size,num_hidden_size,num_output_size,
                                             num_hidden_Layer,num_training_set,num_mini_batch,
                                             learning_rate,inputs,desired_outputs);

    SA_MNIST->SA_Train();
    

    //MLP_MNIST->Train_Print_Result();
    

    m_parser.ReadMNIST_Input("t10k-images-idx3-ubyte",num_test_set, inputs);
    m_parser.ReadMNIST_Label("t10k-labels-idx1-ubyte",num_test_set, desired_outputs);
    

    SA_MNIST->Test_Print_Result(inputs, desired_outputs, num_test_set);
    
    
    
    //Finish clock
    finish = clock();
    elapsed_time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"time: "<<elapsed_time<<" sec"<<endl;
    

    delete SA_MNIST;
    
    for (int i = 0; i < num_training_set; i++)
    {
        delete [] desired_outputs[i];
        delete [] inputs[i];
    }

    delete[] inputs;
    delete[] desired_outputs;
    
 
    return 0;
}