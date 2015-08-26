
#include "SA_Layer.h"
#include "SA_Global.h"

void SA_Layer::Allocate_Layer(int previous_num, int current_num)
{
    this->num_previous_node   =  previous_num;
    this->num_current_node    =  current_num;
    
    weight          = new float[num_previous_node * num_current_node];
    gradients       = new float[num_previous_node * num_current_node];
    input_layer     = new float[num_previous_node];
    output_layer    = new float[num_current_node];
    deltas          = new float[num_current_node];
    bias_weights    = new float[num_current_node]; 
    bias_gradients  = new float[num_current_node];
    
    Init_Layer();
}

void SA_Layer::Init_Layer()
{
    srand((unsigned)time(NULL));
    for (int j = 0; j < num_current_node; j++)
    {
        output_layer[j]=0.0;
        deltas[j]=0.0;
        for (int i = 0; i < num_previous_node; i++)
        {
            weight[j*num_previous_node+i]   = 0.2 * rand() / RAND_MAX - 0.1;
            gradients[j*num_previous_node+i]= 0.0;
        }
        bias_weights[j] = 1;                                        // bias weights are 1
        bias_gradients[j] = 0;
    }
}

SA_Layer::~SA_Layer()
{
    Deallocate_Layer();
};

void SA_Layer::Deallocate_Layer(){

    delete [] weight;
    delete [] gradients;
    delete [] deltas;
    delete [] output_layer;
    delete [] bias_gradients;
    delete [] bias_weights;
}

float* SA_Layer::Forward_Propagate_Layer(float* input_layers)      // f( sigma(weights * inputs) + bias )
{
    this->input_layer=input_layers;
    for(int j = 0 ; j < num_current_node ; j++)
    {
        float net= 0;
        for(int i = 0 ; i < num_previous_node ; i++)
        {
            net += input_layer[i] * weight[j*num_previous_node+i];
        }
        net+=bias_weights[j];
        
        output_layer[j] = Sigmoid(net);
    }
    return output_layer;
}

void SA_Layer::Backward_Propagate_Output_Layer(float* desiredValues)
{
    for (int k = 0; k < num_current_node; k++)                // calculate deltas with desiredValues and outputValues
    {
        float fnet_derivative = output_layer[k] * (1 - output_layer[k]);
        deltas[k] = fnet_derivative * (desiredValues[k] - output_layer[k]);
    }
    
    for (int k = 0 ; k < num_current_node ; k++)              // accumulate output gradients using deltas and input values
        for (int j = 0 ; j < num_previous_node; j++)
            gradients[k*num_previous_node + j] += - (deltas[k] * input_layer[j]);
    
    for (int k = 0 ; k < num_current_node   ; k++)
            bias_gradients[k] += - deltas[k] ;
    
    
}

void SA_Layer::Backward_Propagate_Hidden_Layer(SA_Layer* previousLayer)
{
    
    float* previousLayer_weight = previousLayer->get_weight();
    float* previousLayer_delta = previousLayer->get_delta();
    int previousLayer_node_num = previousLayer->get_current_num();

    for (int j = 0; j < num_current_node; j++)
    {
        float previous_sum=0;
        for (int k = 0; k < previousLayer_node_num; k++)        // calculate with previous deltas and weights
        {
            previous_sum += previousLayer_delta[k] * previousLayer_weight[k*num_current_node + j];
        }
        deltas[j] = output_layer[j] * (1 - output_layer[j]) * previous_sum;
    }
    
    for (int j = 0; j < num_current_node; j++)                  // accumulate hidden gradients using output deltas and input values
        for (int i = 0; i < num_previous_node ; i++)
            gradients[j*num_previous_node + i] +=  -deltas[j] * input_layer[i];
    
    for (int j = 0 ; j < num_current_node   ; j++)              // accumulate bias gradients with deltas
        bias_gradients[j] += -deltas[j] ;
}

void SA_Layer::Update_Weight_Layer(float learningRate)
{
    for (int j = 0; j < num_current_node; j++)                  // update weights with accumulated gradients
        for (int i = 0; i < num_previous_node; i++)
            weight[j*num_previous_node + i] +=  -learningRate *gradients[j*num_previous_node + i];
    
    for (int j = 0; j < num_current_node; j++)                  // update bias weights with accumulated bias gradients
        bias_weights[j] += -bias_gradients[j];
        //bias_weights[j] += -learningRate*bias_gradients[j];
    
    for (int j = 0; j < num_current_node; j++)                  // initiate gradients after update weights
        for (int i = 0; i < num_previous_node; i++)
            gradients[j*num_previous_node + i] = 0;
    
    for (int j = 0; j < num_current_node; j++)
        bias_gradients[j]=0;
}

float SA_Layer::Sigmoid(float net)
{
    return 1.0 / (1.0 + exp(-net));
}

