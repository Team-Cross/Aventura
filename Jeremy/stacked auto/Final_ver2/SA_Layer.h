#ifndef MLP_Layer_H
#define MLP_Layer_H
#include "SA_Global.h"

class  SA_Layer {
private:
    int num_previous_node;
    int num_current_node;
    
    float* input_layer;
    float* output_layer;
    float* weight;
    float* gradients;
    float* deltas;
    
    float* bias_weights;                // basically, handle the bias weights and bias gradients separately
    float* bias_gradients;
    
public:
    SA_Layer(){};
    ~SA_Layer();
    
    void Allocate_Layer(int previous_node_num, int num_current_node);
    void Init_Layer();
    void Deallocate_Layer();
    
    float* Forward_Propagate_Layer(float* input_layer);
    void Backward_Propagate_Hidden_Layer(SA_Layer* previousLayer);
    void Backward_Propagate_Output_Layer(float* desiredValues);
    
    float Sigmoid(float net);
    void Update_Weight_Layer(float learningRate);
    
    float* get_weight()     {return weight;}
    float* get_delta()      {return deltas;}
    int get_current_num()   {return num_current_node;}
};

#endif