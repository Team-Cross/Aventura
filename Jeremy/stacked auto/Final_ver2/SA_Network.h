#ifndef MLP_Network_H
#define MLP_Network_H
#include "SA_Layer.h"
#include "SA_Global.h"


class SA_Network {
    
private:
    SA_Layer *layer_Network;
    SA_Layer *pre_layer;

    float**  input_network;
    float** output_network;
    float** desired_output;
	float** temp_output;
	float** temp_input;
    
    int num_training_set;
    int num_input_Node;
    int num_hidden_Node;
    int num_output_Node;
    int num_hidden_Layer;
    int num_mini_batch;
    int batch_count;

    float learning_rate;
    float sumError;
    
public:
    SA_Network(int num_input_Node,int num_hidden_Node,int num_output_Node,
                int num_hidden_Layer,int num_training_set, int num_mini_batch,
                float learning_rate,float **trainData, float **Desired_output);
    ~SA_Network();
    void Allocate_Network();
    void Deallocate_Network();
    
    void Train();
	void SA_Train();
    void Forward_Propagate_Network(int num);
	void Pre_Forward_Network(int num, int layer_num);
    void Backward_Propagate_Network(int num);
	void Pre_Back_Network(int num, int layer_num);
    void Update_Weight_Network();
	void Pre_Update_Weight(int layer_num);
    void Handle_Error();
	void Pre_Handle_Error(int layer_num);
    void Output_Layer_Training(int layer_num);

    void Train_Print_Result();
    void Test_Print_Result(float** input, float** desired_output, int num_test_set);

	float get_accuarcy();
    
};

#endif
