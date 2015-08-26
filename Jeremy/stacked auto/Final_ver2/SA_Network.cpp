#include "SA_Network.h"
#include "SA_Global.h"

SA_Network::SA_Network(int num_input_Node,   int num_hidden_Node, int num_output_Node, int num_hidden_Layer,
                         int num_training_set, int num_mini_batch,  float learning_rate,  float **training_data, float **p_desired_output)
{
    this->num_training_set  = num_training_set;
    this->num_input_Node    = num_input_Node;
    this->num_hidden_Node   = num_hidden_Node;
    this->num_output_Node   = num_output_Node;
    this->num_hidden_Layer  = num_hidden_Layer;
    this->num_mini_batch    = num_mini_batch;
    this->learning_rate     = learning_rate;
    this->sumError          = 0;
    this->batch_count       = 0;
	
    Allocate_Network();                         // Allocate
    
    this->input_network     = training_data;
	this->temp_input = input_network;
    this->desired_output    = p_desired_output;
}

void SA_Network::Allocate_Network()
{
    this->input_network           = new float*[this->num_training_set];
    this->output_network          = new float*[this->num_training_set];
    this->desired_output          = new float*[this->num_training_set];
    this->temp_output          = new float*[this->num_training_set];

    for(int i = 0;i < this->num_training_set;i++){
        this->output_network[i]   = new float[this->num_output_Node];
        this->input_network[i]    = new float[this->num_input_Node];
        this->desired_output[i]   = new float[this->num_output_Node];
    }
    
    layer_Network = new SA_Layer[num_hidden_Layer+1]();
    pre_layer = new SA_Layer[num_hidden_Layer]();

    layer_Network[0].Allocate_Layer(num_input_Node, num_hidden_Node);
    for (int i = 1; i < num_hidden_Layer; i++)
    {
        layer_Network[i].Allocate_Layer(num_hidden_Node, num_hidden_Node);
    }

	pre_layer[0].Allocate_Layer(num_hidden_Node, num_input_Node);
	for (int i = 1; i < num_hidden_Layer; i++)
    {
        pre_layer[i].Allocate_Layer(num_hidden_Node, num_hidden_Node);
    }

    layer_Network[num_hidden_Layer].Allocate_Layer(num_hidden_Node, num_output_Node);
}

SA_Network::~SA_Network()
{
    Deallocate_Network();
}

void SA_Network::Deallocate_Network()
{
    for (int i = 0; i < num_hidden_Layer+1; i++)
    {
        layer_Network[i].Deallocate_Layer();
    }
}

void SA_Network::SA_Train()
{
	
	int epoch = 0;

	for(int layer_num = 0; layer_num < num_hidden_Layer; layer_num++)
	{
		epoch = 0;
		
		while (epoch < MAXEPOCHS)
		{
			sumError=0;

			for (int i = 0; i < num_training_set; i++)
				temp_output[i] = new float[layer_Network[layer_num].get_current_num()];

			for (int i = 0; i < num_training_set; i++)
			{
				Pre_Forward_Network(i, layer_num);
            
				Pre_Back_Network(i, layer_num);
            
				Pre_Update_Weight(layer_num);
			}
        
			Pre_Handle_Error(layer_num);
        
			cout<< layer_num << ": " << epoch<<" | "<<sumError<<endl;
        
			if (sumError < EMAX)
			{				
				temp_input = temp_output;
				break;
			}
        
			++epoch;
		}
	}
	
	
	Train();

	/*while (epoch < MAXEPOCHS)
	{
		sumError=0;

		for (int i = 0; i < num_training_set; i++)
			Output_Layer_Training(i);
        
		Handle_Error();

		cout<<epoch<<" | "<<sumError<<endl;
        
		if (sumError < EMAX)
			break;
        
		++epoch;	
	}
	*/
}

void SA_Network::Pre_Forward_Network(int num, int layer_num)
{
	float* outputOfHiddenLayer=NULL;
    
    outputOfHiddenLayer=layer_Network[layer_num].Forward_Propagate_Layer(temp_input[num]);
	temp_output[num] = pre_layer[layer_num].Forward_Propagate_Layer(outputOfHiddenLayer);
}

void SA_Network::Pre_Back_Network(int num, int layer_num)
{
	pre_layer[layer_num].Backward_Propagate_Output_Layer(temp_input[num]);
	layer_Network[layer_num].Backward_Propagate_Hidden_Layer(&pre_layer[layer_num]);	
}

void SA_Network::Pre_Update_Weight(int layer_num)
{
	batch_count++;
    
    if( num_mini_batch == batch_count)
    {       
        layer_Network[layer_num].Update_Weight_Layer(learning_rate);        
        pre_layer[layer_num].Update_Weight_Layer(learning_rate);
        
        batch_count=0;
    }
}

void SA_Network::Pre_Handle_Error(int layer_num){
    
	int num_out_node = pre_layer[layer_num].get_current_num();

    //float sumTotal=0;
    for (int i = 0; i < num_training_set; ++i)
    {        
        Pre_Forward_Network(i, layer_num);
        
        for (int j = 0; j < num_out_node; ++j)
        {
            float err = temp_output[i][j] - temp_input[i][j];
            //sumTotal += err * err;
            sumError +=  err * err;
        }        
    }
    //sumError += 0.5*sumTotal;
    sumError /= num_training_set;   //MSE vs SSE

}

void SA_Network::Output_Layer_Training(int num)
{
	output_network[num] = layer_Network[num_hidden_Layer].Forward_Propagate_Layer(temp_input[num]);
	layer_Network[num_hidden_Layer].Backward_Propagate_Output_Layer(output_network[num]);

	batch_count++;
    
    if( num_mini_batch == batch_count)
    {       
        layer_Network[num_hidden_Layer].Update_Weight_Layer(learning_rate);
        
        batch_count=0;
    }
}
void SA_Network::Train()
{
    int epoch = 0;
    while (epoch < MAXEPOCHS)
    {
        sumError=0;
        for (int i = 0; i < num_training_set; i++)
        {
            Forward_Propagate_Network(i);
            
            Backward_Propagate_Network(i);
            
            Update_Weight_Network();
        }
        
        Handle_Error();
        
        cout<<epoch<<" | "<<sumError<<endl;
        
        if (sumError < EMAX)
            break;
        
        ++epoch;
    }
}

void SA_Network::Forward_Propagate_Network(int num)
{
    float* outputOfHiddenLayer=NULL;
    
    outputOfHiddenLayer=layer_Network[0].Forward_Propagate_Layer(input_network[num]);
    for (int i=1; i < num_hidden_Layer ; i++)
    {
        outputOfHiddenLayer=layer_Network[i].Forward_Propagate_Layer(outputOfHiddenLayer);                  //hidden forward
    }
    output_network[num]=layer_Network[num_hidden_Layer].Forward_Propagate_Layer(outputOfHiddenLayer);      // output forward
}

void SA_Network::Backward_Propagate_Network(int num)
{
    layer_Network[num_hidden_Layer].Backward_Propagate_Output_Layer(desired_output[num]);  // back_propa_output
    for (int i= num_hidden_Layer-1; i >= 0  ; i--)
        layer_Network[i].Backward_Propagate_Hidden_Layer(&layer_Network[i+1]);              // back_propa_hidden
}

void SA_Network::Update_Weight_Network()
{
    batch_count++;
    
    if( num_mini_batch == batch_count)
    {
        for (int i = 0; i < num_hidden_Layer; i++)
            layer_Network[i].Update_Weight_Layer(learning_rate);
        
        layer_Network[num_hidden_Layer].Update_Weight_Layer(learning_rate);
        
        batch_count=0;
    }
}

void SA_Network::Handle_Error(){
    
    //float sumTotal=0;
    for (int i = 0; i < num_training_set; ++i)
    {        
        Forward_Propagate_Network(i);
        
        for (int j = 0; j < num_output_Node; ++j)
        {
            float err = desired_output[i][j] - output_network[i][j];
            //sumTotal += err * err;
            sumError +=  err * err;
        }
        
    }
    //sumError += 0.5*sumTotal;
    sumError /= num_training_set;   //MSE vs SSE

}

void SA_Network::Train_Print_Result()
{
    cout<<"===============TRAIN RESULT==============="<<endl<<endl;
    for (int i = 0; i < num_training_set; ++i)
    {
        Forward_Propagate_Network(i);
        cout<<"=====================Training Number [ "<<i<<" ]"<<endl;
        for (int j = 0; j < num_output_Node; ++j)
            cout<<" "<<desired_output[i][j]<<"   "<<output_network[i][j]<<endl;
    }
    cout<<"=========================================="<<endl<<endl;
}

void SA_Network::Test_Print_Result(float** input, float** desired_output, int num_test_set)
{

	int sums = 0;
	int max = 0;
	float error_rate = 0.0f;

    this->input_network=input;
    this->desired_output=desired_output;
    	
    //cout<<"================TEST RESULT==============="<<endl<<endl;
    for (int i = 0; i < num_test_set; ++i)
    {
        Forward_Propagate_Network(i);
        //cout<<"=====================Test Number [ "<<i<<" ]"<<endl;
        for (int j = 0; j < num_output_Node; ++j)
		{
            //cout<<" "<<desired_output[i][j]<<"   "<<output_network[i][j]<<endl;

			if(output_network[i][max] < output_network[i][j])
				max = j;
		}

		if(desired_output[i][max] == 1.0f)
			sums++;
    }

    cout<<"=========================================="<<endl;

	error_rate = 100 - (sums / (float)num_test_set) * 100;

	cout << "error rate: " << error_rate << endl;
}