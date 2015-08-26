#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H

/*
* Lastly Modified : 2015.08.03
* MNIST_Parser_v5
* Code originally motivated from 'mrgloom' in Stack Overflow
* Modified and Customized by Kevin Na
* Deprecated methods are deleted
* support parsing method for original MNIST database
* support parsing & convert method for MLP_inputs/outputs
*/

#include <iostream>
#include <fstream>

using namespace std;

class MNIST_Parser {
public:
    void ReadMNIST_Input(string filename, int num_images, float** inputs);
    void ReadMNIST_Label(string filename, int num_labels, float** outputs);
private:
	int BytetoInt(int byte); // convert Byte to Int
};

#endif
