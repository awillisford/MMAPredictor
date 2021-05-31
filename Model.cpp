#include "Model.h"
#include <iostream>

Model::Model(std::vector<std::vector<float>> input, uint numLayers, uint neuronsPerLayer, float learningRate) {
    // assign class attribute to constructor parameter
    this->learningRate = learningRate;

    // create vector for input layer's weights
    weights.push_back(new std::vector<std::vector<float>*>);
    // create a weight vector that holds type float for each node in input
    for (int it = 0; it < input[0].size() * 2; ++it) { // 2 inputs
        weights[0]->push_back(new std::vector<float>);
    }

    // create layers ----- UPDATE COMMENT
    for (uint x = 0; x < numLayers; ++x) {
        layers.push_back(new std::vector<uint>);
        weights.push_back(new std::vector<std::vector<float>*>); // create hidden layer weights
        // create neurons  ------- UPDATE COMMENT
        for (uint z = 0; z < neuronsPerLayer; ++z) {
            layers[x]->push_back(z);
            weights[x + 1]->push_back(new std::vector<float>); // creates weight vector for each node in hidden layers
        }
    }

    // assign weights for each node past input layer
    for (uint x = 0; x < numLayers + 1; ++x) {
        float temp = 1.0; // temporary float variable to push back into vector
        for (uint z = 0; z < neuronsPerLayer; ++z) {
            // assign weights per node based on number of neurons in each layer
            if (x < numLayers) {
                for (int it = 0; it < neuronsPerLayer; ++it) {
                    (*weights[x])[z]->push_back(temp);
                }
            }
            // create two weights per node in last hidden layer since there are two outputs
            else {
                for (int it = 0; it < 2; ++it) {
                    (*weights[x])[z]->push_back(temp);
                }
            }
        }
    }
}

std::string Model::weightsToString() {
    std::string str;
    str += "["; // show beginning of weight vector

    // layers
    for (int z = 0; z < weights.size(); ++z) {
        str += "["; // show beginning of layer
        // node 
        for (int x = 0; x < weights[z]->size(); ++x) {
            str += "["; // beginning of node
            // node vector 
            for (int y = 0; y < (*weights[z])[x]->size(); ++y) {
                // if not last element
                if (y < (*weights[z])[x]->size() - 1)
                    str += (*(*weights[z])[x])[y] , ", "; // end of current element, onto next
                // if last element
                else
                    str += (*(*weights[z])[x])[y];
            }
            // if not last node in layer
            if (x < weights[z]->size() - 1)
                str += "], "; // end of current node, onto next
            // last node in layer
            else
                str += "]";
        }
        // if not last layer
        if (z < weights.size() - 1)
            str += "],\n "; // end of current layer, onto next
        // last layer
        else 
            str += "]";
    }
    str += "]"; // show end of weight vector

    return str;
}

int main() {
    std::cout << "Working (currently)...";
    return 0;
}