#include "include/Model.hpp"
#include "include/CsvToVector.hpp"
#include <iostream>
#include <sstream>

Model::Model(uint numHiddenLayers, uint neuronsPerLayer, float learningRate) {
    // assign class attribute to constructor parameter
    this->learningRate = learningRate;

    // create layer and weight vector for input
    weights.push_back(new std::vector<std::vector<float>*>);
    layers.push_back(new std::vector<uint>);

    // create input nodes for layer vector and input weights 
    for (int it = 0; it < CsvToVector::features[0].size(); ++it) {
        layers[0]->push_back(it);
        weights[0]->push_back(new std::vector<float>);
    }

    // create hidden layer nodes and weights
    for (uint x = 0; x < numHiddenLayers; ++x) {
        // create hidden layers and weights
        layers.push_back(new std::vector<uint>); 
        weights.push_back(new std::vector<std::vector<float>*>);
        // node creation and weights vector per node
        for (uint y = 0; y < neuronsPerLayer; ++y) {
            layers[x + 1]->push_back(y);
            weights[x + 1]->push_back(new std::vector<float>);

            // create each weight float per nodes in next layer
            float temp = 1.1;
            // if not currently on last hidden layer
            if (x < numHiddenLayers - 1) {
                for (uint z = 0; z < neuronsPerLayer; ++z) {
                    (*weights[x])[y]->push_back(temp);
                }
            }
            // if on last layer
            else {
                // create two weights per node
                for (uint z = 0; z <= 1; ++z) {
                    (*weights[x])[y]->push_back(temp);
                }
            }
        }
    }

    // create output layer nodes 
    layers.push_back(new std::vector<uint>);
    for (uint it = 0; it <= 2; ++it) {
        layers.back()->push_back(it);
    }
}

template <typename T> std::string Model::toStr(const T& t) { 
   std::ostringstream os; 
   os << t; 
   return os.str(); 
} 

std::string Model::weightsToString() {
    std::string str;
    str += "["; // show beginning of weight vector

    // weight layers
    for (int x = 0; x < weights.size(); ++x) {
        str += "[";
        // node weight vector
        int nodeLayerSize = weights[x]->size();
        for (int y = 0; y < nodeLayerSize; ++y) {
            str += "[";
            // individual weights
            int nodeSize = (*weights[x])[y]->size();
            for (int z = 0; z < nodeSize; ++z) {
                // if not last element
                if (z < nodeSize - 1) {
                    str += toStr((*(*weights[x])[y])[z]);
                    str += ", ";
                    std::cout << "nL: " << (*(*weights[x])[y])[z] << '\n';
                }
                // if last element
                else {
                    str += toStr((*(*weights[x])[y])[z]);
                    std::cout << "last: " << (*(*weights[x])[y])[z] << '\n';
                }
            }
            // if not last element
            if (y < nodeLayerSize - 1) {
                str += "], ";
            }
            // if last element
            else {
                str += "]";
            }
        }
        // if not last element
        if (x < weights.size() - 1) {
            str += "],\n";
        }
        // if last element
        else {
            str += "]";
        }
    }
    str += "]";

    return str;
}