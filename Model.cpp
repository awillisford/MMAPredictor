#include "include/Model.hpp"
#include "include/CsvToVector.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

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

    // initialize input weights
    float temp = 0.6;
    int numInputs = 0;
    for (int y = 0; y < weights[0]->size(); ++y) {
        for (int z = 0; z < neuronsPerLayer; ++z) {
            (*weights[0])[y]->push_back(temp);
        }
        numInputs++;
    }
    std::cout << "Inputs: " << numInputs << '\n';

    // create hidden layer nodes and weights
    for (uint x = 0; x < numHiddenLayers; ++x) {
        // create hidden layers and weights
        layers.push_back(new std::vector<uint>); 
        cache.push_back(new std::vector<float>);
        activated.push_back(new std::vector<float>);
        weights.push_back(new std::vector<std::vector<float>*>);
        // node creation and weights vector per node
        for (uint y = 0; y < neuronsPerLayer; ++y) {
            layers[x + 1]->push_back(y);
            cache[x]->push_back(y);
            activated[x]->push_back(y);
            weights[x + 1]->push_back(new std::vector<float>);

            // create each weight float per nodes in next layer
            float temp = 0.4;
            // if not currently on last hidden layer
            if (x < numHiddenLayers - 1) {
                for (uint z = 0; z < neuronsPerLayer; ++z) {
                    (*weights[x + 1])[y]->push_back(temp);
                }
            }
            // if on last layer
            else {
                // create two weights per node, since theres two outputs
                for (uint z = 0; z <= 1; ++z) {
                    (*weights[x + 1])[y]->push_back(temp);
                }
            }
        }
    }

    // create output layer nodes 
    layers.push_back(new std::vector<uint>);
    cache.push_back(new std::vector<float>);
    activated.push_back(new std::vector<float>);
    for (uint it = 0; it <= 1; ++it) { // two iterations for two outputs
        layers.back()->push_back(it);
        cache.back()->push_back(it);
        activated.back()->push_back(it);
    }
    // initialize biases
    init_biases();

    // initialize gradients
    nablaB = biases;
    nablaW = weights;
    nablaCache = nablaB;
}

float Model::sigmoid(const float& in, bool derivative) {
    // using derivative when already passed through sigmoid function
    if (derivative == true) {
        return in * (1 - in);
    }

    return 1/(1 + std::exp(-in));
}

void Model::forward(const std::vector<float>& feature) {
    // zero values of each cache
    for (int x = 0; x < cache.size(); ++x) {
        for (int y = 0; y < cache[x]->size(); ++y) {
            (*cache[x])[y] = 0;
        }
    }

    // forward feed
    for (int x = 0; x < weights.size(); ++x) {
        for (int y = 0; y < weights[x]->size(); ++y) {
            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                // weights from input
                if (x == 0) {
                    (*cache[x])[z] += (*(*weights[x])[y])[z] * feature[y]; // add weight * value to cache value
                }
                // weights from hidden nodes
                else {
                    (*cache[x])[z] += (*(*weights[x])[y])[z] * (*activated[x - 1])[y]; // add weight * value to cache value
                }
                // last group of weights in layer
                if (y == weights[x]->size() - 1) {
                    (*cache[x])[z] += (*biases[x])[z]; // add biases
                    (*activated[x])[z] = sigmoid((*cache[x])[z]); // activated equal to cache through activation function
                }
            }
        }
    }
}

void Model::backward(int currentLabel) {
    // print label to user
    std::cout << "- [" << CsvToVector::labels[currentLabel][0] << ", "
                       << CsvToVector::labels[currentLabel][1] << "] label\n";
    // print loss to user
    std::cout << "- " << MSE(*activated.back(), CsvToVector::labels[currentLabel])
              << " loss\n";
    // start from end, weight layers
    for (int x = weights.size() - 1; x >= 0; x--) {
        // iterate through each weight vector in weight layer         
        for (int y = 0; y < weights[x]->size(); ++y) {
            float summationActivation; // holds summation of partial deriv of loss with respect to activated node values

            // assign gradient to unactivated value of output nodes
            if (x == weights.size() - 1) {
                (*nablaCache[x])[0] = sigmoid(MSE(*activated.back(), CsvToVector::labels[currentLabel], true, 0), true);
                (*nablaCache[x])[1] = sigmoid(MSE(*activated.back(), CsvToVector::labels[currentLabel], true, 1), true);
            }

            (*nablaB[x])[y] = (*nablaCache[x])[y]; // bias gradient = cache of same node

            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                // hidden weight layers
                if (x > 0) {
                    // set weight gradients from hidden layer output
                    (*(*nablaW[x])[y])[z] = (*nablaCache[x])[z] * (*activated[x - 1])[y];
                    // summation of weight * cache gradient
                    summationActivation += (*(*weights[x])[y])[z] * (*nablaCache[x])[z];
                }
                // input weight layer
                else {
                    // set weight gradients from input
                    (*(*nablaW[x])[y])[z] = (*nablaCache[x])[z] * CsvToVector::features[currentLabel][y];
                }
            }
            // set gradients of cache from activation for all layers spare input
            if (x > 0) {
                (*nablaCache[x - 1])[y] = sigmoid(summationActivation, true);
            }
        }
    }

    // update weights and biases by gradients
    for (int x = 0; x < weights.size(); ++x) {
        for (int y = 0; y < weights[x]->size(); ++y) {
            // update biases
            (*biases[x])[y] -= (*nablaB[x])[y] * learningRate;
            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                // update weights
                (*(*weights[x])[y])[z] -= (*(*nablaW[x])[y])[z] * learningRate;
            }
        }
    }
}

float Model::MSE(std::vector<float> output, std::vector<float> label, bool derivative, int element) {
    if (derivative == true) {
        return output[element] - label[element];
    }
    float sum = 0;
    int size = output.size();
    for (int x = 0; x < size; ++x) {
        float squared = std::pow(label[x] - output[x], 2) ;
        sum += squared;
    }
    return (sum / size);
}

void Model::init_biases() {
    for (int x = 0; x < layers.size(); ++x) {
        // if not input layer
        if (x > 0) {
            biases.push_back(new std::vector<float>);
            // add bias element for each node
            for (int y = 0; y < layers[x]->size(); ++y) {
                float temp = 0.5;
                biases[x - 1]->push_back(temp);
            }
        }
        // skip applying biases for input layer
        else {
            continue;
        }
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

    for (int x = 0; x < weights.size(); ++x) {
        if (x > 0)
            str += " [";
        else
            str += "[";
        int nodeLayerSize = weights[x]->size();
        for (int y = 0; y < nodeLayerSize; ++y) {
            str += "[";
            int nodeSize = (*weights[x])[y]->size();
            for (int z = 0; z < nodeSize; ++z) {
                if (z < nodeSize - 1) {
                    str += toStr((*(*weights[x])[y])[z]);
                    str += ", ";
                }
                else {
                    str += toStr((*(*weights[x])[y])[z]);
                }
            }
            if (y < nodeLayerSize - 1) {
                str += "], ";
            }
            else {
                str += "]";
            }
        }
        if (x < weights.size() - 1) {
            str += "],\n";
        }
        else {
            str += "]";
        }
    }
    str += "(weights)]\n";
    return str;
}

std::string Model::nablaWToString() {
    std::string str;
    str += "["; // show beginning of weight vector

    for (int x = 0; x < nablaW.size(); ++x) {
        if (x > 0)
            str += " [";
        else
            str += "[";
        int nodeLayerSize = nablaW[x]->size();
        for (int y = 0; y < nodeLayerSize; ++y) {
            str += "[";
            int nodeSize = (*nablaW[x])[y]->size();
            for (int z = 0; z < nodeSize; ++z) {
                if (z < nodeSize - 1) {
                    str += toStr((*(*nablaW[x])[y])[z]);
                    str += ", ";
                }
                else {
                    str += toStr((*(*nablaW[x])[y])[z]);
                }
            }
            if (y < nodeLayerSize - 1) {
                str += "], ";
            }
            else {
                str += "]";
            }
        }
        if (x < nablaW.size() - 1) {
            str += "],\n";
        }
        else {
            str += "]";
        }
    }
    str += "(nablaW)]\n";
    return str;
}

std::string Model::layersToString() {
    std::string str;
    str += "[";

    for (int x = 0; x < layers.size(); ++x) {
        if (x == 0)
            str += "[";
        else
            str += " [";
        
        for (int y = 0; y < layers[x]->size(); ++y) {
            if (y < layers[x]->size() - 1) {
                str += toStr((*layers[x])[y]);
                str += ", ";
            }
            else
                str += toStr((*layers[x])[y]);
        }
        if (x < layers.size() - 1)
            str += "],\n";
        else
            str += "]";
    }
    str += "(layers)]\n";

    return str;
}

std::string Model::biasesToString() {
    std::string str;
    str += "[";

    for (int x = 0; x < biases.size(); ++x) {
        if (x == 0)
            str += "[";
        else
            str += " [";
        
        for (int y = 0; y < biases[x]->size(); ++y) {
            if (y < biases[x]->size() - 1) {
                str += toStr((*biases[x])[y]);
                str += ", ";
            }
            else
                str += toStr((*biases[x])[y]);
        }
        if (x < biases.size() - 1)
            str += "],\n";
        else
            str += "]";
    }
    str += "(biases)]\n";

    return str;
}

std::string Model::activatedToString() {
    std::string str = "[";
    for (int x = 0; x < activated.size(); ++x) {
        if (x > 0) {
            str += " [";
        }
        else {
            str += "[";
        }
        for (int y = 0; y < activated[x]->size(); ++y) {
            if (y < activated[x]->size() - 1) {
                str += toStr((*activated[x])[y]);
                str += ", ";
            }
            else {
                str += toStr((*activated[x])[y]);
            }
        }
        if (x < activated.size() - 1) {
            str += "],\n";
        }
        else {
            str += "]";
        }
    }
    str += "(activated)]\n";

    return str;
}

std::string Model::cacheToString() {
    std::string str = "[";
    for (int x = 0; x < cache.size(); ++x) {
        if (x > 0) {
            str += " [";
        }
        else {
            str += "[";
        }
        for (int y = 0; y < cache[x]->size(); ++y) {
            if (y < cache[x]->size() - 1) {
                str += toStr((*cache[x])[y]);
                str += ", ";
            }
            else {
                str += toStr((*cache[x])[y]);
            }
        }
        if (x < cache.size() - 1) {
            str += "],\n";
        }
        else {
            str += "]";
        }
    }
    str += "(cache)]\n";

    return str;
}

std::string Model::nablaCacheToString() {
    std::string str = "[";
    for (int x = 0; x < nablaCache.size(); ++x) {
        if (x > 0) {
            str += " [";
        }
        else {
            str += "[";
        }
        for (int y = 0; y < nablaCache[x]->size(); ++y) {
            if (y < nablaCache[x]->size() - 1) {
                str += toStr((*nablaCache[x])[y]);
                str += ", ";
            }
            else {
                str += toStr((*nablaCache[x])[y]);
            }
        }
        if (x < nablaCache.size() - 1) {
            str += "],\n";
        }
        else {
            str += "]";
        }
    }
    str += "(nablaCache)]\n";

    return str;
}

std::string Model::nablaBToString() {
    std::string str = "[";
    for (int x = 0; x < nablaB.size(); ++x) {
        if (x > 0) {
            str += " [";
        }
        else {
            str += "[";
        }
        for (int y = 0; y < nablaB[x]->size(); ++y) {
            if (y < nablaB[x]->size() - 1) {
                str += toStr((*nablaB[x])[y]);
                str += ", ";
            }
            else {
                str += toStr((*nablaB[x])[y]);
            }
        }
        if (x < nablaB.size() - 1) {
            str += "],\n";
        }
        else {
            str += "]";
        }
    }
    str += "(nablaB)]\n";

    return str;
}