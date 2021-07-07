#include "include/Model.hpp"
#include "include/CsvToVector.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

typedef unsigned int uint;

Model::Model(uint numHiddenLayers, uint neuronsPerLayer, float learningRate) {
    // assign learning rate member to constructor parameter
    this->learningRate = learningRate;

    init_members(numHiddenLayers, neuronsPerLayer);
}

void Model::init_members(const uint hiddenLayers, const uint neuronsPerLayer) {
    // create weight layer and gradient for input
    weights.push_back(new std::vector<std::vector<float>*>);
    nablaWeights.push_back(new std::vector<std::vector<float>*>);

    // create input weights 
    for (int x = 0; x < CsvToVector::features[0].size(); ++x) {
        weights[0]->push_back(new std::vector<float>);
        nablaWeights[0]->push_back(new std::vector<float>);
        for (int y = 0; y < neuronsPerLayer; ++y) {
            (*weights[0])[x]->push_back(0.5);
            (*nablaWeights[0])[x]->push_back(0.5);
        }
    }

    for (int x = 0; x < hiddenLayers; ++x) {
        weights.push_back(new std::vector<std::vector<float>*>);
        nablaWeights.push_back(new std::vector<std::vector<float>*>);
        cache.push_back(new std::vector<float>);
        nablaCache.push_back(new std::vector<float>);
        biases.push_back(new std::vector<float>);
        nablaBiases.push_back(new std::vector<float>);
        activated.push_back(new std::vector<float>);

        for (int y = 0; y < neuronsPerLayer; ++y) {
            weights[x + 1]->push_back(new std::vector<float>);
            nablaWeights[x + 1]->push_back(new std::vector<float>);
            cache[x]->push_back(0.5);
            nablaCache[x]->push_back(0.5);
            biases[x]->push_back(0.4);
            nablaBiases[x]->push_back(0.4);
            activated[x]->push_back(0.5);

            if (x < hiddenLayers - 1) {
                for (int z = 0; z < neuronsPerLayer; ++z) {
                    (*weights[x + 1])[y]->push_back(0.5);
                    (*nablaWeights[x + 1])[y]->push_back(0.5);
                }
            }
            else {
                for (int z = 0; z < 2; ++z) {
                    (*weights[x + 1])[y]->push_back(0.5);
                    (*nablaWeights[x + 1])[y]->push_back(0.5);
                }
            }
        }
        // last hidden layer
        if (x == hiddenLayers - 1) {
            // create members for output nodes
            cache.push_back(new std::vector<float>);
            nablaCache.push_back(new std::vector<float>);
            biases.push_back(new std::vector<float>);
            nablaBiases.push_back(new std::vector<float>);
            activated.push_back(new std::vector<float>);
            for (int y = 0; y < 2; ++y) {
                cache[x + 1]->push_back(0.5);
                nablaCache[x + 1]->push_back(0.5);
                biases[x + 1]->push_back(0.4);
                nablaBiases[x + 1]->push_back(0.4);
                activated[x + 1]->push_back(0.5);
            }
        }
    }
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

            (*nablaBiases[x])[y] = (*nablaCache[x])[y]; // bias gradient = cache of same node

            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                // hidden weight layers
                if (x > 0) {
                    // set weight gradients from hidden layer output
                    (*(*nablaWeights[x])[y])[z] = (*nablaCache[x])[z] * (*activated[x - 1])[y];
                    // summation of weight * cache gradient
                    summationActivation += (*(*weights[x])[y])[z] * (*nablaCache[x])[z];
                }
                // input weight layer
                else {
                    // set weight gradients from input
                    (*(*nablaWeights[x])[y])[z] = (*nablaCache[x])[z] * CsvToVector::features[currentLabel][y];
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
            (*biases[x])[y] -= (*nablaBiases[x])[y] * learningRate;
            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                // update weights
                (*(*weights[x])[y])[z] -= (*(*nablaWeights[x])[y])[z] * learningRate;
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

    for (int x = 0; x < nablaWeights.size(); ++x) {
        if (x > 0)
            str += " [";
        else
            str += "[";
        int nodeLayerSize = nablaWeights[x]->size();
        for (int y = 0; y < nodeLayerSize; ++y) {
            str += "[";
            int nodeSize = (*nablaWeights[x])[y]->size();
            for (int z = 0; z < nodeSize; ++z) {
                if (z < nodeSize - 1) {
                    str += toStr((*(*nablaWeights[x])[y])[z]);
                    str += ", ";
                }
                else {
                    str += toStr((*(*nablaWeights[x])[y])[z]);
                }
            }
            if (y < nodeLayerSize - 1) {
                str += "], ";
            }
            else {
                str += "]";
            }
        }
        if (x < nablaWeights.size() - 1) {
            str += "],\n";
        }
        else {
            str += "]";
        }
    }
    str += "(nablaWeights)]\n";
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
    for (int x = 0; x < nablaBiases.size(); ++x) {
        if (x > 0) {
            str += " [";
        }
        else {
            str += "[";
        }
        for (int y = 0; y < nablaBiases[x]->size(); ++y) {
            if (y < nablaBiases[x]->size() - 1) {
                str += toStr((*nablaBiases[x])[y]);
                str += ", ";
            }
            else {
                str += toStr((*nablaBiases[x])[y]);
            }
        }
        if (x < nablaBiases.size() - 1) {
            str += "],\n";
        }
        else {
            str += "]";
        }
    }
    str += "(nablaBiases)]\n";

    return str;
}