#pragma once

#include <vector>
#include <string>

typedef unsigned int uint;

class Model {
    private:
    // attributes 
    std::vector<std::vector<float>*> cache; // holds unactivated values of node
    std::vector<std::vector<float>*> activated; // holds activated values of nodes
    std::vector<std::vector<float>*> biases; // holds biases for nodes, except input
    std::vector<std::vector<float>*> nablaBiases; // holds gradient for each bias
    std::vector<std::vector<float>*> nablaCache; // holds gradient for each activated value
    std::vector<std::vector<std::vector<float>*>*> nablaWeights; // holds gradient for each weight
    std::vector<std::vector<std::vector<float>*>*> weights;
    float learningRate;

    // initialize model members
    void init_members(const uint hiddenLayers, const uint neuronsPerLayer);

    // used for changing float to string
    template <typename T> std::string toStr(const T& t);

    // sigmoid activation function
    float sigmoid(const float& in, bool derivative=false);
    
    // mean squared error function for calculating loss
    float MSE(std::vector<float> output, std::vector<float> label, bool derivative=false, int element=0);
 
    public:
    // constructor
    Model(uint numLayers, uint neuronsPerLayer, float lr = 0.05);

    // feed input forward and get output
    void forward(const std::vector<float>& feature);

    // backpropagate to compute gradients for each weight and bias
    void backward(int currentLabel);
    
    // weight vector to string for printing to console
    std::string weightsToString();

    // biases vector to string for printing to console
    std::string biasesToString();

    // activated vector to string for printing to console
    std::string activatedToString();

    std::string nablaWToString();

    std::string nablaCacheToString();

    std::string cacheToString();

    std::string nablaBToString();
};