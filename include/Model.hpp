#pragma once

#include <vector>
#include <string>

typedef unsigned int uint;

class Model {
    private:
    // members 
    std::vector<std::vector<float>*> cache; // holds unactivated values of node
    std::vector<std::vector<float>*> activated; // holds activated values of nodes
    std::vector<std::vector<float>*> biases; // holds biases for nodes, except input
    std::vector<std::vector<float>*> nablaBiases; // holds gradient for each bias
    std::vector<std::vector<float>*> nablaCache; // holds gradient for each activated value
    std::vector<std::vector<std::vector<float>*>*> nablaWeights; // holds gradient for each weight
    std::vector<std::vector<std::vector<float>*>*> weights;
    long double summationLoss;
    float learningRate;

    // initialize model members
    void init_members(const uint hiddenLayers, const uint neuronsPerLayer);

    // sigmoid activation function
    float sigmoid(const float& in, bool derivative=false);

    // rectified linear unit activation function
    float ReLU(const float& in);
    
    // mean squared error function for calculating loss
    float MSE(std::vector<float> output, std::vector<float> label, bool derivative=false, int element=0);

    // takes input, usually dereferenced float, and returns string
    template <typename T> std::string toStr(const T& t);

    // returns string of double nested vector with name
    std::string str2(const std::vector<std::vector<float>*>& vec, const std::string name);

    // returns string of triple nested vector with name
    std::string str3(const std::vector<std::vector<std::vector<float>*>*>& vec, const std::string name);
 
    public:
    // constructor
    Model(std::vector<std::vector<float>>& features, uint numLayers, uint neuronsPerLayer, float lr = 0.05);

    // feed input forward and get output
    void forward(const std::vector<float>& feature);

    // backpropagate to compute gradients for each weight and bias
    void backward(const std::vector<float>& feature, const std::vector<float>& label);

    // prints average loss over a single epoch
    void printLoss(std::vector<std::vector<float>>& features);

    // randomize weights and biases on domain [0, 1]
    void randomize();

    // prints vector members of model
    friend std::ostream& operator<<(std::ostream& out, Model& mod);
};