#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    std::vector<std::vector<std::vector<float>>> data = CsvToVector::extract_data("data/data_filtered.csv");
    std::vector<std::vector<float>> features = data[0];
    std::vector<std::vector<float>> labels = data[1];

    Model m1(features, 2, 200, 0.1);

    m1.randomize(); // randomize weights and biases

    int EPOCHS = 5;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int elem = 0; elem < features.size(); ++elem) {
            m1.forward(features[elem]);
            m1.backward(features[elem], labels[elem]);
        }
        std::cout << "epoch(" << epoch << ") : ";
        m1.printLoss(features);
    }
} 