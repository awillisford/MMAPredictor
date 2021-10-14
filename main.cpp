#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    std::vector<std::vector<std::vector<float>>> data = CsvToVector::extract_data("data/data_filtered.csv");
    std::vector<std::vector<float>> features = data[0];
    std::vector<std::vector<float>> labels = data[1];

    std::vector<std::vector<float>> train_features(&features[0], &features[2600]);
    std::vector<std::vector<float>> test_features(&features[2600], &features.back()+1);

    std::vector<std::vector<float>> train_labels(&labels[0], &labels[2600]);
    std::vector<std::vector<float>> test_labels(&labels[2600], &labels.back()+1);

    Model m1(features, 1, 200, 0.005);

    m1.randomize(4); // randomize weights and biases

    int EPOCHS = 5;

    // train
    std::cout << "Training:\n";
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int elem = 0; elem < train_features.size(); ++elem) {
            m1.forward(train_features[elem], train_labels[elem]);
            m1.backward(train_features[elem], train_labels[elem]);
        }
        std::cout << "epoch(" << epoch << ") : ";
        m1.printLoss(train_features);
    }

    // test
    std::cout << "\nTesting:\n";
    for (int elem = 0; elem < test_features.size(); ++elem) {
        m1.forward(test_features[elem], test_labels[elem]);
    }
    m1.printLoss(test_features);
} 