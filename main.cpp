#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    CsvToVector::extract_data("data/data_filtered.csv");

    Model m1(2, 2, 0.01);

    int EPOCHS = 3;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        for (int elem = 0; elem < CsvToVector::features.size(); ++elem) {
            // std::cout << "features size: " << CsvToVector::features.size() << '\n';
            // std::cout << "elem=" << elem << "\n";

            m1.forward(CsvToVector::features[elem]);

            m1.backward(elem);
        }
    }
} 