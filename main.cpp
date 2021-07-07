#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    CsvToVector::extract_data("data/data_filtered.csv");
    
    Model m1(2, 3, 0.05);

    int EPOCHS = 2;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        for (int elem = 0; elem < CsvToVector::features.size(); ++elem) {

            // std::cout << "---------------BEFORE FORWARD PASS-------------\n";
            // std::cout << m1.activatedToString() << "\n";
            // std::cout << m1.cacheToString() << "\n";
            // std::cout << m1.nablaCacheToString() << "\n";
            // std::cout << m1.biasesToString() << "\n";
            // std::cout << m1.nablaBToString() << "\n";
            // std::cout << m1.weightsToString() << "\n";
            // std::cout << m1.nablaWToString() << "\n";

            m1.forward(CsvToVector::features[elem]);

            // std::cout << "---------------AFTER FORWARD PASS-------------\n";
            // std::cout << m1.activatedToString() << "\n";
            // std::cout << m1.cacheToString() << "\n";
            // std::cout << m1.nablaCacheToString() << "\n";
            // std::cout << m1.biasesToString() << "\n";
            // std::cout << m1.nablaBToString() << "\n";
            // std::cout << m1.weightsToString() << "\n";
            // std::cout << m1.nablaWToString() << "\n";

            m1.backward(elem);

            // std::cout << "---------------AFTER BACKWARD PASS-------------\n";
            // std::cout << m1.activatedToString() << "\n";
            // std::cout << m1.cacheToString() << "\n";
            // std::cout << m1.nablaCacheToString() << "\n";
            // std::cout << m1.biasesToString() << "\n";
            // std::cout << m1.nablaBToString() << "\n";
            // std::cout << m1.weightsToString() << "\n";
            // std::cout << m1.nablaWToString() << "\n";

            if (epoch == EPOCHS && elem == CsvToVector::features.size() - 1) {
                int t;
                std::cin >> t;
            }
        }
    }
} 