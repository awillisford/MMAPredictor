#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    CsvToVector::extract_data("data/data_filtered.csv");

    Model m1(1, 1, 0.03);

    int EPOCHS = 3;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        for (int elem = 0; elem < CsvToVector::features.size(); ++elem) {
            // std::cout << "features size: " << CsvToVector::features.size() << '\n';
            // std::cout << "elem=" << elem << "\n";

            std::cout << "---------------BEFORE FORWARD PASS-------------\n";
            std::cout << m1.activatedToString() << "\n";
            std::cout << m1.cacheToString() << "\n";
            std::cout << m1.nablaCacheToString() << "\n";
            std::cout << m1.biasesToString() << "\n";
            std::cout << m1.nablaBToString() << "\n";
            std::cout << m1.weightsToString() << "\n";
            std::cout << m1.nablaWToString() << "\n";

            m1.forward(CsvToVector::features[elem]);

            std::cout << "---------------AFTER FORWARD PASS-------------\n";
            std::cout << m1.activatedToString() << "\n";
            std::cout << m1.cacheToString() << "\n";
            std::cout << m1.nablaCacheToString() << "\n";
            std::cout << m1.biasesToString() << "\n";
            std::cout << m1.nablaBToString() << "\n";
            std::cout << m1.weightsToString() << "\n";
            std::cout << m1.nablaWToString() << "\n";

            m1.backward(elem);

           /* pDerivCost/pDerivOutput = nablaW = Output elem - cost elem
            *
            * Weights is the exact same output as nablaW, so something is def wrong
            *
            */


            std::cout << "---------------AFTER BACKWARD PASS-------------\n";
            std::cout << m1.activatedToString() << "\n";
            std::cout << m1.cacheToString() << "\n";
            std::cout << m1.nablaCacheToString() << "\n";
            std::cout << m1.biasesToString() << "\n";
            std::cout << m1.nablaBToString() << "\n";
            std::cout << m1.weightsToString() << "\n";
            std::cout << m1.nablaWToString() << "\n";

            int t;
            std::cin >> t;
        }
    }
} 