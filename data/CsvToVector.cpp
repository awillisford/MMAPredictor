#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include "../include/CsvToVector.hpp"

// declare class members
std::vector<std::vector<float>> CsvToVector::features;
std::vector<std::vector<float>> CsvToVector::labels;

bool CsvToVector::contains_number(const std::string& str) {
    /* string::npos is returned by method find when the digit/digits
     * were not found; therefore, if any of the digits are found, 
     * it will return that digit and be != to string::npos, which returns true */

    // return true if number found in string, else return false
    return (str.find_first_of("0123456789") != std::string::npos);
}

float CsvToVector::contains_text(const std::string& str) {
    // fighter color winner
    if (str == "Blue") {
        return -1;
    }
    else if (str == "Red") {
        return -2;
    }
    // fighter stance
    else if (str == "Orthodox") {
        return 0;
    }
    else if (str == "Southpaw") {
        return 1;
    }
    else if (str == "Switch") {
        return 2;
    }
    else if (str == "Open Stance") {
        return 3;
    }
    // title fight
    else if (str == "FALSE") {
        return 0;
    }
    else if (str == "TRUE") {
        return 1;
    }
    // weight divisions
    else if (str == "Flyweight") {
        return 0;
    }
    else if (str == "Bantamweight") {
        return 1;
    }
    else if (str == "Featherweight") {
        return 2;
    }
    else if (str == "Lightweight") {
        return 3;
    }
    else if (str == "Welterweight") {
        return 4;
    }
    else if (str == "Middleweight") {
        return 5;
    }
    else if (str == "LightHeavyweight") {
        return 6;
    }
    else if (str == "Heavyweight") {
        return 7;
    }
    else if (str == "CatchWeight") {
        return 8;
    }
    else if (str == "WomenStrawweight") {
        return 9;
    }
    else if (str == "WomenFlyweight") {
        return 10;
    }
    else if (str == "WomenBantamweight") {
        return 11;
    }
    else if (str == "WomenFeatherweight") {
        return 12;
    }

    // throw exception if text is not any of the above
    else {
        try {
            throw 999;
        }
        catch(int e) {
            // std::cout << "An exception occured in function \"contains_text()\", exception no. "
            //           << e << '\n';
        }
        return 999;
    }
}

float CsvToVector::check_cell_type(const std::string& cell) {
    // if cell contains a number return type float of string
    if (contains_number(cell)) {
        // std::cout << "cell value: " << std::stof(cell) << ": ";
        return std::stof(cell);
    }
    // return float value assigned to cell given by contains_text()
    else {
        // std::cout << "cell value: '" << cell << "': ";
        return contains_text(cell);
    }
}

bool CsvToVector::compare_floats(float& a, float b, float epsilon) {
    // std::cout << "fabs(" << a << " - " << b << ") = " << fabs(a - b) << ", " << "ret: " << (fabs(a - b) < epsilon) << " - ";
    return fabs(a - b) < epsilon;
}

void CsvToVector::extract_data(const std::string& csvFile) {
    std::ifstream fin(csvFile); // input stream
    std::string line; // declare string to hold each line
    bool header = true;

    // loop through csv line by line
    while(std::getline(fin, line)) {
        // if header then continue to next interation
        if (header) {
            header = false;
            continue;
        }
        std::vector<float> temp_vector; // create vector to push back in features
        std::stringstream ss(line);
        std::string cell; // holds data in cell

        // iterate through cells in line
        while(std::getline(ss, cell, ',')) {
            float cell_checked = check_cell_type(cell);
            // std::cout << "cell_checked: " << cell_checked << ", ";

            if (compare_floats(cell_checked, -1)) {
                std::vector<float> blue = {1, 0};
                labels.push_back(blue);
                // std::cout << "Labels push_back: blue - {1, 0} " << '\n';
                continue;
            }
            else if (compare_floats(cell_checked, -2)) {
                std::vector<float> red = {0, 1};
                labels.push_back(red);
                // std::cout << "Labels push_back: red - {0, 1} " << '\n';
                continue;
            }
            else {
                temp_vector.push_back(cell_checked);
                // std::cout << "temp_vector push_back: " << cell_checked << '\n';

            }
        }
        features.push_back(temp_vector); // push vector containing row into features
        // std::cout << "Features: push_back(temp_vector)" << '\n';
    }
}

void CsvToVector::print_features() {
    std::cout << "["; // show beginning of vector
    // iterate through vector
    for (int x = 0; x < features.size(); ++x) {
        // add spacing to align brackets 
        if (x > 0) {
            std::cout << " ["; // show beginning of new vector
        }
        else {
            std::cout << "["; // show beginning of new vector
        }
        // iterate through subvectors
        for (int y = 0; y < features[x].size(); ++y) {
            // if not last element
            if (y < features[x].size() - 1) {
                std::cout << features[x][y] << ", ";
            }
            // last element
            else {
                std::cout << features[x][y];
            }
        }
        // if not last row
        if (x < features.size() - 1) {
            std::cout << "],\n"; // end vector and print newline
        }
        else {
            std::cout << "]"; // end of last vector
        }
    }
    std::cout << "]"; // show end of vector
}