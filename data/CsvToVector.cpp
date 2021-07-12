#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include "../include/CsvToVector.hpp"

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
        return 0.25;
    }
    else if (str == "Southpaw") {
        return 0.5;
    }
    else if (str == "Switch") {
        return 0.75;
    }
    else if (str == "Open Stance") {
        return 1;
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
        return 0.076923;
    }
    else if (str == "Bantamweight") {
        return 0.153846;
    }
    else if (str == "Featherweight") {
        return 0.230769;
    }
    else if (str == "Lightweight") {
        return 0.307692;
    }
    else if (str == "Welterweight") {
        return 0.384615;
    }
    else if (str == "Middleweight") {
        return 0.461538;
    }
    else if (str == "LightHeavyweight") {
        return 0.538462;
    }
    else if (str == "Heavyweight") {
        return 0.615385;
    }
    else if (str == "CatchWeight") {
        return 0.692308;
    }
    else if (str == "WomenStrawweight") {
        return 0.769231;
    }
    else if (str == "WomenFlyweight") {
        return 0.846154;
    }
    else if (str == "WomenBantamweight") {
        return 0.923077;
    }
    else if (str == "WomenFeatherweight") {
        return 1;
    }

    // throw exception if text is not any of the above
    else {
        try {
            throw 999;
        }
        catch(int e) {
            std::cout << "An exception occured in function \"contains_text()\", exception no. "
                      << e << '\n';
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

std::vector<std::vector<std::vector<float>>> CsvToVector::extract_data(const std::string& csvFile) {
    std::vector<std::vector<std::vector<float>>> data; // holds data
    std::vector<std::vector<float>> temp;
    // push vectors to hold features [0] and labels [1]
    for (int x : {0, 1})
        data.push_back(temp);

    std::vector<float> blue = {1, 0}; // blue winner
    std::vector<float> red = {0, 1}; // red winner
    std::ifstream fin(csvFile); // input stream
    std::string line; // declare string to hold each line
    bool header = true; // to skip header

    // loop through csv line by line
    while(std::getline(fin, line)) {
        // if header then continue to next interation
        if (header) {
            header = false;
            continue;
        }
        std::vector<float> row; // create vector to hold data from row
        std::stringstream ss(line);
        std::string cell; // holds data in cell

        // iterate through cells in line
        while(std::getline(ss, cell, ',')) {
            float cell_checked = check_cell_type(cell);
            if (compare_floats(cell_checked, -1)) {
                data[1].push_back(blue);
                continue;
            }
            else if (compare_floats(cell_checked, -2)) {
                data[1].push_back(red);
                continue;
            }
            else {
                row.push_back(cell_checked);
            }
        }
        data[0].push_back(row); // push vector containing row into features
    }
    return data;
}
