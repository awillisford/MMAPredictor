#pragma once

#include <vector>
#include <string>

class CsvToVector {
    private:
    // determine if cell contains number values or not
    static bool contains_number(const std::string& str);

    // determine if cell contains text or not, and assigns value to text
    static float contains_text(const std::string& str);

    // sends cell values through contains_number() or contains_text(), and returns new float value
    static float check_cell_type(const std::string& cell);

    // used for comparing difference between different float values, and placement in either features or labels
    static bool compare_floats(float& a, float b, float epsilon = 0.01f);

    public:
    // takes data from csv and puts data into matrix of float values
    static void extract_data(const std::string& csvFile);


    // class members
    static std::vector<std::vector<float>> features;
    static std::vector<std::vector<float>> labels;
};
