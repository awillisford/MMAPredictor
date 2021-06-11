#pragma once

#include <vector>
#include <string>

class CsvToVector {
    private:
    // TEMP COMMENT -- UPDATE
    static bool contains_number(const std::string& str);

    // assign float values to each of the different text
    static float contains_text(const std::string& str);

    // TEMP COMMENT -- UPDATE
    static float check_cell_type(const std::string& cell);

    // TEMP COMMENT -- UPDATE
    static bool compare_floats(float& a, float b, float epsilon = 0.01f);

    public:
    // TEMP COMMENT -- UPDATE
    static void extract_data(const std::string& csvFile);

    // class members
    static std::vector<std::vector<float>> features;
    static std::vector<std::vector<float>> labels;
};
