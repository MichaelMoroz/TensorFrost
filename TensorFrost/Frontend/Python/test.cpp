#include <vector>

std::vector<int> add_one(const std::vector<int>& v) {
    std::vector<int> result;
    for (int i : v) {
        result.push_back(i + 2);
    }
    return result;
}