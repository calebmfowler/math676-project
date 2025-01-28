#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {42, 1, -1, 13, 7, 18};

    const auto printer = [](const auto &num) { std::cout << num << ", "; };
    std::for_each (nums.begin(), nums.end(), printer);
    std::cout << std::endl;

    std::vector<std::string> types;

    const auto typewriter = [](const int &num) {
        if (num % 2 == 0)
            return "even";
        else
            return "odd";
    };
    std::transform(std::begin(nums), std::end(nums), std::back_inserter(types), typewriter);

    for (const auto &type : types) {
        std::cout << type << ", ";
    }
    std::cout << std::endl;
}