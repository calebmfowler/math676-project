#include <iostream>

int main() {
    const auto meaningAdder = [](unsigned int &num) { num += 42; };

    unsigned int num = 0;
    std::cout << "initial: " << num << std::endl;
}