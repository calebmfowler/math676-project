#include <iostream>

namespace myNamespace {
    void print() {
        std::cout << std::endl;
    }
}

namespace myOtherNamespace {
    void print() {
        std::cout << std::endl;
    }
}

void print() {
    std::cout << std::endl;
}

int main() {
    using namespace std;

    print();
}
