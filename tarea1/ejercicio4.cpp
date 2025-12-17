#include <iostream>
#include <random>
using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 1.0);

// 1 -> es as
// 0 -> else
bool isAs(int ases, int n) {
    return dist(gen) <= static_cast<double>(ases) / n;
}

int main() {
    int n = 52;
    int ases = 4;
    int casos = 100000000;
    int exitos = 0;

    for (int i = 0; i < casos; i++) {
        bool as1 = isAs(ases, n);
        bool as2 = isAs(ases-1, n-1);

        exitos += as1 && as2 
            ? 1 
            : 0;
    }

    cout << static_cast<double>(exitos) / casos << '\n';

    return 0;
}