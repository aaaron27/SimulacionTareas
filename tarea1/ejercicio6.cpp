#include <iostream>
#include <random>
using namespace std;

/*
Se tiene un mazo de cartas convencional, el cual ha sido barajdo. Se reparten 5 cartas y se desea establecer la probabilidad que dentro de esas 5 cartas se encuentre un par de ases.
a. Resuelva el problema utilizando probabilidad cl sica.
b. Resuelva el problema construyendo un programa de simulacin.
*/

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 1.0);

// 1 -> es as
// 0 -> else
bool isAs(int ases, int n) {
    return dist(gen) <= static_cast<double>(ases) / n;
}

double a() {
    return 103776.0f / 2598960;
}

double b(){
    int casos = 1000000;
    int exitos = 0;
    
    for (int i = 0; i < casos; i++) {
        int ases = 4;
        int n = 52;

        for (int j = 0; j < 5; j++) {
            if (isAs(ases, n)) {
                ases--;
            }
            n--;
        }

        if (ases == 2) exitos++;
    }

    return static_cast<double>(exitos) / casos;
}

int main() {
    cout << "A: " << a() << '\n';
    cout << "B: " << b() << '\n';

    return 0;
}