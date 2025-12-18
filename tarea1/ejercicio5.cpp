#include <iostream>
#include <random>
using namespace std;

/*
Se tiene un mazo de cartas convencional.
a. Cual es la probabilidad que al sacar una carta se obtenga una espada o un trebol?
b. Construya un programa de simulaci n que genere estos resultados.
*/

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 1.0);

// 1 -> trevol o espada
// 0 -> else
bool trevolOEspada() {
    return dist(gen) <=  0.5f;
}

int main() {
    int casos = 1000000;
    int exitos = 0;

    for (int i = 0; i < casos; i++) {
        bool isTrevolOrEspada = trevolOEspada();

        exitos += isTrevolOrEspada
            ? 1 
            : 0;
    }


    cout << "A: 1/2\n";
    cout << "B: " << static_cast<double>(exitos) / casos << '\n';

    return 0;
}