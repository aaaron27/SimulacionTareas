#include <iostream>
#include <cmath>
using namespace std;

typedef long long int lli;

/*
Se tienen 3 monedas. Las cuales pueden dar como resultado escudo o corona.
*/

// 1 -> corona
// 0 -> escudo
bool moneda() {
    return rand()%2;
}

//a. Utilizando probabilidad matematica calcule el valor de obtener dos coronas.
double ejercicioA() {
    return 0.375f;
}


// b. Con un programa de simulacion, calcule la probabilidad de obtener 0 coronas.
double ejercicioB() {
    int casos = 10000000;
    int exitos = 0;

    for (int i = 0; i < casos; i++) {
        if (!(moneda() + moneda() + moneda()))
            exitos++;
    }

    return static_cast<double>(exitos) / casos;
}

// c. Con un programa de simulacion, calcule la probabilidad de obtener 1 coronas.
double ejercicioC() {
    int casos = 10000000;
    int exitos = 0;

    for (int i = 0; i < casos; i++) {
        if (moneda() + moneda() + moneda() == 1)
            exitos++;
    }

    return static_cast<double>(exitos) / casos;
}

// d. Con un programa de simulacin, calcule la probabilidad de obtener 2 coronas.
double ejercicioD() {
    int casos = 10000000;
    int exitos = 0;

    for (int i = 0; i < casos; i++) {
        if (moneda() + moneda() + moneda() == 2)
            exitos++;
    }

    return static_cast<double>(exitos) / casos;
}

// e. Con un programa de simulacion, calcule la probabilidad de obtener 3 coronas.
double ejercicioE() {
    int casos = 10000000;
    int exitos = 0;

    for (int i = 0; i < casos; i++) {
        if (moneda() + moneda() + moneda() == 3)
            exitos++;
    }

    return static_cast<double>(exitos) / casos;
}

int main() {
    // esta vara reinicia la seed porque sin esto siempre daria la misma secuencia
    srand(time({}));

    cout << "A: " << ejercicioA() << '\n';
    cout << "B: " << ejercicioB() << '\n';
    cout << "C: " << ejercicioC() << '\n';
    cout << "D: " << ejercicioD() << '\n';
    cout << "E: " << ejercicioE() << '\n';
    cout << "F: Distribucion Binomial\n";
    
    return 0;
}