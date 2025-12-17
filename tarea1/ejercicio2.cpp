#include <iostream>
#include <cmath>
using namespace std;

typedef long long int lli;

/*
Se tienen 3 monedas. Las cuales pueden dar como resultado escudo o corona.
*/

const int maxn = 1000;
int C[maxn+1][maxn+1];

void fillC() {
    C[0][0]++;
    for (int n = 1; n <= maxn; n++) {
        C[n][0] = C[n][n] = 1;
        for (int k = 1; k < n; k++) C[n][k] = C[n-1][k-1] + C[n-1][k];
    }
}

// 1 -> corona
// 0 -> escudo
bool moneda() {
    return rand()%2;
}

//a. Utilizando probabilidad matematica calcule el valor de obtener dos coronas.
double ejercicioA() {
    fillC();
    return C[3][2] * pow(0.5, 2) * 0.5f;
}


// b. Con un programa de simulaci n, calcule la probabilidad de obtener 0 coronas.
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

// e. Con un programa de simulaci n, calcule la probabilidad de obtener 3 coronas.
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

    cout << "Ejercicio A\n" << ejercicioA() << '\n';
    cout << "Ejercicio B\n" << ejercicioB() << '\n';
    cout << "Ejercicio C\n" << ejercicioC() << '\n';
    cout << "Ejercicio D\n" << ejercicioD() << '\n';
    cout << "Ejercicio E\n" << ejercicioE() << '\n';
    
    return 0;
}