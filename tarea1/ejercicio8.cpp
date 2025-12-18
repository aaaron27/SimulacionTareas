#include <iostream>
#include <random>
using namespace std;

/*
Se tiene un juego con una probabilidad de ganar de 0.48 y una probabilidad de perder de 0.52. Cada vez que se apuesta una cantidad de dinero se gana lo que se apuesta o se pierde todo. Se inicia el juego con $100 d lares y se desea llegar a obtener $200.
a. Si se utiliza una estrategia de apuesta (x,x) con x=10. Es decir, si gano apuesto "x" y si pierdo apuesto "x", determine la probabilidad de obtener la cantidad deseada.
b. Si se utiliza una estrategia de apuesta (x,2*x) con x=10. Es decir, si gano apuesto "x", pero si pierdo apuesto el doble de lo que perd  2*x, determine la probabilidad de obtener la cantidad deseada.
*/

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 1.0);

// 1 gana 
// 0 else
bool apuesta() {
    return dist(gen) < 0.48f;
}

double probabilidad(int x) {
    int casos = 100000;
    int objetivo = 200;
    int exitos = 0;
    
    for (int i = 0; i < casos; i++) {
        bool flag = true;
        int dinero = 100;
    
        while (flag) {
            dinero += apuesta() ? x : -x;
            
            if (dinero >= objetivo) {
                exitos++;
                flag = false;
            }

            if (dinero < x) flag = false;
        }
    }

    return static_cast<double>(exitos) / casos;
}

int main() {
    cout << "A: " << probabilidad(10) << '\n';
    cout << "B: " << probabilidad(20) << '\n';

    return 0;
}