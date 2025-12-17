#include <iostream>
#include <random>
using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 1.0);

// 1 gana 
// 0 else
bool apuesta() {
    return dist(gen) < 0.48f;
}

double probabilidad(int x) {
    int casos = 1000000;
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