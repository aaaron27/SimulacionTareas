#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
using namespace std;

/*
Se tiene un mazo de cartas convencional, el cual ha sido barajdo. Se reparten 5 cartas y se desea establecer la probabilidad que las 5 cartas sumen entre 17 y 21.
Resuelva el problema construyendo un programa de simulacion.
*/

random_device rd;
mt19937 gen(rd());

vector<int> baraja = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
};

int sumCartas() {
    shuffle(baraja.begin(), baraja.end(), gen);
    int asses=0;
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        if(baraja[i]!=1){sum += baraja[i];}
        else {
            asses++;
        }
    }
    while(asses--) {
        if(sum+14<=21) {
            sum+=14;
        }
        else{sum++;}
    }

    return sum;
}

int main() {
    int casos = 100000;
    int exitos = 0;
    
    for (int i = 0; i < casos; i++) {
        int sum = sumCartas();
        if (sum >= 17 && sum <= 21) exitos++;
    }

    cout << "A: " << static_cast<double>(exitos) / casos << '\n';

    return 0;
}