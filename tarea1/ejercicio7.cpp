#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
using namespace std;

random_device rd;
mt19937 gen(rd());

int sumCartas() {
    vector<int> baraja = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10,
    };
    
    shuffle(baraja.begin(), baraja.end(), gen);

    int sum = 0;
    for (int i = 0; i < 5; i++) sum += baraja[i];

    return sum;
}

int main() {
    int casos = 1000000;
    int exitos = 0;
    
    for (int i = 0; i < casos; i++) {
        int sum = sumCartas();
        if (sum >= 17 && sum <= 21) exitos++;
    }

    cout << static_cast<double>(exitos) / casos << '\n';

    return 0;
}