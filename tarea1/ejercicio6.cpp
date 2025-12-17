#include <iostream>
#include <random>
using namespace std;

typedef long long int lli;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 1.0);

const int maxn = 100;
int C[maxn+1][maxn+1];

void fillC() {
    C[0][0]++;
    for (int n = 1; n <= maxn; n++) {
        C[n][0] = C[n][n] = 1;
        for (int k = 1; k < n; k++) C[n][k] = C[n-1][k-1] + C[n-1][k];
    }
}

// 1 -> es as
// 0 -> else
bool isAs(int ases, int n) {
    return dist(gen) <= static_cast<double>(ases) / n;
}

double a() {
    fillC();
    return static_cast<double>(C[4][2] * C[48][3]) / C[52][5];
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