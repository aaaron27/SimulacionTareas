#include <bits/stdc++.h>
using namespace std;
#define lli long long int
#define vi vector<int>
#define vlli vector<lli>
#define vvlli vector<vlli>
#define vc vector<char>
#define vs vector<string>
#define yatogami ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
#define tohka lli t; cin>>t; while(t--)
#define miku lli n; cin >> n;
#define nakano lli m; cin >> m;
#define ai(i,a,b) for(i=a;i<b;i++)
#define ichijou(i,a,b) for(i=a;i>b;i--)
#define renako(x) cout << x << endl;
#define amaori(x) cout << x << ' ';
const double S1 = 0.90;
const double S2 = 0.90;
const double S1_F = 0.80;
const double S2_F = 0.70;
const double P = 0.80;
const double S2_M = 0.80;

mt19937 rng(std::random_device{}());
uniform_real_distribution<double> dist(0.0, 1.0);

bool is_active(double probability) {
    return dist(rng) < probability;
}
double a(lli exitos, lli simulaciones) {
    lli i;
    ai(i,0,simulaciones) {
        bool link_S1 = is_active(S1);
        bool link_S2 = is_active(S2);
        bool link_S1_F = is_active(S1_F);
        bool link_S2_F = is_active(S2_F);
        bool path_upper = link_S1 && link_S1_F;
        bool path_lower = link_S2 && link_S2_F;
        if (path_upper || path_lower) {
            exitos++;
        }
    }
    double probabilidad_calculada = (double)exitos / simulaciones;
    return probabilidad_calculada;
}

double b(lli exitos, lli simulaciones) {
    lli i;
    ai(i,0,simulaciones) {
        bool link_S1 = is_active(S1);
        bool link_S2 = is_active(S2);
        bool link_S1_F = is_active(S1_F);
        bool link_S2_F = is_active(S2_M);
        bool path_upper = link_S1 && link_S1_F;
        bool path_lower = link_S2 && link_S2_F;
        if (path_upper || path_lower) {
            exitos++;
        }
    }
    double probabilidad_calculada = (double)exitos / simulaciones;
    return probabilidad_calculada;
}

double c(lli exitos, lli simulaciones) {
    lli i;
    ai(i,0,simulaciones) {
        bool link_T_S1 = is_active(S1);
        bool link_T_S2 = is_active(S2);
        bool link_S1_F = is_active(S1_F);
        bool link_S2_F = is_active(S2_F);
        bool link_Bridge = is_active(P);
        bool path_upper = link_T_S1 && link_S1_F;
        bool path_lower = link_T_S2 && link_S2_F;
        bool path_cross_1 = link_T_S1 && link_Bridge && link_S2_F;
        bool path_cross_2 = link_T_S2 && link_Bridge && link_S1_F;
        if (path_upper || path_lower || path_cross_1 || path_cross_2) {
            exitos++;
        }
    }
    double probabilidad_calculada = (double)exitos / simulaciones;
    return probabilidad_calculada;
}

int main() {
    lli exitos = 0,i;
    lli simulaciones;
    renako("Cuantas simulaciones quiere que se den?")
    cin >> simulaciones;
    double proba = a(0,simulaciones), probb = b(0,simulaciones), probc = c(0, simulaciones);
    std::cout << "--- Resultados de la Simulacion con " << simulaciones << " Simulaciones " << std::endl;

    std::cout << "Probabilidad simulada (Ejercicio A): " << proba << std::endl;
    // Calculo teorico para A:
    // Ruta arriba: 0.9 * 0.8 = 0.72 -> P(Fallo) = 0.28
    // Ruta abajo:  0.9 * 0.7 = 0.63 -> P(Fallo) = 0.37
    // Fallo total: 0.28 * 0.37 = 0.1036
    // Exito total: 1 - 0.1036 = 0.8964
    std::cout << "Probabilidad teorica (Ejercicio A):   0.8964" << std::endl;
    std::cout << "Diferencia en el ejercicio A: " << std::abs(proba - 0.8964) << std::endl;


    std::cout << "Probabilidad simulada (Ejercicio B): " << probb << std::endl;
    // Calculo teorico para B: 1 - (ProbFalloArriba * ProbFalloAbajo)
    // Ruta arriba: 0.9*0.8 = 0.72 -> Fallo: 0.28
    // Ruta abajo:  0.9*0.8 = 0.72 -> Fallo: 0.28
    // Fallo total: 0.28 * 0.28 = 0.0784
    // Exito total: 1 - 0.0784 = 0.9216
    std::cout << "Probabilidad teorica (Ejercicio B):   0.9216" << std::endl;
    std::cout << "Diferencia en el ejercicio B: " << std::abs(probb - 0.9216) << std::endl;


    std::cout << "Probabilidad simulada (Ejercicio C): " << probc << std::endl;

    std::cout << "Probabilidad teorica (Ejercicio C):   0.92376" << std::endl;
    std::cout << "Diferencia en el ejercicio C: " << std::abs(probc - 0.92376) << std::endl;

    return 0;
}
