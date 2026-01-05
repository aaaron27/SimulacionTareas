#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int N = 1000000;
const char FILE_PATH1[] = "../data/C1.txt";
const char FILE_PATH2[] = "../data/C2.txt";

void generating() {
    FILE *f = fopen(FILE_PATH1, "w");
    FILE *f2 = fopen(FILE_PATH2, "w");

    for (int i = 0; i < N; i++) {
        if (i < N-1) {
            fprintf(f, "%d\n", rand() % 4 + 1);
            fprintf(f2, "%d\n", rand() % 8 + 1);
        } else {
            fprintf(f, "%d", rand() % 4 + 1);
            fprintf(f2, "%d", rand() % 8 + 1);
        }
    }

    fclose(f);
    fclose(f2);
}

int main() {
    srand(time(NULL));

    generating();

    return 0;
}