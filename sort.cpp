#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// ---------------- BUBBLE SORT ----------------

// Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for(int i = 0; i < n - 1; i++) {
        for(int j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for(int i = 0; i < n; i++) {

        #pragma omp parallel for
        for(int j = i % 2; j < n - 1; j += 2) {
            if(arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// ---------------- MERGE SORT ----------------

void merge(vector<int>& arr, int l, int m, int r) {

    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for(int i = 0; i < n1; i++)
        L[i] = arr[l + i];

    for(int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while(i < n1 && j < n2) {
        if(L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while(i < n1)
        arr[k++] = L[i++];

    while(j < n2)
        arr[k++] = R[j++];
}

// Sequential Merge Sort
void sequentialMergeSort(vector<int>& arr, int l, int r) {
    if(l < r) {
        int m = (l + r) / 2;

        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int l, int r) {

    if(l < r) {

        int m = (l + r) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);

            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

// ---------------- MAIN ----------------

int main() {

    vector<int> arr = {9, 5, 1, 4, 3, 8, 2, 7, 6};

    vector<int> a1 = arr;
    vector<int> a2 = arr;
    vector<int> a3 = arr;
    vector<int> a4 = arr;

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    sequentialBubbleSort(a1);
    end = omp_get_wtime();

    cout << "Sequential Bubble Sort Time: "
         << end - start << " sec\n";

    // Parallel Bubble Sort
    start = omp_get_wtime();
    parallelBubbleSort(a2);
    end = omp_get_wtime();

    cout << "Parallel Bubble Sort Time: "
         << end - start << " sec\n";

    // Sequential Merge Sort
    start = omp_get_wtime();
    sequentialMergeSort(a3, 0, a3.size() - 1);
    end = omp_get_wtime();

    cout << "Sequential Merge Sort Time: "
         << end - start << " sec\n";

    // Parallel Merge Sort
    start = omp_get_wtime();
    parallelMergeSort(a4, 0, a4.size() - 1);
    end = omp_get_wtime();

    cout << "Parallel Merge Sort Time: "
         << end - start << " sec\n";

    return 0;
}