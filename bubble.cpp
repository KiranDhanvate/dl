#include <iostream>
#include <omp.h>
using namespace std;

// Swap function
void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

// Sequential Bubble Sort
void sequentialBubbleSort(int a[], int n)
{
    for (int i = 0; i < n; i++)
    {
        bool swapped = false;

        for (int j = 0; j < n - 1; j++)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

// Parallel Bubble Sort (Odd-Even Sort)
void parallelBubbleSort(int a[], int n)
{
    for (int i = 0; i < n; i++)
    {
        int first = i % 2;

        #pragma omp parallel for
        for (int j = first; j < n - 1; j += 2)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int a[100], b[100];

    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
        b[i] = a[i]; // copy array
    }

    double start, end;

    // Sequential
    start = omp_get_wtime();
    sequentialBubbleSort(a, n);
    end = omp_get_wtime();

    cout << "\nSequential Sorted Array:\n";
    for (int i = 0; i < n; i++)
        cout << a[i] << " ";

    cout << "\nTime: " << end - start << endl;

    // Parallel
    start = omp_get_wtime();
    parallelBubbleSort(b, n);
    end = omp_get_wtime();

    cout << "\nParallel Sorted Array:\n";
    for (int i = 0; i < n; i++)
        cout << b[i] << " ";

    cout << "\nTime: " << end - start << endl;

    return 0;
}
