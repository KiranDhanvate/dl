#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {

    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);

    cout << "Enter elements:\n";

    for(int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    int sum = 0;
    int minVal = arr[0];
    int maxVal = arr[0];

    // Parallel Reduction
    #pragma omp parallel for reduction(+:sum) reduction(min:minVal) reduction(max:maxVal)
    for(int i = 0; i < n; i++) {

        sum += arr[i];

        if(arr[i] < minVal)
            minVal = arr[i];

        if(arr[i] > maxVal)
            maxVal = arr[i];
    }

    double avg = (double)sum / n;

    cout << "\nMinimum = " << minVal;
    cout << "\nMaximum = " << maxVal;
    cout << "\nSum = " << sum;
    cout << "\nAverage = " << avg << endl;

    return 0;
}