#include <iostream>
#include <time.h>
using namespace std;

void main() {
	int** arr1 = new int*[11000];
	int** arr2 = new int*[11000];

	for (int i = 0; i < 11000; ++i) {
		arr1[i] = new int[11000];
		arr2[i] = new int[11000];
		for (int j = 0; j < 11000; ++j) {
			arr1[i][j] = 0;
			arr2[i][j] = 1;
		}
	}

	clock_t t = clock();
	int** toReturn = new int*[11000];
	for (int i = 0; i < 11000; ++i)
	{
		toReturn[i] = new int[22000];
	}

	for (int i = 0; i < 11000; ++i)
	{

		for (int j = 0; j < 11000; ++j)
		{
			toReturn[i][j] = arr1[i][j];
		}

		for (int k = 0; k < 11000; ++k)
		{
			toReturn[i][11000 + k] = arr2[i][k];
		}

	}

	t = clock() - t;

	cout << t;

	int k;
	cin >> k;
}
