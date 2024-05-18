//	author: Benedykt Bela

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <time.h>

using namespace std;



//	ilosc zmiennych typu double, ktore mozemy przechowywac w pamieci GPU
const int rozmiar = 220000000;	// 262135808
//	tablicowa wartosc stalej pi sluzaca do wyliczania bledu
const double pi = 3.1415926535897932;



//	funcja na GPU sluzaca do wypelniania tablicy liczbami ze wzoru Leibniza dla typu float
__global__ void dodawanie_float_fill(float *tab, int rozmiar)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;

	//	jezeli indeks i jest mniejszy od rozmiaru wypelnianej tablicy
	if (i <= rozmiar)
		tab[i - 1] = 4 * pow(-1, i - 1) / (2 * i - 1);
}



//	funkcja na GPU sluzaca do dodawania rownoleglego liczb typu float z tablicy -
//	w miejce tablicy o indeksie i zapisujemy wartosc sumy liczby spod tego indeksu 
//	oraz liczby spod indeksu i + krok
//	i - wartosci od zera do polowy wielkosci zadanego wektora
//	krok - suma i oraz zmiennej krok daje indeks oddalony o polowe dlugosci wektora
__global__ void dodawanie_float(float *tab, int half, int krok)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//	jezeli indeks i jest mniejszy od polowy wektora skladnikow
	if (i < half)
		tab[i] = tab[i] + tab[i + krok];
}



//	ponizsze dwie funkcje sa analogiczne jak powyzsze dwie, ale operuja na zmiennych typu double
__global__ void dodawanie_double_fill(double *tab, int rozmiar)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	if (i <= rozmiar)
		tab[i - 1] = 4 * pow(-1, i - 1) / (2 * i - 1);
}



__global__ void dodawanie_double(double *tab, int half, int krok)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < half)
		tab[i] = tab[i] + tab[i + krok];
}



//	deklaracje funkcji zdefiniowanych w dalszej czesci
void CPU_float();
void CPU_double();
void GPU_float();
void GPU_double();



int main()
{
	cout << endl;

	
	//	zmienne potrzebne do obliczania czasu wykonania danego fragmentu programu
	double start, stop;
	
	//	zmiennej start nadajemy wartosc czasu w momencie rozpoczecia programu
	start = clock();

	//	wykonania programu na GPU oraz zmiennych typu float
	GPU_float();

	//	zmiennej stop nadajemy wartosc czasu po wykonaniu funkcji GPU_float()
	stop = clock();
	//	roznica stop - start daje nam czas wykonania programu co wypisujemy w konsoli
	cout << "Czas wykonania na GPU float:   " << stop - start << endl << endl;


	//	ponizszy kod dziala analogicznie jak ten powyzej z ta ronica, ze wykonuje sie odpowiednio
	//	na GPU dla typu double oraz na CPU dla typu float oraz na CPU dla typu double
	start = clock();

	GPU_double();

	stop = clock();
	cout << "Czas wykonania na GPU double:   " << stop - start << endl << endl;


	start = clock();

	CPU_float();

	stop = clock();
	cout << "Czas wykonania na CPU float:   " << stop - start << endl << endl;


	start = clock();

	CPU_double();

	stop = clock();
	cout << "Czas wykonania na CPU double:   " << stop - start << endl << endl;
	

	cout << endl << endl;
    return 0;
}



//	program do obliczania liczby pi na GPU dla typu float
void GPU_float()
{
	//	tworze wskaznik na tablice zmiennych typu float dla CPU oraz GPU
	float *d_tab = new float[rozmiar];
	float *tab = new float[rozmiar];
	//	alokuje potrzebna pamiec na GPU
	cudaMalloc(&d_tab, rozmiar * sizeof(float));
	//	zmienna blocks mowi nam ile blokow watkow po 1024 watki musimy odpalic zeby wykonac dane zadanie
	int blocks = rozmiar / 1024 + 1;

	//	wypelniam tablice danymi zgodnie ze wzorem Leibniza na GPU
	dodawanie_float_fill << <blocks, 1024 >> > (d_tab, rozmiar);

	//	actual_size zawiera aktualny rozmiar sumowanej tablicy
	int actual_size = rozmiar;
	//	zmienna half zawiera rozmiar polowy sumowanej tablicy zaokraglajac w dol
	int half = actual_size / 2;
	//	zmienna krok zawiera rozmiar polowy sumowanej tablicy zaokraglajac w gore
	int krok = (actual_size + 1) / 2;


	//	wykonuje petle dopoki rozmiar sumowanej tablicy jest wiekszy niz 1
	while (actual_size > 1)
	{
		//	ponizsza konstrukcja if-else zapisuje do zmiennej blocks polowe jej poprzedniej wartosci
		//	zaokraglajac w gore
		if (blocks % 2 == 0)
			blocks = blocks / 2;
		else
			blocks = blocks / 2 + 1;

		//	sumuje po dwa skladniki umieszczajac obliczona sume w miejscu pierwszego skladnika
		dodawanie_float << <blocks, 1024 >> > (d_tab, half, krok);

		//	ponizsza konstrukcja if-else zapisuje aktualne wielkosci sumowanej tablicy
		if (actual_size % 2 == 0)
			actual_size = actual_size / 2;
		else
			actual_size = (actual_size / 2) + 1;

		//	aktualizuje ponizsze zmienne zgodnie z zasada opisana przy deklaracji
		half = actual_size / 2;
		krok = (actual_size + 1) / 2;
	}
	

	//	kopiuje z GPU sume wszystkich skladnikow sumowanego wektora, ktora znajduje sie 
	//	w tym momencie pod adresem pierszego elementu tablicy d_tab
	cudaMemcpy(&tab[0], &d_tab[0], sizeof(float), cudaMemcpyDeviceToHost);


	//	wypisuje wyliczona wartosc liczby pi oraz blad obliczony zgodnie z opisem w instrukcji do laboratorium
	cout <<"GPU float:   "<<setprecision(20)<< tab[0] << endl;
	cout <<"GPU float blad:   "<<setprecision(16)<< ((double)tab[0]-pi)/pi << endl;

	//	zwalniam miejsce wykorzystywane w tej funkcji na GPU
	cudaFree(d_tab);
}



//	analogicznie jak GPU_float, ale dla zmiennych typu double
void GPU_double()
{
	double *d_tab = new double[rozmiar];
	double *tab = new double[rozmiar];
	cudaMalloc(&d_tab, rozmiar * sizeof(double));
	int blocks = rozmiar / 1024 + 1;

	dodawanie_double_fill << <blocks, 1024 >> > (d_tab, rozmiar);

	int actual_size = rozmiar;
	int half = actual_size / 2;
	int krok = (actual_size + 1) / 2;


	while (actual_size > 1)
	{
		if (blocks % 2 == 0)
			blocks = blocks / 2;
		else
			blocks = blocks / 2 + 1;

		dodawanie_double << <blocks, 1024 >> > (d_tab, half, krok);

		if (actual_size % 2 == 0)
			actual_size = actual_size / 2;
		else
			actual_size = (actual_size / 2) + 1;

		half = actual_size / 2;
		krok = (actual_size + 1) / 2;
	}


	cudaMemcpy(&tab[0], &d_tab[0], sizeof(double), cudaMemcpyDeviceToHost);


	cout << "GPU double:   " << setprecision(20) << tab[0] << endl;
	cout << "GPU double blad:   " << setprecision(16) << (tab[0] - pi) / pi << endl;

	cudaFree(d_tab);
}


//	funkcja wykonuje dokladnie to samo co jej odpowiednik na GPU, tylko zamiast obliczen rownoleglych
//	zastosowana jest petla for wykonujaca obliczenie jedno po drugim
void CPU_float() 
{
	float *abc = new float[rozmiar];


	for (int i = 1; i <= rozmiar; i++)
	{
		abc[i-1] = 4 * pow(-1, i - 1) / (2 * i - 1);
	}


	int actual_size = rozmiar;
	int half = actual_size / 2;
	int krok = (actual_size + 1) / 2;

	
	while (actual_size > 1)
	{
		for (int i = 0; i < half; i++)
			abc[i] = abc[i] + abc[i + krok];

		if (actual_size % 2 == 0)
			actual_size = actual_size / 2;
		else
			actual_size = (actual_size / 2) + 1;

		half = actual_size / 2;
		krok = (actual_size + 1) / 2;
	}


	cout << setprecision(20) << "CPU float:    " << abc[0] << endl;
	cout << "CPU float blad:   " << setprecision(16) << ((double)abc[0] - pi) / pi << endl;
}



//	analogicznie jak CPU_float, ale dla zmiennych typu double
void CPU_double()
{
	double *abc = new double[rozmiar];


	for (int i = 1; i <= rozmiar; i++)
	{
		abc[i - 1] = 4 * pow(-1, i - 1) / (2 * i - 1);
	}
	

	int actual_size = rozmiar;
	int half = actual_size / 2;
	int krok = (actual_size + 1) / 2;


	while (actual_size > 1)
	{
		for (int i = 0; i < half; i++)
			abc[i] = abc[i] + abc[i + krok];

		if (actual_size % 2 == 0)
			actual_size = actual_size / 2;
		else
			actual_size = (actual_size / 2) + 1;

		half = actual_size / 2;
		krok = (actual_size + 1) / 2;
	}


	cout << setprecision(20) << "CPU double:    " << abc[0] << endl;
	cout << "CPU double blad:   " << setprecision(16) << (abc[0] - pi) / pi << endl;
}


