#ifndef RANDOM_H
#define RANDOM_H
#include "Setting.h"
#include <cstdlib>

unsigned long long *next_random;

extern "C"
void randReset(INT rank) {
	next_random = (unsigned long long *)calloc(workThreads, sizeof(unsigned long long));
	srand(rank+1);
	for (INT i = 0; i < workThreads; i++)
		next_random[i] = rand();
}

unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

INT rand_max(INT id, INT x, INT size, INT rank) {
	INT part = x / size ;
	INT a = part * rank;
	INT b = part * (rank+1);
	if(rank != size - 1){
		INT res = (randd(id) % (b-a)) + a;
		while (res < a)
			res += (b-a);
		return res;
	}
	else{
		INT res = (randd(id) % (x-a)) + a;
		while (res < a)
			res += (x-a);
		return res;
	}
}

//[a,b)
INT rand(INT a, INT b){
	return (rand() % (b-a))+ a;
}
#endif
