#include "../src/llab.h"


int main(){
	float* p = (float*)malloc(sizeof(float)*9);
	int* v = (int*)malloc(sizeof(int)*3);
	v[0] = 0;
	v[1] = 1;
	v[2] = 2;
	p[0] = 0;
	p[0] = 0.8;
	p[0] = 0.7;
	p[0] = 0.2;
	p[0] = 0.6;
	p[0] = 0.3;
	p[0] = 0.4;
	merge_sort_for_probabilities(p, v,0,2,9);
	int i;
	for(i = 0; i < 3; i++){
		printf("%d\n",v[i]);
	}
}
