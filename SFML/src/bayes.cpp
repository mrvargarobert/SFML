#include "stdh.h"
#include "util.h"
#include "dtree.h"
#include "bayes.h"
#define _USE_MATH_DEFINES
#include <math.h>

Bayes::Bayes(){
	model = new Bayesmodel;	

}

Bayes::~Bayes(){
	delete model;
}
void Bayes::saveModel(const char* fname){
	FILE* f = fopen(fname, "w");
	fclose(f);
}
void Bayes::loadModel(const char* fname){
	FILE* f = fopen(fname, "r");
	fclose(f);
}
T Bayes::train(){
	//copiem data in train data
	model->traindata = data;
	return 0;
}

float Bayes::predict(T* sample) {

	float p0 = 0, p1 = 0, c0 = 0 , c1 = 0, prob0 = 1, prob1 = 1;
	float *x0, *x1, *s0, *s1, *pd0, *pd1;
	x0 = new float[model->traindata->M]; //means for each feature class label -1;
	x1 = new float[model->traindata->M]; //means for each feature class label 1;
	s0 = new float[model->traindata->M]; //variance for each feature class label -1;
	s1 = new float[model->traindata->M]; //variance for each feature class label 1;
	pd0 = new float[model->traindata->M]; //probability distribution for each feature for label -1
	pd1 = new float[model->traindata->M]; //probability distribution for each featrue for label 1;

	for (int i = 0; i < model->traindata->M; i++) {
		x0[i] = 0;
		x1[i] = 0;
		s0[i] = 0;
		s1[i] = 0;
		pd0[i] = 0;
		pd1[i] = 0;
	}
	for(int i = 0; i < model->traindata->N; i++)
	{
		if (model->traindata->w[i] != 0) {

			for (long long j = 0; j < model->traindata->M; j++) {
				if (model->traindata->l[i] == 1)
					x1[j] += model->traindata->d[i*model->traindata->M + j];
				else 
					x0[j] += model->traindata->d[i*model->traindata->M + j];
			}
		}
			if (model->traindata->l[i] == 1)
				c1++;
			else 
				c0++;
	}
	for (int i = 0; i < model->traindata->M; i++) {
		x0[i] = x0[i]/c0;
		x1[i] = x1[i]/c1;
	}

	for(int i = 0; i < model->traindata->N; i++)
	{
		if (model->traindata->w[i] != 0) {

			for (long long j = 0; j < model->traindata->M; j++) {
				if (model->traindata->l[i] == 1)
					s1[j] += pow(model->traindata->d[i*model->traindata->M + j] - x1[j], 2);
				else 
					s0[j] += pow(model->traindata->d[i*model->traindata->M + j] - x0[j], 2);
			}
		}
	}
	for (int i = 0; i < model->traindata->M; i++) {
		//compute variance
		s0[i] = s0[i]/c0;
		s1[i] = s1[i]/c1;
		//compute likelyhood
		pd0[i] = (1.0/sqrt(2 * M_PI * s0[i])) * exp(- (pow(sample[i] - x0[i], 2))/(2 * s0[i]));
		pd1[i] = (1.0/sqrt(2 * M_PI * s1[i])) * exp(- (pow(sample[i] - x1[i], 2))/(2 * s1[i]));
		//compute prducts of likelyhoods
		prob0 *= pd0[i];
		prob1 *= pd1[i];

	}	
	p0 = c0/model->traindata->N;
	p1 = c1/model->traindata->N;
	
	if (p0*prob0 > p1*prob1) 
		return -1;
	else 
		return 1;

}