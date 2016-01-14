#include "stdh.h"
#include "util.h"
#include "dtree.h"
#include "knn.h"

Knn::Knn(){
	model = new Knnmodel;	
	model->dist = 0;
	model->k = 1;
}

Knn::Knn(int k){
	model = new Knnmodel;	
	model->dist = 0;
	model->k = k;
	model->traindata = NULL;
}

Knn::~Knn(){
	delete[] model->dist;
	delete model;
}

T Knn :: train(const char* fname){
	data = new Data();
	data->loadData(fname);
	return train();
}

void Knn::saveModel(const char* fname){
	FILE* f = fopen(fname, "w");
}
void Knn::loadModel(const char* fname){
	FILE* f = fopen(fname, "r");
}

T Knn::train(){
	//copiem data in train data
	model->traindata = data;
	return 0;
}
typedef struct _lbldist {
	float dist;
	int label;

} lbldist;
bool compareByDistance(const lbldist &a, const lbldist &b)
{
    return a.dist < b.dist;
}
float Knn::predict(T* sample) {
	int label = 0;
	int lbl0=0, lbl1=0;
	//comparam cu fiecare instanta din train data care are weight!=0; returnez eticheta, -1 sau 1
	float sum = 0;
	std::vector<lbldist> distances;
	lbldist ld;
	for(int i = 0; i < model->traindata->N; i++) {
		sum = 0;
		if (model->traindata->w[i] != 0) {
		for (long long j = 0; j < model->traindata->M; j++) {
			sum += pow(sample[j] - model->traindata->d[i*model->traindata->M + j], 2); 
			 }
		}
		ld.dist = sqrt(sum);
		ld.label = model->traindata->l[i];
		distances.push_back(ld);
	}
	sort(distances.begin(), distances.end(), compareByDistance);
	for (int i = 0; i< model->k ; i++) {
		if (distances.at(i).label == -1)
			lbl0++;
		else lbl1++;
	}
	if (lbl0 > lbl1)
		return -1;
	else return 1;
}