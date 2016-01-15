#include "data.h"
#include "util.h"
#include "boosting.h"
#include "knn.h"
#include "bayes.h"

int main(){
	//if separate training and test set is available
#if 0
	//train classifier on the training set
	Data* d = new Data();
	d->loadData("..\\data\\adult_train_num.csv");
	Timer t;	
	t.tic("Classifier training");
	Classifier* c = new Boosting(15);
	c->setData(d);
	c->train();
	t.toc();
	delete d;
	
	//test the classifier on the test set
	d->loadData("..\\data\\adult_test_num.csv");
	t.tic("Classifier testing");
	float err = c->calcError();
	printf("test set error = %f\n", err);
	t.toc();
#else
	//if a single set is available

	Data* d = new Data();
	//d->loadData("..\\data\\wdbc.csv");
	d->loadData("..\\data\\pima-indians-diabetes.csv");
	//d->loadData("..\\data\\skin_nonskin.csv");
	Timer t;
	t.tic("Classifier crossvalidation");
	//Classifier* c = new Boosting(100);
	//Classifier* c = new Knn(71);
	Classifier* c = new Bayes();
	c->setData(d);
	const int nrfolds = 10;
	float* errs = new float[nrfolds];
	c->crossvalidation(nrfolds, errs);
	t.toc();
	delete[] errs;
#endif
}