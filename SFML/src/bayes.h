#ifndef BAYES_H
#define BAYES_H
#include "dtree.h"
#include "classifier.h"


/**
Structure containing the model used for classification. 
*/
struct Bayesmodel{
	Data* traindata;
};

/**
Class implementing the Bayes classification algorithm.
*/
class Bayes : public Classifier{
public:
	Bayes();
	~Bayes();

	//model io
	void loadModel(const char* fname) override; 
	void saveModel(const char* fname) override;

	//training
	T train() override;	
	T train(const char* fname);

	//predicting/classifying
	T predict(T* sample) override;

private:
	Bayesmodel *model;
};
#endif