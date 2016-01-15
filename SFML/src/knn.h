#ifndef KNN_H
#define KNN_H
#include "dtree.h"
#include "classifier.h"


/**
Structure containing the model used for classification. 
*/
struct Knnmodel{
	T* dist; /**< Array containing distances from the sample point to every labeled point*/
	int k;   /**< Number of neighbours */
	Data* traindata;
};

/**
Class implementing the Knn algorithm.
*/
class Knn : public Classifier{
public:
	Knn();
	Knn(int k);
	~Knn();

	//model io
	void loadModel(const char* fname) override; 
	void saveModel(const char* fname) override;

	//training
	T train() override;	
	T train(const char* fname);

	//predicting/classifying
	T predict(T* sample) override;

private:
	Knnmodel *model;
};
#endif