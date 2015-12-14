#include "stdh.h"
#include "util.h"
#include "dtree.h"
#include "boosting.h"

Boosting::Boosting(){
	//prepare structures
	model = new Boostmodel;	
	model->bp = new Boostparams;
	model->bp->nrweak = -1;
	model->bp->maxDepth = MAX_DTREE_DEPTH;
	model->bp->type = 0;
	model->bp->pruning = 0;
	model->thetas = 0;
	model->c = 0;
	model->trees = 0;
}

Boosting::Boosting(int nrweak){
	//prepare structures
	model = new Boostmodel;		
	model->bp = new Boostparams;
	model->bp->nrweak = nrweak;
	model->bp->maxDepth = MAX_DTREE_DEPTH;
	model->bp->type = 0;
	model->bp->pruning = 0;
	model->thetas = 0;
	model->c = new T2[nrweak];
	model->trees = new DTree[nrweak];	
}

Boosting::~Boosting(){
	delete[] model->c;
	delete[] model->trees;		
	delete[] model->thetas;	
	delete model->bp;
	delete model;
}

void Boosting::saveModel(const char* fname){
	FILE* f = fopen(fname, "w");
	if (!model || !model->bp)
	{
		printf("Model not found\n");
		getch();
		exit(-3);
	}

	fprintf(f, "boosted-classifier-model-file\n");
	fprintf(f, "type:%d\n", model->bp->type);
	fprintf(f, "nrweak:%d\n", model->bp->nrweak);
	fprintf(f, "pruning:%f\n", model->bp->pruning);
	fprintf(f, "weakdepth:%d\n", model->bp->maxDepth);

	for (int i = 0; i<model->bp->nrweak; i++)
	{
		fprintf(f, "w:%f, rej:%f\n", model->c[i], model->thetas[i]);
		model->trees[i].save(f);
	}
	fclose(f);
}

void Boosting::loadModel(const char* fname){
	FILE* f = fopen(fname, "r");
	if (!f) {
		printf("Classifier file not found: %s\n", fname);
		getch();  exit(-1);
	}
	
	char str[100];
	fscanf(f, "%s\n", str);
	fscanf(f, "type:%d\n", &model->bp->type);
	fscanf(f, "nrweak:%d\n", &model->bp->nrweak);
	fscanf(f, "pruning:%f\n", &model->bp->pruning);
	fscanf(f, "weakdepth:%d\n", &model->bp->maxDepth);

	if (model->thetas) delete[] model->thetas;
	if (model->c) delete[] model->c;
	if (model->trees) delete[] model->trees;

	model->thetas = new T[model->bp->nrweak];
	model->c = new T2[model->bp->nrweak];
	model->trees = new DTree[model->bp->nrweak];
	for (int i = 0; i<model->bp->nrweak; i++)
	{
		if (sizeof(T2)==8)
			fscanf(f, "w:%lf, rej:%f\n", &model->c[i], &model->thetas[i]);
		else
			fscanf(f, "w:%f, rej:%f\n", &model->c[i], &model->thetas[i]);
		model->trees[i].load(f);
		model->trees[i].setLabel(model->c[i]);
	}
	fclose(f);
}

void Boosting::presortData(T3*& oi){
	printf("Presorting input data\n");
	//sort columns into ordered indexes	
	oi = new T3[data->M*data->N];
#pragma omp parallel for
	for (int j = 0; j<data->M; j++)
	{
		vector<Pair> pairs;
		/*
		pairs.reserve(data->N);
		for(int i=0, iM=0; i<data->N; i++, iM+=data->M)
		pairs.push_back(Pair( data->d[iM+j], (T3)i) );
		*/
		pairs.resize(data->N);
		for (long long i = 0, iM = 0; i<data->N; i++, iM += data->M)
			pairs[i] = Pair(data->d[iM + j], (T3)i);
		sort(pairs.begin(), pairs.end(), pairs[0]);
		T3* oijn = oi + j*data->N;
		for (int i = 0; i<data->N; i++)
			oijn[i] = pairs[i].i;
	}
	printf("Done sorting (%d x %d entries)\n", data->N, data->M);
}

T Boosting::train(T* descr, T*labels, int N, int M){
	data->N = N;
	data->M = M;
	data->w = new T2[N];
#define COPY_DATA 0
#if COPY_DATA //copy data
	data->d = new T[N*M];
	data->l = new T[N];
	memcpy(data->d, descr, N*M*sizeof(T));
	memcpy(data->l, labels, N*sizeof(T));
#else
  data->l = labels;
	data->d = descr;
#endif
	for(int i=0; i<N; i++)
    data->w[i] = ((T2)1.0)/N;
	
	float ret = train();
#if !COPY_DATA //this way the destructor does not deallocate the memory locations ad d and l
	data->d = 0;
	data->l = 0;
#endif
	return ret;
}

T Boosting :: train(const char* fname){
	data = new Data();
	data->loadData(fname);
	return train();
}

T Boosting::train(){
	int nrweak = model->bp->nrweak;
	const long long N = data->N;
	const long long M = data->M;
	T2* lbls = new T2[N];
	T2* scores = new T2[N]; memset(scores, 0, N*sizeof(T2));
	T2* c = model->c;
	DTree* trees = model->trees;

	T3* oi = 0;
	presortData(oi);

	breaking_pts.clear();
	double errth = 0.01;
		
	//Timer t; t.tic("time elapsed");
	int m=0;
	for(; m<nrweak; m++)
	{
		//printf("Training tree %d\n",m);
		trees[m].p.maxDepth = model->bp->maxDepth;
		//t.tic();
		trees[m].train( data, oi );		
		//t.toc();		

		//find the weighted training error for the m-th weak classifier		
#pragma omp parallel for //reduction(+:errm)
		for(long long i=0; i<N; i++)
		{
			lbls[i] = trees[m].predict( data->d+i*M ); //? maybe try to eliminate			
		}
		
		T2 errm = 0;
#if 1
		for(int i=0; i<N; i++) 
			if (lbls[i] * data->l[i]<0) 
				errm += data->w[i];		
#else
		T2 err1 = 0, err2 = 0, sump = 0, sumn = 0;
		for (int i = 0; i<N; i++)
		if (data->l[i]>0)
		{
			if (lbls[i]>0)
				err1 += data->w[i];
			sump += data->w[i];
		}				
		else
		{
			if (lbls[i]<0)
				err2 += data->w[i];
			sumn += data->w[i];
		}
		errm = 0.5*(err1 / sump + err2 / sumn);
#endif
					
		if (errm > 0.5 )
			break;		
				
		T2 sumw = 0;
		c[m] = 0.5*log( ( 1-errm) / (errm+EPS) );

		T2 err = 0;
		for (int i = 0; i < N; i++)
		{
			scores[i] += c[m] * lbls[i];
			if (scores[i] * data->l[i] < 0)
				err++;
		}
		err /= data->N;
		printf("Error Using %d trees: %5.4f %%, acc %5.4f %%\n", m, err*100.f, (1 - err)*100.f);
		if (err < errth)
		{
			breaking_pts.push_back(m);
			errth *= 0.1;
		}

		for(int i=0; i<N; i++)
		{
#if 0
			//binary weight change
			if (lbls[i] * data->l[i] > 0)
				data->w[i] = 1;
			else
				data->w[i] = 0;
#else
			//normal adaboost weight change
			data->w[i] *= exp( -c[m] * lbls[i] * data->l[i] );
#endif
			sumw += data->w[i];
		}
		trees[m].setLabel(c[m]);		

		for(int i=0; i<data->N; i++)
			data->w[i] /= sumw;
	}
	model->bp->nrweak = m;
	nrweak = m;

	//show the training error using all the weak classifiers
	for (int m2=nrweak; m2<=nrweak; m2++)
	{
		model->bp->nrweak = m2;
		T2 err = 0;
		for(long long i=0; i<N; i++)
		{
			T lbl = predict( data->d+i*M );
			if ( lbl * data->l[i] < 0 )
				err++;
		}
		err /= data->N;
		printf("\nUsing all %d trees: err %5.4f %%, acc %5.4f %%\n", m2, err*100.f, (1-err)*100.f);
	}

	//find the rejection thresholds using direct backward pruning (Viola Jones)
	model->thetas = new T[nrweak];
	T* sampletrace  = new T[N];
	memset(sampletrace, 0, N*sizeof(T));

	for(int j=0; j<model->bp->nrweak; j++){
		T minsampletrace = INF;
		for(long long i=0; i<N; i++)
			if ( data->l[i] == 1.0 )
			{				
				T pred = model->trees[j].predict( data->d+i*M );
				sampletrace[i] += pred;			
				if ( sampletrace[i] < minsampletrace )
					minsampletrace = sampletrace[i];
			}
		model->thetas[j] = minsampletrace - EPS;
	}
	
	//show the training error using all the weak classifiers and the rejection thrs
	double start = clock();
	T err2 = 0;
	T avg = 0;
	for(long long i=0; i<data->N; i++)
	{
		T lbl = 0;
		int j;
		for(j=0; j<nrweak; j++)
		{
			lbl += model->trees[j].predict( data->d+i*M );
				if (lbl<model->thetas[j])
				{
					lbl = -1;
					j++;
					break;
				}
		}
		avg += j;
		if ( lbl * data->l[i] < 0 )
			err2++;
	}
	double end = clock();
	avg /= N;
	err2 /= data->N;
	printf("Training data results: err2 %5.4f %%, acc %5.4f %%\n", err2*100.f, (1-err2)*100.f);	
	
	printf("\nMade %d classifications in %f seconds, speed: %f ms/instance\n", N, (end-start)/CLOCKS_PER_SEC, (end-start)*1000/CLOCKS_PER_SEC/N );
	printf("Average number of feature evaluations %.2f out of %d\n", avg, model->bp->nrweak);

#if 0
	//save breaking points, the number of weak learners required to achieve error = 1e-k
	printf(f, "\n");
	double errth = 0.01;
	for (int i = 0; i<5; i++, errth *= 0.1)
		printf(f, "%d  %g\n", breaking_pts[i], errth);
#endif
	
	delete[] oi;
	delete[] lbls;	
	delete[] scores;
	delete[] sampletrace;
	printf("Done!\n");
	return err2;
}

T Boosting::predict(T* sample){
	T pred = 0;
	for(int i=0; i<model->bp->nrweak; i++)
		pred += model->trees[i].predict(sample); //DDD rewrite with 
	return pred;
}

T Boosting::predictCascade(T* sample, float cascade_th){ //predict using all the weak classifiers 
	T pred = cascade_th;
	for(int i=0; i<model->bp->nrweak; i++)
	{
		pred += model->trees[i].predict(sample);
		if (pred<model->thetas[i])
		{
				pred = -1000; //assign a negative score as prediction
				break;
		}
	}
	return pred - cascade_th;
}

T Boosting::predictCascadeFixed(T* sample, float cascade_th){ //predict using all the weak classifiers 
	T pred = 0;
	for (int i = 0; i<model->bp->nrweak; i++)
	{
		pred += model->trees[i].predict(sample);
		if (pred<cascade_th)
		{
			pred = -1000; //assign a negative score as prediction
			break;
		}
	}
	return pred;
}

void Boosting::recalculateThresholds(float Q){
	//recalculates the cascade thesholds so that thetas_new[end] = thetas[end]+Q and thetas_new[0]=thetas[0]
	float t0 = model->thetas[0];
	float tend = model->thetas[model->bp->nrweak-1]+Q;
	float incr = 1.f/(model->bp->nrweak-1);
	int i;
	float t;
	for(t=0.f, i=0; i<model->bp->nrweak; i++, t+=incr)
		model->thetas[i] = (1-t)*t0+t*tend;
}

void Boosting::test(int test_nr){
	//test_nr = 8;
	char fname[256];
	sprintf(fname, "D:\\matlab-proj\\palletdetection\\boosting\\train_%03d.txt", test_nr);
	Data d;
	d.loadData(fname);
	//d.saveData("d:\\matlab-proj\\palletdetection\\boosting\\train_dump.txt");
	setData(&d);
	train();
	saveModel("D:\\matlab-proj\\palletdetection\\boosting\\model.txt");


	sprintf(fname, "D:\\matlab-proj\\palletdetection\\boosting\\test_%03d.txt", test_nr);
	d.loadData(fname);
	//d.saveData("d:\\matlab-proj\\palletdetection\\boosting\\test_dump.txt");
	setData(&d);
	double err = 0;
	for (long long i = 0; i<data->N; i++)
	{
		T lbl = 0;
		int j;
		for(j=0; j<model->bp->nrweak; j++)
		{
			lbl += model->trees[j].predict( data->d+i*data->M );
			if (lbl<model->thetas[j])
			{
				lbl = -1;
				j++;
				break;
			}
		}
		if ( lbl * data->l[i] < 0 )
			err++;
	}
	double end = clock();
	err /= data->N;
	printf("Test set results: err2 %5.4f %%, acc %5.4f %%\n", err*100.f, (1 - err)*100.f);
	getch();
}

#if 0 //probability estimate (real adaboost)
T pm = 0;
T sumw = 0;
for (int i = 0; i<N; i++)
{
	pm = trees[m].predictp(data[i]);
	if (pm == 0)
		pm += EPS;
	if (pm >= 1)
		pm = 1 - EPS;
	fm = 0.5* log(pm[i]) / log(1 - pm[i]);
	data[i][wi] *= exp(-data[i][data[0].size() - 1]] * fm[i]);
	sumw += data[i][wi];
}
#endif