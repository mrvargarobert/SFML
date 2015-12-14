#include "data.h"
#include "util.h"

const int RED = 1;
const int MAXNRSAMPLES = (int)1e6; //1e5

Data::Data(){
	d = 0; l = 0; w = 0;
}

Data::~Data(){
	if (d) delete[] d;
	if (l) delete[] l;
	if (w) delete[] w;
}

void Data::loadData(const char* fname){
	//read the data from a csv file where the first column is the binary class label 1/-1
	FILE* f = fopen2(fname, "r");	
	printf("\nLoading data from file\n");

	//find nr of features by reading the first instance
	long long nrfeatures = 0;
	int lbl; T tmp;
	fscanf(f, "%d", &lbl);
	while (fscanf(f, ",%f", &tmp) == 1) 	nrfeatures++;
	
	//find the file size
	fseek(f, 0, SEEK_END);
	long long file_size = ftell(f);
	//file_size = 7.5 * 1024 * 1024 * 1024;
	rewind(f);

	//allocate memory and set pointer
	w = new T2[MAXNRSAMPLES];
	l = new T[MAXNRSAMPLES];
	d = new T[MAXNRSAMPLES*nrfeatures];
	T* descr = d;

	ull nrsamples = 0;

	const T2 POSW = 1.0;
	//read the training data while there are training samples
	printf("\n");
	while (nrsamples<MAXNRSAMPLES)
	{
		
		if (fscanf(f, "%d", &lbl) < 1 || lbl == 0)
			break;

		if (lbl == 1){
			w[nrsamples] = POSW;
			l[nrsamples] = 1;
		}
		else{
			w[nrsamples] = 1;
			l[nrsamples] = -1;
		}

		for (int j = 0; j<nrfeatures; j++)
			fscanf(f, ",%f", descr++);
		nrsamples++;
		if (nrsamples % 100 == 0){
#if 1
			printf("\rsample index: %9d", nrsamples);			
#else
			//does not work for large files
			long long progress = ftell(f);
			if (file_size < progress)
				file_size *= 1024L;
			printf("\r %.2lf %%", progress*100.0 / file_size);
#endif
		}
	}
	fclose(f);
	printf("\nDone.\n"); if (nrsamples == MAXNRSAMPLES) printf("Stopping prematurely (%d)\n", nrsamples);

	if ((nrsamples > 65535 && sizeof(T3) == 2) || (nrsamples > 4294967295 && sizeof(T3) == 4)){
		printf("Error, nrsamples=%d is larger than T3 type can handle (65535 / 4294967295)\n", nrsamples);
		getch();
		exit(-2);
	}
	printf("nr samples %d\n", nrsamples);

	//set fields and normalize weights
	N = nrsamples;
	M = nrfeatures;
	T2 iN = 1.0 / N;
	for (int i = 0; i<N; i++)
		w[i] *= iN;

	//check format
	if (nrfeatures != (descr - d) / nrsamples){
		printf("Incorrect format\n");
		return;
	}
}

void Data::loadSparseData(const char* fname, int nrfeatures){
	return; //not implemented 
}

void Data::loadDataBinary(const char* fname, int dim){
	int nrsamples = 0;
	int tmplbl;
	ifstream in(fname, ios::in | ios::binary);
	if (!in)
	{
		printf("File not found %s\n", fname);
		return;
	}

	T2 sumw = 0;

	w = new T2[MAXNRSAMPLES];
	l = new T[MAXNRSAMPLES];
	d = new T[MAXNRSAMPLES*(long long)dim];
	T* descr = d;
	while (nrsamples<MAXNRSAMPLES){
		//if (in.eof())	break;		
		in.read((char*)&tmplbl, 4);
		if (!in) 
			break;

		int d = 0;
#if 0 //buffering not, very useful
		const int chunk = 2000;
		while (d + chunk < dim){
			in.read((char*)(descr + d), sizeof(T)*chunk);//RRR change back
			d += chunk;
		}
#endif
		in.read((char*)(descr + d), sizeof(T)*(dim - d));

		if (nrsamples % RED == 0){
			if (tmplbl == 1)
				l[nrsamples] = 1;
			else
				l[nrsamples] = -1;
			w[nrsamples] = 1;
			sumw += w[nrsamples];
			descr += dim;
		}
		nrsamples++;
	}
	in.close();
	printf("Read %d x %d samples\n", nrsamples, dim);

	N = nrsamples;
	M = dim;
	for (int i = 0; i<nrsamples; i++)
		w[i] /= nrsamples;

#if 0
	FILE* ff = fopen("d:\\imgdb\\stereo_pairs\\zzzz_zz_zz_classifier\\debug_bin.txt", "w");
	for (long long u = 0; u < N*M; u++)
		fprintf(ff, "%f\n", d[u]);
	fclose(ff);
#endif
}

void Data::saveData(const char* fname){
	FILE* f = fopen(fname, "w");
	for (long long i = 0; i < N; i++){
		fprintf(f, "%f", l[i]);
		for (long long j = 0; j < M; j++)
			fprintf(f, ",%f", d[i*M+j]);
		fprintf(f, "\n");
	}
	fclose(f);
}

void Data::normalizeData(Data* data){
	const int nrsamples = data->N;
	const int nrfeatures = data->M;

	T* means = new T[nrfeatures];
	T* stds = new T[nrfeatures];
	memset(means, 0, nrfeatures*sizeof(T));
	memset(stds, 0, nrfeatures*sizeof(T));

	//calculate mean
	for (int i = 0, i2=0; i<nrsamples; i++, i2+=nrfeatures)
	{
		for (int j = 0; j<nrfeatures; j++)
		{
			means[j] += data->d[i2 + j];
		}
	}

	T inrsamples = 1.f / nrsamples;
	for (int j = 0; j<nrfeatures; j++)
		means[j] *= inrsamples;

	//calculate standard deviation
	for (int i = 0, i2 = 0; i<nrsamples; i++, i2+=nrfeatures)
	{
		for (int j = 0; j<nrfeatures; j++)
		{
			stds[j] += sqr2(data->d[i2 + j] - means[j]);
		}
	}

	//normalize
	for (int i = 0, i2 = 0; i<nrsamples; i++, i2+=nrfeatures)
	{
		for (int j = 0; j<nrfeatures; j++)
		{
			data->d[i2 + j] = (data->d[i2 + j] - means[j]) / sqrt(stds[j] + EPS);
		}
	}

	delete[] means;
	delete[] stds;
}