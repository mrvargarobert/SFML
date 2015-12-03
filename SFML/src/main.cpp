#include "data.h"


int main(){
	
	FILE * f = fopen("test_file.csv", "w");
	for (int i = 0; i < 100; i++) {
		fprintf(f, "\n");
		for (int j = 0; j < 361; j++)
			fprintf(f, "%d ", i);
	}
	 fclose(f);
	 Data d = Data();
	 d.loadData("test_file.csv");
	 return 0;
}