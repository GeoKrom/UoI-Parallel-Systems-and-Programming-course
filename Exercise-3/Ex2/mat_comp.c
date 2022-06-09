/* Name: George Krommydas
 * A.M.: 3260
 * Comparing Matrices Code
 */
#include <stdio.h>
#include <stdlib.h>

#define N 2048

int parCMat[N][N], C[N][N];
int readmat(char *fname, int *mat, int n),
    writemat(char *fname,int *mat, int n);

int main(){

    int i, j;

    printf("Comparing matrices...\n");
    
    if(readmat("CmatPar2048", (int *) parCMat, N) < 0){
        exit(1 + printf("File problem in Par!\n"));
    }
    
    if(readmat("Cmat2048", (int *) C, N) < 0){
        exit(1 + printf("File problem in Serial!\n"));
    }

    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            if(parCMat[i][j] != C[i][j]){
                printf("Output Matrix from parallel program is incorrect!\n");
                exit(1);
            }
        }
    }
    printf("Process has been completed...\n");
    printf("All done and is correct!\n");
    return 0;

}

#define _mat(i,j) (mat[(i)*n + (j)])


int readmat(char *fname, int *mat, int n)
{
	FILE *fp;
	int  i, j;
	
	if ((fp = fopen(fname, "r")) == NULL)
		return (-1);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			if (fscanf(fp, "%d", &_mat(i,j)) == EOF)
			{
				fclose(fp);
				return (-1); 
			};
	fclose(fp);
	return (0);
}


int writemat(char *fname, int *mat, int n)
{
	FILE *fp;
	int  i, j;
	
	if ((fp = fopen(fname, "w")) == NULL)
		return (-1);
	for (i = 0; i < n; i++, fprintf(fp, "\n"))
		for (j = 0; j < n; j++)
			fprintf(fp, " %d", _mat(i, j));
	fclose(fp);
	return (0);
}
