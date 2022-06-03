#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 2048
#define Afile "Amat2048"
#define Bfile "Bmat2048"
#define Cfile "Cmat2048"

int A[N][N], B[N][N], C[N][N];
int readmat(char *fname, int *mat, int n), 
    writemat(char *fname, int *mat, int n);

int main()
{
	int i, j, k, sum;
	struct timeval start, end;
	double execution_time = 0.0;

	/* Read A & B matrices from files
	 */
	if (readmat(Afile, (int *) A, N) < 0) 
		exit( 1 + printf("file problem\n") );
	if (readmat(Bfile, (int *) B, N) < 0) 
		exit( 1 + printf("file problem\n") );
	gettimeofday(&start, NULL);
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			for (k = sum = 0; k < N; k++)
				sum += A[i][k]*B[k][j];
			C[i][j] = sum;
		};

	gettimeofday(&end, NULL);
	/* Save result in file
	 */
	execution_time = (double)(end.tv_sec - start.tv_sec)+ (double)(end.tv_usec - start.tv_usec)*1E-06;
	printf("Total Serial Execution time: %lf\n", execution_time);
	writemat(Cfile, (int *) C, N);

	return (0);
}


/* Utilities to read & write matrices from/to files
 * VVD
 */

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
