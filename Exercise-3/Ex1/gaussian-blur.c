/* Name: Georgios Krommydas
 * A.M.: 3260
 * Parallel Program for Gaussian Blur with MPI
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#pragma pack(push, 2)          
	typedef struct bmpheader_ 
	{
		char sign;
		int size;
		int notused;
		int data;
		int headwidth;
		int width;
		int height;
		short numofplanes;
		short bitpix;
		int method;
		int arraywidth;
		int horizresol;
		int vertresol;
		int colnum;
		int basecolnum;
	} bmpheader_t;
#pragma pack(pop)

/* This is the image structure, containing all the BMP information 
 * plus the RGB channels.
 */
typedef struct img_
{
	bmpheader_t header;
	int rgb_width;
	unsigned char *imgdata;
	unsigned char *red;
	unsigned char *green;
	unsigned char *blue;
} img_t;

void gaussian_blur_serial(int, img_t *, img_t *);
void gaussian_blur_mpi(int, img_t *, img_t *, int, int);


/* START of BMP utility functions */
static
void bmp_read_img_from_file(char *inputfile, img_t *img) 
{
	FILE *file;
	bmpheader_t *header = &(img->header);

	file = fopen(inputfile, "rb");
	if (file == NULL)
	{
		fprintf(stderr, "File %s not found; exiting.", inputfile);
		exit(1);
	}
	
	fread(header, sizeof(bmpheader_t)+1, 1, file);
	if (header->bitpix != 24)
	{
		fprintf(stderr, "File %s is not in 24-bit format; exiting.", inputfile);
		exit(1);
	}

	img->imgdata = (unsigned char*) calloc(header->arraywidth, sizeof(unsigned char));
	if (img->imgdata == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for image data; exiting.");
		exit(1);
	}
	
	fseek(file, header->data, SEEK_SET);
	fread(img->imgdata, header->arraywidth, 1, file);
	fclose(file);
}

static
void bmp_clone_empty_img(img_t *imgin, img_t *imgout)
{
	imgout->header = imgin->header;
	imgout->imgdata = 
		(unsigned char*) calloc(imgout->header.arraywidth, sizeof(unsigned char));
	if (imgout->imgdata == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for clone image data; exiting.");
		exit(1);
	}
}

static
void bmp_write_data_to_file(char *fname, img_t *img) 
{
	FILE *file;
	bmpheader_t *bmph = &(img->header);

	file = fopen(fname, "wb");
	fwrite(bmph, sizeof(bmpheader_t)+1, 1, file);
	fseek(file, bmph->data, SEEK_SET);
	fwrite(img->imgdata, bmph->arraywidth, 1, file);
	fclose(file);
}

static
void bmp_rgb_from_data(img_t *img)
{
	bmpheader_t *bmph = &(img->header);

	int i, j, pos = 0;
	int width = bmph->width, height = bmph->height;
	int rgb_width = img->rgb_width;

	for (i = 0; i < height; i++) 
		for (j = 0; j < width * 3; j += 3, pos++)
		{
			img->red[pos]   = img->imgdata[i * rgb_width + j];
			img->green[pos] = img->imgdata[i * rgb_width + j + 1];
			img->blue[pos]  = img->imgdata[i * rgb_width + j + 2];  
		}
}

static
void bmp_data_from_rgb(img_t *img)
{
	bmpheader_t *bmph = &(img->header);
	int i, j, pos = 0;
	int width = bmph->width, height = bmph->height;
	int rgb_width = img->rgb_width;

	for (i = 0; i < height; i++ ) 
		for (j = 0; j < width* 3 ; j += 3 , pos++) 
		{
			img->imgdata[i * rgb_width  + j]     = img->red[pos];
			img->imgdata[i * rgb_width  + j + 1] = img->green[pos];
			img->imgdata[i * rgb_width  + j + 2] = img->blue[pos];
		}
}

static
void bmp_rgb_alloc(img_t *img)
{
	int width, height;

	width = img->header.width;
	height = img->header.height;

	img->red = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->red == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the red channel; exiting.");
		exit(1);
	}

	img->green = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->green == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the green channel; exiting.");
		exit(1);
	}

	img->blue = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->blue == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the blue channel; exiting.");
		exit(1);
	}

	img->rgb_width = width * 3;
	if ((width * 3  % 4) != 0) {
	   img->rgb_width += (4 - (width * 3 % 4));  
	}
}

static
void bmp_img_free(img_t *img)
{
	free(img->red);
	free(img->green);
	free(img->blue);
	free(img->imgdata);
}

/* END of BMP utility functions */

/* check bounds */
int clamp(int i , int min , int max)
{
	if (i < min) return min;
	else if (i > max) return max;
	return i;  
}

/* Sequential Gaussian Blur */
void gaussian_blur_serial(int radius, img_t *imgin, img_t *imgout)
{
	int i, j;
	int width = imgin->header.width, height = imgin->header.height;
	double row, col;
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width ; j++) 
		{
			for (row = i-radius; row <= i + radius; row++)
			{
				for (col = j-radius; col <= j + radius; col++) 
				{
					int x = clamp(col, 0, width-1);
					int y = clamp(row, 0, height-1);
					int tempPos = y * width + x;
					double square = (col-j)*(col-j)+(row-i)*(row-i);
					double sigma = radius*radius;
					double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

					redSum += imgin->red[tempPos] * weight;
					greenSum += imgin->green[tempPos] * weight;
					blueSum += imgin->blue[tempPos] * weight;
					weightSum += weight;
				}    
			}
			imgout->red[i*width+j] = round(redSum/weightSum);
			imgout->green[i*width+j] = round(greenSum/weightSum);
			imgout->blue[i*width+j] = round(blueSum/weightSum);

			redSum = 0;
			greenSum = 0;
			blueSum = 0;
			weightSum = 0;
		}
	}
}


/* Parallel Gaussian Blur with MPI parallelization */
void gaussian_blur_mpi(int radius, img_t *imgin, img_t *imgout, int ID, int nproc)
{
	/* TODO: Implement parallel Gaussian Blur using MPI parallelization */
	int i, j, WORK;
	int width = imgin->header.width, height = imgin->header.height;
	double row, col;
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;
	unsigned char *red_buffer, *green_buffer, *blue_buffer;
	MPI_Status status;
	
	WORK = height/nproc; // Rows per Process
	red_buffer = (unsigned char *) calloc(WORK*height, sizeof(unsigned char)); // Red channel for every other process
	green_buffer = (unsigned char *) calloc(WORK*height, sizeof(unsigned char)); // Green channel for every other process
	blue_buffer = (unsigned char *) calloc(WORK*height, sizeof(unsigned char)); // Blue channel for every other process
	
	MPI_Bcast(&radius, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(ID == 0){
		for(i = 1; i < nproc; i++){
			MPI_Send(&imgin->red[i*WORK], WORK*height, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
			MPI_Send(&imgin->green[i*WORK], WORK*height, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
			MPI_Send(&imgin->blue[i*WORK], WORK*height, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
		}
		for (i = 0; i < WORK; i++)
		{
			for (j = 0; j < width ; j++)
			{
				for (row = i-radius; row <= i + radius; row++)
				{
					for (col = j-radius; col <= j + radius; col++) 
					{
						int x = clamp(col, 0, width-1);
						int y = clamp(row, 0, WORK-1);
						int tempPos = y * width + x;
						double square = (col-j)*(col-j)+(row-i)*(row-i);
						double sigma = radius*radius;
						double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

						redSum += imgin->red[tempPos] * weight;
						greenSum += imgin->green[tempPos] * weight;
						blueSum += imgin->blue[tempPos] * weight;
						weightSum += weight;
					}    
				}
				imgout->red[i*width+j] = round(redSum/weightSum);
				imgout->green[i*width+j] = round(greenSum/weightSum);
				imgout->blue[i*width+j] = round(blueSum/weightSum);

				redSum = 0;
				greenSum = 0;
				blueSum = 0;
				weightSum = 0;
			}
		}
		for(i = WORK; i < WORK*nproc; i++){
			MPI_Recv(imgout->red, WORK*height, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(imgout->green, WORK*height, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(imgout->blue, WORK*height, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}else{
		
		MPI_Recv(red_buffer, WORK*height, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(green_buffer, WORK*height, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(blue_buffer, WORK*height, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		
		for (i = 0; i < WORK; i++)
		{
			for (j = 0; j < width ; j++)
			{
				for (row = i-radius; row <= i + radius; row++)
				{
					for (col = j-radius; col <= j + radius; col++) 
					{
						int x = clamp(col, 0, width-1);
						int y = clamp(row, 0, WORK-1);
						int tempPos = y * width + x;
						double square = (col-j)*(col-j)+(row-i)*(row-i);
						double sigma = radius*radius;
						double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

						redSum += red_buffer[tempPos] * weight;
						greenSum += green_buffer[tempPos] * weight;
						blueSum += blue_buffer[tempPos] * weight;
						weightSum += weight;
					}    
				}
				imgout->red[i*width+j] = round(redSum/weightSum);
				imgout->green[i*width+j] = round(greenSum/weightSum);
				imgout->blue[i*width+j] = round(blueSum/weightSum);
				
				redSum = 0;
				greenSum = 0;
				blueSum = 0;
				weightSum = 0;
			}
		}
		for(i = 0; i < WORK; i++){
			MPI_Send(imgout->red, WORK*height, MPI_UNSIGNED_CHAR, 0, ID*WORK+i, MPI_COMM_WORLD);
			MPI_Send(imgout->green, WORK*height, MPI_UNSIGNED_CHAR, 0, ID*WORK+i, MPI_COMM_WORLD);
			MPI_Send(imgout->blue, WORK*height, MPI_UNSIGNED_CHAR, 0, ID*WORK+i, MPI_COMM_WORLD);
		}
	}
	
	free(red_buffer);
	free(green_buffer);
	free(blue_buffer);
}

double timeit(void (*func)(), int radius, 
    img_t *imgin, img_t *imgout)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	func(radius, imgin, imgout);
	gettimeofday(&end, NULL);
	return (double) (end.tv_usec - start.tv_usec) / 1000000 
		+ (double) (end.tv_sec - start.tv_sec);
}


char *remove_ext(char *str, char extsep, char pathsep) 
{
	char *newstr, *ext, *lpath;

	if (str == NULL) return NULL;
	if ((newstr = malloc(strlen(str) + 1)) == NULL) return NULL;

	strcpy(newstr, str);
	ext = strrchr(newstr, extsep);
	lpath = (pathsep == 0) ? NULL : strrchr(newstr, pathsep);
	if (ext != NULL) 
	{
		if (lpath != NULL) 
		{
			if (lpath < ext) 
				*ext = '\0';
		} 
		else 
			*ext = '\0';
	}
	return newstr;
}


int main(int argc, char *argv[]) 
{
	int i, j, radius, myid, nproc;
	double exectime_serial = 0.0, exectime_mpi = 0.0, t1, t2, total, start_com, end_com, overheads;
	double w1, w2, w3, w4;
	struct timeval start, stop; 
	char *inputfile, *noextfname;   
	char seqoutfile[128], paroutfile_mpi[128];
	img_t imgin, imgout, pimgout_mpi;
	MPI_Status status;

	start_com = MPI_Wtime();
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	
	if(myid == 0){
		
		w1 = MPI_Wtime();
		if (argc < 3)
		{
			fprintf(stderr, "Syntax: mpirun -np X -hostfile nodes_Y %s <blur-radius> <filename>, \n\te.g. mpirun -np 16 -hostfile nodes_4 %s 2 500.bmp\n", 
				argv[0], argv[0]);
			fprintf(stderr, "Available images: 500.bmp, 1000.bmp, 1500.bmp\n");
			exit(1);
		}
		
		inputfile = argv[2];
		radius = atoi(argv[1]);

		if (radius < 0)
		{
			fprintf(stderr, "Radius should be an integer >= 0; exiting.");
			exit(1);
		}
		
		noextfname = remove_ext(inputfile, '.', '/');
		sprintf(seqoutfile, "%s-r%d-serial.bmp", noextfname, radius);
		sprintf(paroutfile_mpi, "%s-r%d-MPI.bmp", noextfname, radius);

		bmp_read_img_from_file(inputfile, &imgin);
		bmp_clone_empty_img(&imgin, &imgout);
		bmp_clone_empty_img(&imgin, &pimgout_mpi);
		bmp_rgb_alloc(&imgin);
		bmp_rgb_alloc(&imgout);
		bmp_rgb_alloc(&pimgout_mpi);

		printf("<<< Gaussian Blur (h=%d,w=%d,r=%d) >>>\n", imgin.header.height, 
	       imgin.header.width, radius);
		
		/* Image data to R,G,B */
		bmp_rgb_from_data(&imgin);
		
		/* Run & time serial Gaussian Blur */
		exectime_serial = timeit(gaussian_blur_serial, radius, &imgin, &imgout);
		
		/* Save the results (serial) */
		bmp_data_from_rgb(&imgout);
		bmp_write_data_to_file(seqoutfile, &imgout);
		w2 = MPI_Wtime();
	}
	
	
	/* Run & time MPI Gaussian Blur */
	t1 = MPI_Wtime();
	gaussian_blur_mpi(radius, &imgin, &pimgout_mpi, myid, nproc);
	t2 = MPI_Wtime();
	
	
	if(myid == 0){
		/* Save the results (parallel w/ mpi) */
		w3 = MPI_Wtime();
		bmp_data_from_rgb(&pimgout_mpi);
		bmp_write_data_to_file(paroutfile_mpi, &pimgout_mpi);
		bmp_img_free(&imgin);
		bmp_img_free(&imgout);
		bmp_img_free(&pimgout_mpi);
		w4 = MPI_Wtime();
		end_com = MPI_Wtime();
		exectime_mpi = t2 - t1;
		total = (end_com - start_com) - (w2 - w1) - (w4 - w3);
		overheads = total - exectime_mpi;
		printf("Total execution time (sequential): %lf\n", exectime_serial);
		printf("Total execution time (MPI): %lf\nOverheads time: %lf\nComputation time: %lf", total, overheads, exectime_mpi);	
	}	
	MPI_Finalize();

	return 0;
}
