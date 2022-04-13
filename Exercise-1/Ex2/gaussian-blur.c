#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#pragma pack(push, 2)          
    typedef struct bmp_img_ 
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
    } bmp_img_t;
#pragma pop

typedef struct bmpdata_
{
    bmp_img_t *bmp;
    int radius;
    int height;
    int width;
    int rgb_width;
    unsigned char *imgdata;
    unsigned char *imgdata_bak;
    unsigned char *red;
    unsigned char *green;
    unsigned char *blue;
} bmpdata_t;

static
double timeit(void (*func)(bmpdata_t *), bmpdata_t *bmpdata)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    func(bmpdata);
    gettimeofday(&end, NULL);
    return (double) (end.tv_usec - start.tv_usec) / 1000000 
        + (double) (end.tv_sec - start.tv_sec);
}

static
char* read_imgdata_fromfile(char *inputfile, bmpdata_t *bmpdata) {
    char* data;

    FILE* file;
    if (!(file = fopen(inputfile, "rb"))) 
    {
        printf("File not found; exiting.");
        free(bmpdata->bmp);
        exit(1);
    }
    fread(bmpdata->bmp, 54, 1, file);
    if (bmpdata->bmp->bitpix != 24)
    {
        free(bmpdata->bmp);
        printf("File is not in 24-bit format; exiting.");
        exit(1);
    }
    data = (char*) malloc (bmpdata->bmp->arraywidth);
    fseek(file, bmpdata->bmp->data, SEEK_SET);
    fread(data, bmpdata->bmp->arraywidth, 1, file);
    fclose(file);
    return data;
}

static
char *remove_ext(char *str, char extsep, char pathsep) {
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
            {
                *ext = '\0';
            }
        } 
        else 
        {
            *ext = '\0';
        }
    }

    return newstr;
}

static
void write_imgdata_tofile(char *inputfile, char *suffix, bmpdata_t *bmpdata) {
    FILE* file;
    char fileNameBuffer[32];

    sprintf(fileNameBuffer, "%s-r%d-%s.bmp", 
        remove_ext(inputfile, '.', '/'), bmpdata->radius, suffix);

    file = fopen(fileNameBuffer, "wb");
    fwrite(bmpdata->bmp, sizeof(bmp_img_t)+1, 1, file);
    fseek(file, bmpdata->bmp->data, SEEK_SET);
    fwrite(bmpdata->imgdata, bmpdata->bmp->arraywidth, 1, file);
    fclose(file);
}

static
int set_boundary(int i , int min , int max){
    if( i < min) return min;
    else if( i > max ) return max;
    return i;  
}

static
void set_rgb_fromimgdata(bmpdata_t *bmpdata)
{
    int i, j, pos = 0;
    int width = bmpdata->width, height = bmpdata->height;
    int rgb_width = bmpdata->rgb_width;

    for (i = 0; i < height; i++) 
    {
        for (j = 0; j < width * 3; j += 3, pos++)
        {
            bmpdata->red[pos] = bmpdata->imgdata[i * rgb_width + j];
            bmpdata->green[pos] = bmpdata->imgdata[i * rgb_width + j + 1];
            bmpdata->blue[pos] = bmpdata->imgdata[i * rgb_width + j + 2];  
        }
    }
}

static
void set_imgdata_fromrgb(bmpdata_t *bmpdata)
{
    int i, j, pos = 0;
    int width = bmpdata->width, height = bmpdata->height;
    int rgb_width = bmpdata->rgb_width;

    for (i = 0; i < height; i++ ) 
    {
        for (j = 0; j < width* 3 ; j += 3 , pos++) 
        {
            bmpdata->imgdata[i * rgb_width  + j] = bmpdata->red[pos];
            bmpdata->imgdata[i * rgb_width  + j + 1] = bmpdata->green[pos];
            bmpdata->imgdata[i * rgb_width  + j + 2] = bmpdata->blue[pos];
        }
    }
}

/* Serial Gaussian Blur */
void gaussian_blur_serial(bmpdata_t *bmpdata)
{
    int i, j;
    int width = bmpdata->width, height = bmpdata->height;
    int radius = bmpdata->radius;
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
                    int x = set_boundary(col, 0, width-1);
                    int y = set_boundary(row, 0, height-1);
                    int tempPos = y * width + x;
                    double square = (col-j)*(col-j)+(row-i)*(row-i);
                    double sigma = radius*radius;
                    double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

                    redSum += bmpdata->red[tempPos] * weight;
                    greenSum += bmpdata->green[tempPos] * weight;
                    blueSum += bmpdata->blue[tempPos] * weight;
                    weightSum += weight;
                }    
            }
            bmpdata->red[i*width+j] = round(redSum/weightSum);
            bmpdata->green[i*width+j] = round(greenSum/weightSum);
            bmpdata->blue[i*width+j] = round(blueSum/weightSum);

            redSum = 0;
            greenSum = 0;
            blueSum = 0;
            weightSum = 0;
        }
    }
}

/* Parallel Gaussian Blur */
void gaussian_blur_omp_loops(bmpdata_t *bmpdata)
{
    /* TODO: Implement parallel Gaussian Blur using OpenMP */
}

/* Parallel Gaussian Blur with tasks*/
void gaussian_blur_omp_tasks(bmpdata_t *bmpdata)
{
    /* TODO: Implement parallel Gaussian Blur using OpenMP with tasks*/
}

static
void backup_imgdata(bmpdata_t *bmpdata)
{
    memcpy(bmpdata->imgdata_bak, bmpdata->imgdata, bmpdata->bmp->arraywidth);
}

static
void restore_imgdata(bmpdata_t *bmpdata)
{
    free(bmpdata->imgdata);
    bmpdata->imgdata = bmpdata->imgdata_bak;
}

int main(int argc, char *argv[]) {
    int i, j;
    double exectime_serial = 0.0, exectime_omp_loops = 0.0, exectime_omp_tasks = 0.0;
    struct timeval start, stop; 
    char *inputfile;   
    int width, height, radius; // copies of actual fields
    bmpdata_t *bmpdata = malloc(sizeof(bmpdata_t));

    if (argc < 3)
    {
        printf("Syntax: %s <blur-radius> <filename>, \n\te.g. %s 2 500.bmp\n", argv[0], argv[0]);
        printf("Available images: 500.bmp, 1000.bmp, 1500.bmp\n");
        exit(1);
    }

    inputfile = argv[2];

    bmpdata->radius = radius = atoi(argv[1]);
    bmpdata->bmp = (bmp_img_t*) malloc(sizeof(bmp_img_t)+1);
    bmpdata->imgdata = read_imgdata_fromfile(inputfile, bmpdata);
    bmpdata->imgdata_bak = (char *) malloc(bmpdata->bmp->arraywidth);
    bmpdata->width = width = bmpdata->bmp->width;
    bmpdata->height = height = bmpdata->bmp->height;
    bmpdata->red = (unsigned char*) malloc(width*height);
    bmpdata->green = (unsigned char*) malloc(width*height);
    bmpdata->blue = (unsigned char*) malloc(width*height);
    bmpdata->rgb_width =  width * 3 ;

    if ((width * 3  % 4) != 0) {
       bmpdata->rgb_width += (4 - (width * 3 % 4));  
    }

    printf("<<< Gaussian Blur (h=%d,w=%d,r=%d) >>>\n", height, width, radius);
    
    /* Backup as they will be modified later */
    backup_imgdata(bmpdata);

    /* Image data to R,G,B */
    set_rgb_fromimgdata(bmpdata);
    exectime_serial = timeit(gaussian_blur_serial, bmpdata);

    /* Flush the results (serial) */
    set_imgdata_fromrgb(bmpdata);
    write_imgdata_tofile(inputfile, "serial", bmpdata);

    restore_imgdata(bmpdata);
    set_rgb_fromimgdata(bmpdata);
    exectime_omp_loops = timeit(gaussian_blur_omp_loops, bmpdata);

    /* Flush the results (parallel_loops) */
    set_imgdata_fromrgb(bmpdata);
    write_imgdata_tofile(inputfile, "parallel_loops", bmpdata);

    restore_imgdata(bmpdata);
    set_rgb_fromimgdata(bmpdata);
    exectime_omp_tasks = timeit(gaussian_blur_omp_tasks, bmpdata);

    /* Flush the results (parallel_tasks) */
    set_imgdata_fromrgb(bmpdata);
    write_imgdata_tofile(inputfile, "parallel_tasks", bmpdata);
        
    printf("Total execution time (serial):              %lf\n", exectime_serial);
    printf("Total execution time (parallel with loops): %lf\n", exectime_omp_loops);
    printf("Total execution time (parallel with tasks): %lf\n", exectime_omp_tasks);

    free(bmpdata->bmp);
    free(bmpdata->red);
    free(bmpdata->green);
    free(bmpdata->blue);
    free(bmpdata->imgdata);

    return 0;
}

