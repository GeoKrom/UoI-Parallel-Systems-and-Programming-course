# Parallel-Systems-and-Programming-course

Assignments for course CSE/MYE023 - Parallel Systems and Programming, Department of Computer Science and Engineering, UoI

# Assignment 1
On this assignment was used the [OpenMP API](https://www.openmp.org/) in order to create parallel programs from certain applications.

## First Program

The first program computes all prime numbers up to number N = 10.000.000. The parallel method that was used is the parallel and for constructs above the for loop. 
The schedule policy was also used and in particular the best one was static with chunk size 1.000. For the experiments were used from 1 to 4 threads to observe the recession of execution time.
We find that with the increase of the number threads, the execution time is reduced approximately ideally and in fact by 1/NumThreads.

### How to run
```bash
    gcc -fopenmp primes.c -o primes
    ./primes
```

## Second Program

The second program is about image filtering with the gaussian blur method. On this program there were created two method. The first method was with the for loop construct with static schedule. The for construct was placed before the first loop. 
The second method was with tasks. Every task is a row of the image, thus the task construct will be before the second loopFor the experiments were used from 1 to 4 threads to observe the recession of execution time.
We find that with the increase of the number threads, the execution time is reduced approximately ideally and in fact by 1/NumThreads. 

### How to run
```bash
    gcc -fopenmp gaussian-blur.c -lm -o gb
    ./gb <radius> <input_image.bmp>
```

## Third Program

The third and final program is about matrix multiplication. The method that was used is the taskloop contruct.

### How to run
```bash
    gcc -fopenmp matmul.c -o matmul
    ./matmul
```
or
(A simple dot product computation program with taskloop)
```bash
    gcc -fopenmp dotproduct.c -o dotprod
    ./dotprod
```

# Assignment 2

On this assignment was used a cluster with GPUs to gain information for the system with [CUDA runtime API](https://docs.nvidia.com/cuda/archive/11.6.0/) and use the GPU in order to create parallel a program with [OpenMP API](https://www.openmp.org/).

## First Program
The first program was is about finding information for a CUDA device and print its properties

### How to run
Here is used the NVIDIA compiler.

If the program ends with .cu, then write the command line
```bash
    nvcc cuinfo.cu -o cuinfo
    ./cuinfo
```
If the program ends with .c, then write the command line
```bash
    nvcc -x cuinfo.c -o cuinfo
    ./cuinfo
```

## Second Program
The second program is about image filtering with the Gaussian Method. In order to do the filtering, OpenMP with CUDA was used. 
There were two constraints with the problem. The first constraint was the number of threads, were they should be a factor of 32 for better performance. The second contraint was the number of blocks (or teams) in GPU. These constraints were used in order to compute with all th available cores of gpu for higher performnce and fast results.
In order for the gpu to apply the gaussian blur filter, first OpenMP offloads all data and instructions to the gpu with the line code. The next part is to create the necessary blocks for the streaming multiprocessors to execute. In addition, the distribute clause evenly allocates the data to the master threads of a block. Last but not least the parallel for clause adds the information from master thread to all the other threads of the block to execute in parallel all the process. 
```C
    #pragma omp target teams distribute parallel for collapse(2)\
        map(to: imgin->red[0:height*width], imgin->green[0:height*width], imgin->blue[0:height*width])\
        map(tofrom: imgout->red[0:height*width], imgout->green[0:height*width], imgout->blue[0:height*width])
```
### How to run
Here is used the LLVM/Clang compiler
```bash
    clang -fopenmp -fopenmp-targets="nvptx64-nvidia-cuda" -lm gaussian-blur.c -o gb
    ./gb <radius> <input_image.bmp>
```
# Assignment 3

On this assignment was used the [OpenMPI API](https://www.open-mpi.org/) in order to create parallel programs from certain applications. 
Also was used and [OpenMP](https://www.openmp.org/) for hybrid programming.

## First Program

The first program is about image filtering with Gauss Method and the use of MPI. 
In order to use MPI hostfiles were created with its hostfile containted the localhost and other computers. 
In particular, here where used three different hostfiles (or virtual machines), with the number would be 2, 4 or 8. 
These nodes have a quadcore processor each. So the number of processes will be 4xY = X, where Y is the number of nodes containt into the hostfile.

### How to run
```bash
    mpicc gaussian-blur.c -lm -o gb
    mpirun -np X -hostfile nodes_Y gb <radius> <input_image.bmp>
```
## Second Program

The second program is about matrix multiplication with another matrix, using hybrid programming OpenMP + MPI. 
Here collective communications were used, in order to distribute the data evenly to the nodes. With the help of OpenMP, parallel regions were created to compute the multiplication.

### How to run
```bash
    mpicc -fopenmp matmul_par.c -o matpar
    mpirun -np X -hostfile nodes_Y matpar
```
