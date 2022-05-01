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

# Assignment 3
