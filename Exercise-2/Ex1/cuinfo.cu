#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* 
 * Retrieves and prints information for every installed NVIDIA
 * GPU device
 */
void cuinfo_print_devinfo()
{
	int num_devs, i;
	cudaDeviceProp dev_prop;
	
	cudaGetDeviceCount(&num_devs);
	if (num_devs == 0)
	{
		printf("No CUDA devices found.\n");
		return;
	}

	for (i = 0; i < num_devs; i++)
	{
		cudaGetDeviceProperties(&dev_prop, i);
		/* TODO: Retrieve and pretty-print all the necessary information */
		printf("----------------------------------------------------------\n");	
		printf("Information for CUDA Device - %d\n", i+1);
		printf("----------------------------------------------------------\n");	
		printf("Device Name: %s\n", dev_prop.name);
		printf("Device CUDA compute capability: %d.%d\n", dev_prop.major, dev_prop.minor);
		printf("Device number of streaming multiprocessors: %d\n", dev_prop.multiProcessorCount);
		printf("Device max number of threads per block: %d\n", dev_prop.maxThreadsPerBlock);
		printf("Device size of global memory: %u bytes\n", dev_prop.totalGlobalMem);
		printf("Device size of shared memory per block: %u bytes\n", dev_prop.sharedMemPerBlock);
		printf("----------------------------------------------------------\n");
	}
}

int main()
{
	cuinfo_print_devinfo();
	return 0;
}
