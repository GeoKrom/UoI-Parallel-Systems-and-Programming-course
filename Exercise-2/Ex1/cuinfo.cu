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
		/* TODO: Retrieve and pretty-print all the necessary information */
	}
}

int main()
{
	cuinfo_print_devinfo();
	return 0;
}
