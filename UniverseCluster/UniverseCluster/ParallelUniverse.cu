/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This is a simple test showing huge access speed gap
 * between aligned and misaligned structures
 * (those having/missing __align__ keyword).
 * It measures per-element copy throughput for
 * aligned and misaligned structures on
 * big chunks of data.
 */


// includes, system
#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_functions.h> // helper utility functions 
#include <helper_cuda.h>      // helper functions for CUDA error checking and initialization
#include "UniverseCluster.h"

////////////////////////////////////////////////////////////////////////////////
// Misaligned types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned char uint8;

typedef unsigned short int uint16;

typedef struct
{
    unsigned char r, g, b, a;
} RGBA8_misaligned;

typedef struct
{
    unsigned int l, a;
} LA32_misaligned;

typedef struct
{
    unsigned int r, g, b;
} RGB32_misaligned;

typedef struct
{
    unsigned int r, g, b, a;
} RGBA32_misaligned;



////////////////////////////////////////////////////////////////////////////////
// Aligned types
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(4)
{
    unsigned char r, g, b, a;
} RGBA8;

typedef unsigned int I32;

typedef struct __align__(8)
{
    unsigned int l, a;
} LA32;

typedef struct __align__(16)
{
    unsigned int r, g, b;
} RGB32;

typedef struct __align__(16)
{
    unsigned int r, g, b, a;
} RGBA32;


////////////////////////////////////////////////////////////////////////////////
// Because G80 class hardware natively supports global memory operations
// only with data elements of 4, 8 and 16 bytes, if structure size
// exceeds 16 bytes, it can't be efficiently read or written,
// since more than one global memory non-coalescable load/store instructions
// will be generated, even if __align__ option is supplied.
// "Structure of arrays" storage strategy offers best performance
// in general case. See section 5.1.2 of the Programming Guide.
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(16)
{
    RGBA32 c1, c2;
} RGBA32_2;



////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b)
{
    return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b)
{
    return a - a % b;
}



////////////////////////////////////////////////////////////////////////////////
// Simple CUDA kernel.
// Copy is carried out on per-element basis,
// so it's not per-byte in case of padded structures.
////////////////////////////////////////////////////////////////////////////////
template<class TData> __global__ void testKernel(
    TData *d_odata,
    TData *d_idata,
    int numElements
)
{
    const int        tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;

    for (int pos = tid; pos < numElements; pos += numThreads)
    {
        d_odata[pos] = d_idata[pos];
    }
}



////////////////////////////////////////////////////////////////////////////////
// Validation routine for simple copy kernel.
// We must know "packed" size of TData (number_of_fields * sizeof(simple_type))
// and compare only these "packed" parts of the structure,
// containig actual user data. The compiler behavior with padding bytes
// is undefined, since padding is merely a placeholder
// and doesn't contain any user data.
////////////////////////////////////////////////////////////////////////////////
template<class TData> int testCPU(
    TData *h_odata,
    TData *h_idata,
    int numElements,
    int packedElementSize
)
{
    for (int pos = 0; pos < numElements; pos++)
    {
        TData src = h_idata[pos];
        TData dst = h_odata[pos];

        for (int i = 0; i < packedElementSize; i++)
            if (((char *)&src)[i] != ((char *)&dst)[i])
            {
                return 0;
            }
    }

    return 1;
}



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//Memory chunk size in bytes. Reused for test
const int       MEM_SIZE = 50000000;
const int NUM_ITERATIONS = 32;

//GPU input and output data
unsigned char *d_idata, *d_odata;
//CPU input data and instance of GPU output data
unsigned char *h_idataCPU, *h_odataGPU;
StopWatchInterface *hTimer = NULL;



template<class TData> int runTest(int packedElementSize, int memory_size)
{
    const int totalMemSizeAligned = iAlignDown(memory_size, sizeof(TData));
    const int         numElements = iDivDown(memory_size, sizeof(TData));

    //Clean output buffer before current test
    checkCudaErrors(cudaMemset(d_odata, 0, memory_size));
    //Run test
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        testKernel<TData><<<64, 256>>>(
            (TData *)d_odata,
            (TData *)d_idata,
            numElements
        );
        getLastCudaError("testKernel() execution failed\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;
    printf(
        "Avg. time: %f ms / Copy throughput: %f GB/s.\n", gpuTime,
        (double)totalMemSizeAligned / (gpuTime * 0.001 * 1073741824.0)
    );

    //Read back GPU results and run validation
    checkCudaErrors(cudaMemcpy(h_odataGPU, d_odata, memory_size, cudaMemcpyDeviceToHost));
    int flag = testCPU(
                   (TData *)h_odataGPU,
                   (TData *)h_idataCPU,
                   numElements,
                   packedElementSize
               );

    printf(flag ? "\tTEST OK\n" : "\tTEST FAILURE\n");

    return !flag;
}

__device__  bool compareWithMask(bit_mask solution, bit_mask respondentMask)
{
	//totalComparisons++;
	bit_mask intermediateResult = solution ^ respondentMask;
	bit_mask finalResult = intermediateResult & respondentMask;
	return finalResult == 0;
}

__device__ int countMatchingRespondents(bit_mask mask, bit_mask* eligibleRespondents, int eligibleRespondentsLength) 
{
	int respondentCount = 0;
	for (unsigned int i = 0; i < eligibleRespondentsLength; i++) {
		if(compareWithMask(mask, eligibleRespondents[i])) {
			respondentCount++;
		}
	}
	return respondentCount;
}

__global__ void inc_ranges(bit_mask *ranges, bit_mask*o_ranges, int rangesLength, bit_mask *respondents, int respondentsLength)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int count = 0;
	int bestCount = 0;
	if (idx < rangesLength)
	{
		unsigned long long currentMask = ranges[idx];
		unsigned long long maxMask = ranges[idx + 1];
  		while(currentMask < maxMask)
		{
			
			int count = countMatchingRespondents(currentMask, respondents, respondentsLength);
			if (count > bestCount)
			{
				bestCount = count;
			}
			
			bit_mask t = currentMask | (currentMask - 1); // t gets v's least significant 0 bits set to 1
			// Next set to 1 the most significant bit to change, 
			// set to 0 the least significant ones, and add the necessary 1 bits.
			unsigned int l = __ffsll(currentMask);
			currentMask = ((t + 1) | ((~t & -~t) - 1) >> (l + 1));  
		}
		o_ranges[idx] = bestCount;
	}
}

int main(int argc, char **argv)
{

    int devID;
    cudaDeviceProp deviceProp;
    printf("[%s] - Starting...\n", argv[0]);

    // find first CUDA device
    devID = findCudaDevice(argc, (const char **)argv);

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("[%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n",
           deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    // Anything that is less than 192 Cores will have a scaled down workload
//    float scale_factor = max((192.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)), 1.0f);

	ifstream file;

	string dataFilePath = "C:\\Users\\Fish\\Documents\\GitHub\\UniverseCluster\\data\\lars_full_double.csv";
	file.open(dataFilePath);
	string respondentDataRow;
	respondentGrid grid;
	respondentListMap problemMap;

	while (getline(file, respondentDataRow))
	{
		stringstream respondentDataRowStream(respondentDataRow);
		vector<bit_mask> row;
		long respondentDatum;

		int index = 0;
		int problemCount = 0;
		string binaryDataString = "";
		Respondent respondent;

		while (respondentDataRowStream >> respondentDatum)
		{
			index++;
			if (index == 1) //then this is the enty_id
			{
				respondent.enty_id = respondentDatum;
			}
			else //this is binary, comma delimited data
			{
				if (respondentDatum == 0)
				{
					binaryDataString += "0";
				}
				else if (respondentDatum == 1)
				{
					binaryDataString += "1";
					problemCount++;
				}
			}
			row.push_back(respondentDatum);

			if (respondentDataRowStream.peek() == ',')
				respondentDataRowStream.ignore();
		}
		respondent.mask = _strtoi64(binaryDataString.c_str(), NULL, 2);
		respondent.strMask = binaryDataString;
		respondent.problemCount = problemCount;
		vector<Respondent>* problemCountRespondentList;
		if (problemMap.count(problemCount) == 0)
		{
			vector<Respondent>* emptyRespondentList = new vector<Respondent>();
			problemCountRespondentList = emptyRespondentList;
			problemMap.insert(pair<int,vector<Respondent>*>(problemCount,emptyRespondentList));
		}
		else
		{
			problemCountRespondentList = problemMap.find(problemCount)->second;
		}
		problemCountRespondentList->push_back(respondent);
		grid.push_back(row);
	}
	for (int x = 0; x < 20; x++) 
	{
		if (problemMap.count(x) != 0)
		{
			cout << "Has " << x << " problems: " << problemMap.find(x)->second->size() <<endl;
		}
	}

	const int maxProblemsSolved = 10;
	
	vector<bit_mask> masks;
	vector<Respondent> eligibleRespondents;
	int numberOfProblems = grid[0].size() - 1;

	//find all eligible respondents
	for (int x = 1; x <= maxProblemsSolved; x++)
	{
		if (problemMap.count(x) > 0)
		{
			eligibleRespondents.insert(eligibleRespondents.end(), problemMap.find(x)->second->begin(), problemMap.find(x)->second->end());
		}
	}

    sdkCreateTimer(&hTimer);

	unsigned long long totalCombos = calcBitCombosFact(maxProblemsSolved,numberOfProblems);

	int totalSteps = 192 * 8;
	unsigned long long step = totalCombos / totalSteps;
	
	//allocate ranges
	bit_mask* h_ranges = new bit_mask[totalSteps + 1];
	bit_mask* h_o_ranges = new bit_mask[totalSteps + 1];

	for (int i = 0; i < totalSteps; i++)
	{
		unsigned long long num = i * step;
		if (i == 0)
		{
			num = 1;
		}
		h_ranges[i] = getNthSet(num,maxProblemsSolved,numberOfProblems);
	}
	h_ranges[totalSteps] = getNthSet(totalCombos,maxProblemsSolved,numberOfProblems);

	//allocate respondents
	bit_mask* h_respondents = new bit_mask[eligibleRespondents.size()];
	for (unsigned int i = 0; i < eligibleRespondents.size(); i++)
	{
		h_respondents[i] = eligibleRespondents[i].mask;
	}

	bit_mask* d_ranges;
	bit_mask* d_respondents;
	bit_mask* d_o_ranges;

	//allocate device ranges
	checkCudaErrors(cudaMalloc((void **)&d_ranges, sizeof(bit_mask) * (totalSteps + 1)));
	checkCudaErrors(cudaMalloc((void **)&d_o_ranges, sizeof(bit_mask) * (totalSteps + 1)));
	checkCudaErrors(cudaMalloc((void **)&d_respondents, sizeof(bit_mask) * eligibleRespondents.size()));

	//copy to device
	checkCudaErrors(cudaMemcpy(d_ranges, h_ranges, sizeof(bit_mask) * (totalSteps + 1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_respondents, h_respondents, sizeof(bit_mask) * eligibleRespondents.size(), cudaMemcpyHostToDevice));


	inc_ranges<<<8,192>>>(d_ranges, d_o_ranges, totalSteps, d_respondents, eligibleRespondents.size());
	cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(h_o_ranges, d_o_ranges, sizeof(bit_mask) * (totalSteps + 1), cudaMemcpyDeviceToHost));

	int best = 0;
	for (int i = 0; i < totalSteps; i++)
	{
		if (h_o_ranges[i] > best)
		{
			best = h_o_ranges[i];
		}
	}
	cout << best << endl;

    //checkCudaErrors(cudaFree(d_ranges));
    //checkCudaErrors(cudaFree(d_respondents));

    sdkDeleteTimer(&hTimer);
    //cudaDeviceReset();
	
	cudaError_t cudaResult = cudaGetLastError();
	if (cudaResult != cudaSuccess)
	{
		cout << cudaGetErrorString(cudaResult) << endl;
		// Do whatever you want here
		// I normally create a std::string msg with a description of where I am
		// and append 
	}

    exit(EXIT_SUCCESS);
}
