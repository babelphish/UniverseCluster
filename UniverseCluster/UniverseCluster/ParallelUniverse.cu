
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

__global__ void inc_ranges(const bit_mask *ranges, bit_mask*o_ranges,int rangesLength,const bit_mask *respondents, int respondentsLength)
{
	extern __shared__ bit_mask shared_masks[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x < respondentsLength)
	{
		shared_masks[threadIdx.x] = respondents[threadIdx.x];
	}
	__syncthreads();
	int bestCount = 0;
	unsigned long long currentMask = ranges[idx];
	unsigned long long maxMask = ranges[idx + 1];
  	while(currentMask <= maxMask)
	{
		int respondentCount = 0;
		for (unsigned int i = 0; i < respondentsLength; i++) {
			bit_mask intermediateResult = currentMask ^ shared_masks[i];
			if ((intermediateResult & shared_masks[i]) == 0) {
				respondentCount++;
			}
		}

		if (respondentCount > bestCount)
		{
			bestCount = respondentCount;
		}

		bit_mask t = currentMask | (currentMask - 1); // t gets v's least significant 0 bits set to 1
		// Next set to 1 the most significant bit to change, 
		// set to 0 the least significant ones, and add the necessary 1 bits.
		unsigned int l = __ffsll(currentMask);

		currentMask = ((t + 1) | ((~t & -~t) - 1) >> l);  
	}
	o_ranges[idx] = bestCount;
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
	StopWatchInterface *hTimer = NULL;

	unsigned long long totalCombos = calcBitCombosFact(maxProblemsSolved,numberOfProblems);

	const int divisions = 512;
	const int threads = 256;

	int totalSteps = threads * divisions;
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

	sdkCreateTimer(&hTimer);
	sdkStartTimer(&hTimer);

	//allocate device ranges
	checkCudaErrors(cudaMalloc((void **)&d_ranges, sizeof(bit_mask) * (totalSteps + 1)));
	checkCudaErrors(cudaMalloc((void **)&d_o_ranges, sizeof(bit_mask) * (totalSteps + 1)));
	checkCudaErrors(cudaMalloc((void **)&d_respondents, sizeof(bit_mask) * eligibleRespondents.size()));

	//copy to device

	checkCudaErrors(cudaMemcpy(d_ranges, h_ranges, sizeof(bit_mask) * (totalSteps + 1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_respondents, h_respondents, sizeof(bit_mask) * eligibleRespondents.size(), cudaMemcpyHostToDevice));
	
	inc_ranges<<<divisions,threads, sizeof(bit_mask) * eligibleRespondents.size()>>>(d_ranges, d_o_ranges, totalSteps, d_respondents, eligibleRespondents.size());
	cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(h_o_ranges, d_o_ranges, sizeof(bit_mask) * (totalSteps + 1), cudaMemcpyDeviceToHost));

	sdkStopTimer(&hTimer);
	printf("Processing time: %f (ms)\n", sdkGetTimerValue( &hTimer) );
    sdkDeleteTimer(&hTimer);

	int best = 0;
	for (int i = 0; i < totalSteps; i++)
	{
		if (h_o_ranges[i] > best)
		{
			best = h_o_ranges[i];
		}
	}
	cout << "best" << best << endl;

	checkCudaErrors(cudaFree(d_o_ranges));
    checkCudaErrors(cudaFree(d_ranges));
    checkCudaErrors(cudaFree(d_respondents));

    cudaDeviceReset();
	
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
