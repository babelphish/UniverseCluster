// UniverseCluster.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <bitset>
#include <math.h>

#include "Respondent.h"
#include "UniverseCluster.h"

using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	ifstream file;

	string dataFilePath = "C:\\Dev\\UniverseCluster\\data\\lars_full_double.csv";
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
		respondent.mask = strtol(binaryDataString.c_str(), NULL, 2);
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

	int64 time1 = GetTimeMs64();
	solveProblems(10, grid, problemMap);
	int64 time2 = GetTimeMs64();
	cout << "Time: " <<  time2 - time1 << endl;
	system ("pause");
	return 0;
}


string formatMask(const bit_mask mask, const int numberOfProblems) 
{
	bitset<maxBits> maskSet(mask);
	string tempMask = maskSet.to_string();
	return tempMask.substr(tempMask.size() - numberOfProblems, numberOfProblems);
}

bool compareWithMask(const bit_mask solution, const bit_mask respondentMask)
{
	//totalComparisons++;
	bit_mask intermediateResult = solution ^ respondentMask;
	bit_mask finalResult = intermediateResult & respondentMask;
	return finalResult == 0;
}


int countMatchingRespondents(const bit_mask mask, const vector<Respondent>& eligibleRespondents) 
{
	int respondentCount = 0;
	for (unsigned int i = 0; i < eligibleRespondents.size(); i++) {
		if(compareWithMask(mask, eligibleRespondents[i].mask)) {
			respondentCount++;
		}
	}
	return respondentCount;
}


void solveProblems(const int maxDepth, const respondentGrid& grid, const respondentListMap& problemMap)
{
	vector<bit_mask> masks;
	vector<Respondent> eligibleRespondents;
	int numberOfProblems = grid[0].size() - 1;

	//generate all the masks we'll be using
	for (int x = 0; x < numberOfProblems; x++) {
		masks.push_back(pow(2.0F,x));
	}
	masks.push_back(0);

	//find all eligible respondents
	for (int x = 1; x <= maxDepth; x++)
	{
		if (problemMap.count(x) > 0)
		{
			eligibleRespondents.insert(eligibleRespondents.end(), problemMap.find(x)->second->begin(), problemMap.find(x)->second->end());
		}
	}

	bit_mask currentMask = 0;
	//calculate the 'max mask' so we know when we're done looping
	bit_mask maxMask = 0;
	for (int i = 1; i <= maxDepth; i++)
	{
		maxMask = maxMask ^ bit_mask(pow(2.0F,(numberOfProblems - i)));
	}

	int position = 0;
	int bestCount = 1;
	vector<bit_mask> maskArray;
	vector<long> positionArray;
	int currentDepth = maxDepth;

	for (int i = 0; i < maxDepth; i++)
	{
		positionArray.push_back(i);
	}

	bool done = false;
			
	currentMask = pow(2.0F,maxDepth) - 1;
	while(currentMask != maxMask)
	{
		bit_mask t = currentMask | (currentMask - 1); // t gets v's least significant 0 bits set to 1
		// Next set to 1 the most significant bit to change, 
		// set to 0 the least significant ones, and add the necessary 1 bits.
		unsigned long l;
		_BitScanForward(&l,currentMask);
		currentMask = ((t + 1) | ((~t & -~t) - 1) >> (l + 1));  

		int count = countMatchingRespondents(currentMask, eligibleRespondents);
		if (count > bestCount) 
		{
			bestCount = count;
			cout << "Best set so far: " << formatMask(currentMask, numberOfProblems) << " with " << bestCount << endl;
			//listMatchingRespondents(currentMask, eligibleRespondents);
		}
	}
}

int64 GetTimeMs64()
{
#ifdef WIN32
 /* Windows */
 FILETIME ft;
 LARGE_INTEGER li;

 /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
  * to a LARGE_INTEGER structure. */
 GetSystemTimeAsFileTime(&ft);
 li.LowPart = ft.dwLowDateTime;
 li.HighPart = ft.dwHighDateTime;

 uint64 ret = li.QuadPart;
 ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
 ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

 return ret;
#else
 /* Linux */
 struct timeval tv;

 gettimeofday(&tv, NULL);

 uint64 ret = tv.tv_usec;
 /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
 ret /= 1000;

 /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
 ret += (tv.tv_sec * 1000);

 return ret;
#endif
}
