#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

#include "Respondent.h"
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <bitset>
#include <math.h>
#include <intrin.h>

using namespace std;

typedef unsigned long long bit_mask;
typedef long long int64; 
typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef vector<vector<bit_mask>> respondentGrid;
typedef map<int,vector<Respondent>*> respondentListMap;
const unsigned int maxBits = 64;

int64 GetTimeMs64();
unsigned long long getNthSet(unsigned long long n, unsigned int numPicked, unsigned int numElements);
unsigned long long calcBitCombosFact(unsigned int numPicked, unsigned int numElements);
void testIterator();
string formatMask(bit_mask mask, int numberOfProblems);
void solveProblems(int maxDepth, const respondentGrid& grid, const respondentListMap& problemMap);
void incrementMask(bit_mask &currentMask);