#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

typedef long long int64; typedef unsigned long long uint64;
typedef vector<vector<long>> respondentGrid ;
typedef map<int,vector<Respondent>*> respondentListMap ;
const unsigned int maxBits = 64;

int64 GetTimeMs64();
void solveProblems(const int maxDepth, const respondentGrid& grid, const respondentListMap& problemMap);