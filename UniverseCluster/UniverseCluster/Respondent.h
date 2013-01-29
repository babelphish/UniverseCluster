#pragma once
#include <string>
#include <sstream>

using namespace std;

class Respondent
{
public:
	long enty_id;
	string strMask;
	unsigned long long mask;
	int problemCount;

	Respondent(void);
	string toString();

	virtual ~Respondent(void);

};

