#include "StdAfx.h"
#include "Respondent.h"


Respondent::Respondent(void)
{
}


string Respondent::toString()
{
	stringstream output;
	output << this->enty_id << ":" << this->strMask;
	return output.str();
}

Respondent::~Respondent(void)
{
}
