#pragma once
#include <iostream>

#define FRAC_ASSERT(what) do { if(!(what)) { std::cout << "Assert failed: " << __LINE__ << ' ' << __FILE__ << ' ' << #what << '\n'; exit(0); } } while(0); 
