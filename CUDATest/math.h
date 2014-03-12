#pragma once

#include <cmath>
#include <cfloat>
#define INFINITY FLT_MAX
#define isinf(f) (!_finite((f)))
#define isnan _isnan // IS Not A Number
const float PI = (float)M_PI;