#ifndef MATH_H
#define MATH_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cmath>
#include <cfloat>

#ifdef INFINITY
#undef INFINITY
#endif
#define INFINITY FLT_MAX

#ifdef isinf
#undef isinf
#endif
#define isinf(f) (!_finite((f)))

#ifdef isnan
#undef isnan
#endif
#define isnan _isnan // IS Not A Number
#define M_PI 3.14159265358979323846
const float PI = (float)M_PI;

#endif
