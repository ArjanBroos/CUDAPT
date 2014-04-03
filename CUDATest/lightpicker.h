#include "arealight.h"

class LightPicker {
public:
	__device__				LightPicker();

	__device__ AreaLight*	GetLight(Shape* shape, const Color& color) const;
	__device__ void		IncreaseIntensity(float step);
	__device__ void		DecreaseIntensity(float step);
	
private:
	float							intensity;
};