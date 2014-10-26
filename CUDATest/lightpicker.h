#ifndef LIGHTPICKER_H
#define LIGHTPICKER_H

#include "arealight.h"

class LightPicker {
public:
    LightPicker();

    AreaLight*	GetLight(Shape* shape, const Color& color) const;
    void		IncreaseIntensity(float step);
    void		DecreaseIntensity(float step);
	
private:
	float							intensity;
};

#endif
