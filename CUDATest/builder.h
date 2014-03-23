#pragma once

#include "colorpicker.h"
#include "materialpicker.h"
#include "lightpicker.h"
#include "object.h"

enum BuildType {
	BT_START = 0,
	BT_LIGHT = BT_START,
	BT_PRIMITIVE,
	BT_END
};

class Builder {
public:
	__device__			Builder();

	__device__ Object*	GetObject(Shape* shape, Point* location) const;

	__device__ void		NextBuildType();
	__device__ void		NextColor();
	__device__ void		NextMaterialType();
	__device__ void		IncreaseAorI(float step);	// Increase albedo or intensity
	__device__ void		DecreaseAorI(float step);	// Decrease albedo or intensity

private:
	BuildType			buildType;
	ColorPicker			colorPicker;
	MaterialPicker		materialPicker;
	LightPicker			lightPicker;
};