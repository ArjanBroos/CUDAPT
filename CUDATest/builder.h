#ifndef BUILDER_H
#define BUILDER_H

#include "colorpicker.h"
#include "materialpicker.h"
#include "lightpicker.h"
#include "object.h"
#include "arealight.h"
#include "primitive.h"
#include "box.h"

enum BuildType {
	BT_START = 0,
	BT_PRIMITIVE = BT_START,
	BT_LIGHT,
	BT_END
};

class Builder {
public:
	__device__			Builder();

	__device__ Shape*	GetShape(const Point& location) const;
	__device__ Point*	GetPosition(AreaLight* neighbor, const Point& isct) const;
	__device__ Point*	GetPosition(Primitive* neighbor, const Point& isct) const;
	__device__ Object*	GetObject(Shape* shape, Point* location) const;

	__device__ void		NextBuildType();
	__device__ void		NextShapeType();
	__device__ void		SetPresetColor(unsigned colorIndex);
	__device__ void		NextMaterialType();
	__device__ void		IncreaseAorI(float step);	// Increase albedo or intensity
	__device__ void		DecreaseAorI(float step);	// Decrease albedo or intensity

private:
	BuildType			buildType;
	ShapeType			shapeType;
	ColorPicker			colorPicker;
	MaterialPicker		materialPicker;
	LightPicker			lightPicker;
};

#endif
