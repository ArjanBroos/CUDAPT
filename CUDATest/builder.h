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
    Builder();

    Shape*	GetShape(const Point& location) const;
    Point*	GetPosition(AreaLight* neighbor, const Point& isct) const;
    Point*	GetPosition(Primitive* neighbor, const Point& isct) const;
    Object*	GetObject(Shape* shape, Point* location) const;

    void		NextBuildType();
    void		NextShapeType();
    void		SetPresetColor(unsigned colorIndex);
    void		NextMaterialType();
    void		IncreaseAorI(float step);	// Increase albedo or intensity
    void		DecreaseAorI(float step);	// Decrease albedo or intensity

private:
	BuildType			buildType;
	ShapeType			shapeType;
	ColorPicker			colorPicker;
	MaterialPicker		materialPicker;
	LightPicker			lightPicker;
};

#endif
