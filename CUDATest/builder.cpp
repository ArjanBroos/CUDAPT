#include "builder.h"
#include "primitive.h"
#include "light.h"
#include "sphere.h"
#include "octree.h"

Builder::Builder() : buildType(BT_START), shapeType(ST_CUBE) {
}

Shape* Builder::GetShape(const Point& location) const {
	if (shapeType == ST_CUBE)
		return new Box(location);
	else if (shapeType == ST_SPHERE)
		return new Sphere(location);
	else
		return NULL;
}

Point* Builder::GetPosition(AreaLight* neighbor, const Point& isct) const {
	Vector n = neighbor->GetShape()->GetNormal(isct);
	Point *nl = &neighbor->GetParent()->bounds[0];
	return new Point((int)(nl->x + n.x + .5f), (int)(nl->y + n.y + .5f), (int)(nl->z + n.z + .5f));
}

Point* Builder::GetPosition(Primitive* neighbor, const Point& isct) const {
	if (neighbor->GetShape()->GetType() == ST_PLANE)
		return new Point(floor(isct.x), floor(isct.y), floor(isct.z));
	Vector n = neighbor->GetShape()->GetNormal(isct);
	Point *nl = &neighbor->GetParent()->bounds[0];
	return new Point((int)(nl->x + n.x + .5f), (int)(nl->y + n.y + .5f), (int)(nl->z + n.z + .5f));
}

Object* Builder::GetObject(Shape* shape, Point* location) const {
	if (buildType == BT_PRIMITIVE) {
		return new Primitive(shape, materialPicker.GetMaterial(colorPicker.GetColor()));
	}
	if (buildType == BT_LIGHT) {
		return lightPicker.GetLight(shape, colorPicker.GetColor());
	}
	
	return NULL;
}

void Builder::NextBuildType() {
	buildType = BuildType(buildType + 1);
	if (buildType == BT_END) buildType = BT_START;
}

void Builder::NextShapeType() {
	shapeType = ShapeType(shapeType + 1);
	if (shapeType == ST_END) shapeType = ST_START;
}

void Builder::SetPresetColor(unsigned index) {
	colorPicker.SetColor(index);
}

void Builder::NextMaterialType() {
	materialPicker.NextType();
}

void Builder::IncreaseAorI(float step) {
	if (buildType == BT_PRIMITIVE)
		materialPicker.IncreaseAlbedo(step/10.f);
	if (buildType == BT_LIGHT)
		lightPicker.IncreaseIntensity(step);
}

void Builder::DecreaseAorI(float step) {
	if (buildType == BT_PRIMITIVE)
		materialPicker.DecreaseAlbedo(step/10.f);
	if (buildType == BT_LIGHT)
		lightPicker.DecreaseIntensity(step);
}
