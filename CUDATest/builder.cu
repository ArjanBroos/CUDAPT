#include "builder.h"
#include "primitive.h"
#include "light.h"

__device__ Builder::Builder() : buildType(BT_START) {
}

__device__ Shape* Builder::GetShape(const Point& location) const {
	return new Box(location);
}

__device__ Point* Builder::GetPosition(AreaLight* neighbor, const Point& isct) const {
	Vector n = neighbor->GetShape()->GetNormal(isct);
	Point* nl = neighbor->loc;
	return new Point(*nl + n + Vector(0.5f, 0.5f, 0.5f));
}

__device__ Point* Builder::GetPosition(Primitive* neighbor, const Point& isct) const {
	if (neighbor->type == PLANE)
		return new Point(floor(isct.x), floor(isct.y), floor(isct.z));
	Vector n = neighbor->GetShape()->GetNormal(isct);
	Point* nl = neighbor->loc;
	return new Point(*nl + n + Vector(0.5f, 0.5f, 0.5f));
}

__device__ Object* Builder::GetObject(Shape* shape, Point* location) const {
	if (buildType == BT_PRIMITIVE)
		return new Primitive(shape, materialPicker.GetMaterial(colorPicker.GetColor()), location);
	if (buildType == BT_LIGHT)
		return lightPicker.GetLight(shape, colorPicker.GetColor(), location);
	
	return NULL;
}

__device__ void Builder::NextBuildType() {
	buildType = BuildType(buildType + 1);
	if (buildType == BT_END) buildType = BT_START;
}

__device__ void Builder::NextColor() {
	colorPicker.NextColor();
}

__device__ void Builder::NextMaterialType() {
	materialPicker.NextType();
}

__device__ void Builder::IncreaseAorI(float step) {
	if (buildType == BT_PRIMITIVE)
		materialPicker.IncreaseAlbedo(step);
	if (buildType == BT_LIGHT)
		lightPicker.IncreaseIntensity(step);
}

__device__ void Builder::DecreaseAorI(float step) {
	if (buildType == BT_PRIMITIVE)
		materialPicker.DecreaseAlbedo(step);
	if (buildType == BT_LIGHT)
		lightPicker.DecreaseIntensity(step);
}