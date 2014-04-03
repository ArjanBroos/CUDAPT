#include "builder.h"
#include "primitive.h"
#include "light.h"
#include "sphere.h"

__device__ Builder::Builder() : buildType(BT_START), shapeType(ST_CUBE) {
}

__device__ Shape* Builder::GetShape(const Point& location) const {
	if (shapeType == ST_CUBE)
		return new Box(location);
	else if (shapeType == ST_SPHERE)
		return new Sphere(location);
	else
		return NULL;
}

__device__ Point* Builder::GetPosition(AreaLight* neighbor, const Point& isct) const {
	Vector n = neighbor->GetShape()->GetNormal(isct);
	Point* nl = neighbor->loc;
	return new Point((int)(nl->x + n.x + .5f), (int)(nl->y + n.y + .5f), (int)(nl->z + n.z + .5f));
}

__device__ Point* Builder::GetPosition(Primitive* neighbor, const Point& isct) const {
	if (neighbor->GetShape()->GetType() == ST_PLANE)
		return new Point(floor(isct.x), floor(isct.y), floor(isct.z));
	Vector n = neighbor->GetShape()->GetNormal(isct);
	Point* nl = neighbor->loc;
	return new Point((int)(nl->x + n.x + .5f), (int)(nl->y + n.y + .5f), (int)(nl->z + n.z + .5f));
}

__device__ Object* Builder::GetObject(Shape* shape, Point* location) const {
	if (buildType == BT_PRIMITIVE) {
		//Object* newPrim = new Primitive(shape, materialPicker.GetMaterial(colorPicker.GetColor()), location);
		//newPrim->type = PRIMITIVE;
		return new Primitive(shape, materialPicker.GetMaterial(colorPicker.GetColor()), location);
	}
	if (buildType == BT_LIGHT) {
		//Object* newLight = lightPicker.GetLight(shape, colorPicker.GetColor(), location);
		//newLight->type = LIGHT;
		return lightPicker.GetLight(shape, colorPicker.GetColor(), location);
	}
	
	return NULL;
}

__device__ void Builder::NextBuildType() {
	buildType = BuildType(buildType + 1);
	if (buildType == BT_END) buildType = BT_START;
}

__device__ void Builder::NextShapeType() {
	shapeType = ShapeType(shapeType + 1);
	if (shapeType == ST_END) shapeType = ST_START;
}

__device__ void Builder::SetPresetColor(unsigned index) {
	colorPicker.SetColor(index);
}

__device__ void Builder::NextMaterialType() {
	materialPicker.NextType();
}

__device__ void Builder::IncreaseAorI(float step) {
	if (buildType == BT_PRIMITIVE)
		materialPicker.IncreaseAlbedo(step/10.f);
	if (buildType == BT_LIGHT)
		lightPicker.IncreaseIntensity(step);
}

__device__ void Builder::DecreaseAorI(float step) {
	if (buildType == BT_PRIMITIVE)
		materialPicker.DecreaseAlbedo(step/10.f);
	if (buildType == BT_LIGHT)
		lightPicker.DecreaseIntensity(step);
}