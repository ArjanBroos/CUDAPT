#include "materialpicker.h"

MaterialPicker::MaterialPicker() : type(MT_START), albedo(1.f) {
}

Material* MaterialPicker::GetMaterial(const Color& color) const {
	if (type == MT_DIFFUSE)
		return new LambertMaterial(color, albedo);
	if (type == MT_MIRROR)
		return new MirrorMaterial(color, albedo);
	if (type == MT_GLASS)
		return new GlassMaterial(color, 0.f, 1.f, 1.5f);

	return NULL;
}

void MaterialPicker::NextType() {
	type = MaterialType(type + 1);
	if (type == MT_END) type = MT_START;
}

void MaterialPicker::IncreaseAlbedo(float step) {
	albedo += step;
	if (albedo > 1.f) albedo = 1.f;
}

void MaterialPicker::DecreaseAlbedo(float step) {
	albedo -= step;
	if (albedo < 0.f) albedo = 0.f;
}
