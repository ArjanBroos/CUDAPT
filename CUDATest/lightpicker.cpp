#include "lightpicker.h"

LightPicker::LightPicker() {
	intensity = 1.f;
}

AreaLight* LightPicker::GetLight(Shape* shape, const Color& color) const {
	return new AreaLight(shape, color, intensity);
}

void LightPicker::IncreaseIntensity(float step) {
	intensity += step;
}

void LightPicker::DecreaseIntensity(float step) {
	intensity -= step;
	if (intensity < 0.f) intensity = 0.f;
}
