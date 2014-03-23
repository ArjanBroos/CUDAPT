#include "lightpicker.h"

__device__ LightPicker::LightPicker() {
	intensity = 1.f;
}

__device__ AreaLight* LightPicker::GetLight(Shape* shape, const Color& color, Point* p) const {
	return new AreaLight(shape, color, intensity, p);
}

__device__ void LightPicker::IncreaseIntensity(float step) {
	intensity += step;
}

__device__ void LightPicker::DecreaseIntensity(float step) {
	intensity -= step;
	if (intensity < 0.f) intensity = 0.f;
}