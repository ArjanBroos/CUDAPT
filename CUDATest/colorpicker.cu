#include "colorpicker.h"

__device__ ColorPicker::ColorPicker() : colorIndex(0) {
	colors[0] = Color(1.f, 0.f, 0.f);	// Red
	colors[1] = Color(0.f, 1.f, 0.f);	// Green
	colors[2] = Color(0.f, 0.f, 1.f);	// Blue
	colors[3] = Color(1.f, 1.f, 0.f);	// Yellow
	colors[4] = Color(0.f, 1.f, 1.f);	// Cyan
	colors[5] = Color(1.f, 0.f, 1.f);	// Purple
	colors[6] = Color(1.f, 1.f, 1.f);	// White
	colors[7] = Color(0.f, 0.f, 0.f);	// Black
}

__device__ Color ColorPicker::GetColor() const {
	return colors[colorIndex];
}

__device__ void ColorPicker::NextColor() {
	colorIndex++;
	if (colorIndex == NR_COLORS) colorIndex = 0;
}

__device__ void ColorPicker::SetColor(unsigned index) {
	colorIndex = index;
}