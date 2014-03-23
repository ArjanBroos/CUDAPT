#pragma once

#include "color.h"

#define NR_COLORS 8

class ColorPicker {
public:
	__device__			ColorPicker();

	__device__ Color	GetColor() const;
	__device__ void		NextColor();

private:
	Color				colors[NR_COLORS];
	unsigned			colorIndex;
};