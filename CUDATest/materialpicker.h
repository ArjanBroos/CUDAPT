#pragma once

#include "lambertmaterial.h"
#include "mirrormaterial.h"

#define NR_COLORS 8

class MaterialPicker {
public:
	__device__				MaterialPicker();

	__device__ Material*	GetMaterial(const Color& color) const;
	__device__ void			NextType();
	__device__ void			IncreaseAlbedo(float step);
	__device__ void			DecreaseAlbedo(float step);

private:
	MaterialType			type;
	float					albedo;
};