#ifndef MATERIALPICKER_H
#define MATERIALPICKER_H

#include "lambertmaterial.h"
#include "mirrormaterial.h"
#include "glassmaterial.h"

#define NR_COLORS 8

class MaterialPicker {
public:
    MaterialPicker();

    Material*	GetMaterial(const Color& color) const;
    void			NextType();
    void			IncreaseAlbedo(float step);
    void			DecreaseAlbedo(float step);

private:
	MaterialType			type;
	float					albedo;
};

#endif
