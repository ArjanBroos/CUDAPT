#ifndef COLORPICKER_H
#define COLORPICKER_H

#include "color.h"

#define NR_COLORS 8

class ColorPicker {
public:
    ColorPicker();

    Color	GetColor() const;
    void		NextColor();
    void		SetColor(unsigned index);

private:
	Color				colors[NR_COLORS];
	unsigned			colorIndex;
};

#endif
