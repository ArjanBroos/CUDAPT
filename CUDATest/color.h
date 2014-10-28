#ifndef COLOR_H
#define COLOR_H

class Color {
public:
	float r, g, b;	// Color components

    Color();
    Color(float r, float g, float b);

    Color operator+(const Color& c) const;
    Color& operator+=(const Color& c);
    Color operator*(const Color& c) const;
    Color& operator*=(const Color& c);
    Color operator*(float s) const;
    Color& operator*=(float s);
    Color operator/(float s) const;
    Color& operator/=(float s);
};

inline Color operator*(float s, const Color& c) {
	return Color(c.r * s, c.g * s, c.b * s);
}

#endif
