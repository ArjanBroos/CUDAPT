#include "color.h"

// Initializes color as black
Color::Color() : r(0.f), g(0.f), b(0.f) {
}

// Initializes color as (r, g, b)
Color::Color(float r, float g, float b) : r(r), g(g), b(b) {
}

Color Color::operator+(const Color& c) const {
	return Color(r + c.r, g + c.g, b + c.b);
}

Color& Color::operator+=(const Color& c) {
	r += c.r;
	g += c.g;
	b += c.b;
	return *this;
}

Color Color::operator*(const Color& c) const {
	return Color(r * c.r, g * c.g, b * c.b);
}

Color& Color::operator*=(const Color& c) {
	r *= c.r;
	g *= c.g;
	b *= c.b;
	return *this;
}

Color Color::operator*(float s) const {
	return Color(r * s, g * s, b * s);
}

Color& Color::operator*=(float s) {
	r *= s;
	g *= s;
	b *= s;
	return *this;
}

Color Color::operator/(float s) const {
	return Color(r / s, g / s, b / s);
}

Color& Color::operator/=(float s) {
	r /= s;
	g /= s;
	b /= s;
	return *this;
}