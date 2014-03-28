#pragma once

#include "builder.h"
#include "colorpicker.h"
#include <SFML\Graphics.hpp>

// Handles builder interface visualization
class Interface {
public:
	Interface(const unsigned WIDTH, const unsigned HEIGHT);

	const sf::Sprite&	GetBuildIcon() const;
	const sf::Sprite&	GetCrosshair() const;

	void				SetScreenSize(unsigned width, unsigned height);

	void				NextBuildType();
	void				NextMaterialType();
	void				NextShapeType();
	void				SetPresetColor(unsigned index);

private:
	sf::Color			GetSFColor(const Color& color) const;
	void				SetBuildIcon();

	BuildType			buildType;
	MaterialType		materialType;
	ShapeType			shapeType;

	sf::Texture			diffCubeTex, diffSphereTex;		// Diffuse textures
	sf::Texture			mirrCubeTex, mirrSphereTex;		// Mirror textures
	sf::Texture			lightCubeTex, lightSphereTex;	// Light textures

	sf::Texture			currentBuildIconTex;			// Texture for the current build selection
	sf::Sprite			currentBuildIconSpr;			// Sprite that represents the current build selection

	sf::Texture			crosshairTex;
	sf::Sprite			crosshairSpr;

	unsigned			colorIndex;						// Index of currently selected preset color
	Color				colors[NR_COLORS];				// Preset colors
};