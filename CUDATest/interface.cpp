#include "interface.h"

Interface::Interface(const unsigned WIDTH, const unsigned HEIGHT) {
	buildType = BT_PRIMITIVE;
	materialType = MT_DIFFUSE;
	shapeType = ST_CUBE;

	colors[0] = Color(1.f, 0.f, 0.f);	// Red
	colors[1] = Color(0.f, 1.f, 0.f);	// Green
	colors[2] = Color(0.f, 0.f, 1.f);	// Blue
	colors[3] = Color(1.f, 1.f, 0.f);	// Yellow
	colors[4] = Color(0.f, 1.f, 1.f);	// Cyan
	colors[5] = Color(1.f, 0.f, 1.f);	// Purple
	colors[6] = Color(1.f, 1.f, 1.f);	// White
	colors[7] = Color(0.f, 0.f, 0.f);	// Black
	colorIndex = 0;

	// Load textures for build icons
	diffCubeTex.loadFromFile("interface/diffCube.png");
	diffSphereTex.loadFromFile("interface/diffSphere.png");
	mirrCubeTex.loadFromFile("interface/mirrCube.png");
	mirrSphereTex.loadFromFile("interface/mirrSphere.png");
	lightCubeTex.loadFromFile("interface/lightCube.png");
	lightSphereTex.loadFromFile("interface/lightSphere.png");

	// Set up crosshair sprite
	crosshairTex.loadFromFile("interface/crosshair.png");
	crosshairSpr.setTexture(crosshairTex);
	crosshairSpr.setPosition((float)WIDTH/2.f-16.f, (float)HEIGHT/2.f-16.f);

	// Initialize first build icon
	SetBuildIcon();
	currentBuildIconSpr.setPosition(20.f, 20.f);
}

const sf::Sprite& Interface::GetBuildIcon() const {
	return currentBuildIconSpr;
}

const sf::Sprite& Interface::GetCrosshair() const {
	return crosshairSpr;
}

void Interface::SetScreenSize(unsigned width, unsigned height) {
	crosshairSpr.setPosition((float)width / 2.f, (float)height / 2.f);
}

void Interface::NextBuildType() {
	buildType = BuildType(buildType + 1);
	if (buildType == BT_END) buildType = BT_START;
	SetBuildIcon();
}

void Interface::NextMaterialType() {
	materialType = MaterialType(materialType + 1);
	if (materialType == MT_END) materialType = MT_START;
	SetBuildIcon();
}

void Interface::NextShapeType() {
	shapeType = ShapeType(shapeType + 1);
	if (shapeType == ST_END) shapeType = ST_START;
	SetBuildIcon();
}

void Interface::SetPresetColor(unsigned index) {
	colorIndex = index;
	SetBuildIcon();
}

sf::Color Interface::GetSFColor(const Color& color) const {
	return sf::Color(
		(sf::Uint8)(color.r * 255),
		(sf::Uint8)(color.g * 255),
		(sf::Uint8)(color.b * 255));
}

// One ugly motherfucker of a function
void Interface::SetBuildIcon() {
	// Primitives
	if (buildType == BT_PRIMITIVE) {
		if (shapeType == ST_CUBE) {
			if (materialType == MT_DIFFUSE) currentBuildIconTex = diffCubeTex;
			if (materialType == MT_MIRROR)	currentBuildIconTex = mirrCubeTex;
		}
		if (shapeType == ST_SPHERE) {
			if (materialType == MT_DIFFUSE) currentBuildIconTex = diffSphereTex;
			if (materialType == MT_MIRROR)	currentBuildIconTex = mirrSphereTex;
		}
	}

	// Lights
	if (buildType == BT_LIGHT) {
		if (shapeType == ST_CUBE)	currentBuildIconTex = lightCubeTex;
		if (shapeType == ST_SPHERE)	currentBuildIconTex = lightSphereTex;
	}

	currentBuildIconSpr.setTexture(currentBuildIconTex);
	currentBuildIconSpr.setColor(GetSFColor(colors[colorIndex]));
}