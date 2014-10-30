#ifndef APPLICATION_H
#define APPLICATION_H

#include "camera.h"
#include "color.h"
#include "kernels.h"
#include "interface.h"
#include <string>

class Application {
public:
    Application(const std::string& title, unsigned width, unsigned height);
	~Application();

	void				Render();					// Calculate one sample per pixel and show output (cumulatively until reset)
	void				Reset();					// Start a new calculation, call whenever camera moves

	bool				HandleEvents();				// Handles any window related events and single key press events
													// Returns false if the window should close and the application should stop
	void				HandleKeyboard();			// Handle reactions to keyboard state (movement mostly)
	void				HandleMouse();				// Handle reactions to mouse movement and presses (rotating camera)
    void                UpdateDeviceCamera();

private:
	void				UpdateTitle();				// Update the title according to iterations and camera position

    Camera*				cam;        				// Camera
    Color*				result;         			// Raw results of tracing
    unsigned char*		pixelData;              	// Pixel data fed to SFML for display
	unsigned			iteration;					// How many'th samples per pixel are we going to calculate?

    Scene*				scene;          			// The scene holding all objects
													// scene is a host pointer to a device pointer
    Builder*			builder;                    // Used for adding objects to the scene

	unsigned			width, height;				// Dimensions of screen in pixels
	unsigned			tileSize;					// A tile of tileSize*tileSize pixels is assigned per CUDA block

//	sf::RenderWindow	window;						// The window on which our application is displayed
//	sf::Vector2i		midScreen;					// Keep track of the middle of our screen portion of the window

//	Interface			interface;					// Handles rendering of the interface

//	sf::Texture			texture;					// Texture used to display our pixel data
//	sf::Image			image;						// Image used to display our pixel data
//	sf::Sprite			sprite;						// Sprite used to display our pixel data

	bool				frozen;						// Is movement being denied?
};

#endif
