#include <string> 	
#include <sstream>
#include <iostream>
#include <fstream>
#include "kernels.h"
bool checkOverwrite(std::string worldName) {
	std::ifstream readWorldFile(worldName);
	// Check existence and ask if overwriting is ok
	if(readWorldFile.is_open()) {
		std::cout << "Are you sure you want to overwrite world " << worldName << "? (y/n)" << std::endl;
		std::string answer;
		std::getline(std::cin, answer);
		readWorldFile.close();
		if(!(answer == "y")) {
			std::cout << "Save process canceled by the user" << std::endl;
			return true;
		}
	}
	return false;
}

void writeWorldFile(std::string fileName, std::stringstream &contents) {
	std::ofstream file(fileName);
	if(file.is_open())
		file << contents.rdbuf();
	else std::cout << "Unable to open file " << fileName << std::endl;
}

//void LaunchLoadBlocks(Scene* scene);

void LaunchLoadBlocks(Scene* scene) {
	// Ask user for the world name to load (for the filename)
	std::cout << "Which world would you like to open?" << std::endl;
	std::string worldName;
	std::getline(std::cin, worldName);
	worldName += ".wrld";
	std::ifstream readWorldFile(worldName);

	// Delete old scene
	LaunchEmptyScene(scene);

	// Declare pointers and allocate device pointers
	Point loc;
	Color col;
	float albedo, intensity;
	MaterialType mat;
	ShapeType shape;
	ObjectType type;

	// Check if world file exists and start load process
	if(readWorldFile.is_open()) {
		std::cout << "Loading world file: " << worldName << ", this can take a few minutes" << std::endl;
		std::string word;
		// Start reading the world block by block
		while( readWorldFile >> word) {
			if(word != "NewObject") {
				std::cout << "The world file is corrupt, world loading process is stopped" << std::endl;
				break;
			}

			readWorldFile >> word;
			if(word != "Location:") {
				std::cout << "The world file is corrupt, world loading process is stopped" << std::endl;
				break;
			}
			int x,y,z;
			readWorldFile >> x;
			readWorldFile >> y;
			readWorldFile >> z;
			loc = Point((float) x,(float) y, (float) z);

			readWorldFile >> word;
			if(word != "Color:") {
				std::cout << "The world file is corrupt, world loading process is stopped" << std::endl;
				break;
			}
			float r,g,b;
			readWorldFile >> r;
			readWorldFile >> g;
			readWorldFile >> b;
			col = Color(r,g,b);

			readWorldFile >> word;
			if(word != "Albedo:") {
				std::cout << "The world file is corrupt, world loading process is stopped" << std::endl;
				break;
			}
			readWorldFile >> albedo;

			readWorldFile >> word;
			if(word != "Intensity:") {
				std::cout << "The world file is corrupt, world loading process is stopped" << std::endl;
				break;
			}
			readWorldFile >> intensity;

			readWorldFile >> word;
			if(word != "Material:") {
				std::cout << "The world file is corrupt, world loading process is stopped" << std::endl;
				break;
			}
			int i;
			readWorldFile >> i;
			mat = (MaterialType) i;

			readWorldFile >> word;
			if(word != "Shape:") {
				std::cout << "The world file is corrupt, world loading process is stopped" << std::endl;
				break;
			}
			readWorldFile >> i;
			shape = (ShapeType) i;

			readWorldFile >> word;
			if(word != "ObjectType:") {
				std::cout << "The world file is corrupt, world loading process is stopped" << std::endl;
				break;
			}
			readWorldFile >> i;
			type = (ObjectType) i;

			// Create the block in the world
			LaunchLoadBlock(scene, loc, col, albedo, intensity, mat, shape, type);
			//LoadBlock<<<1,1>>>(scene, loc, col, albedo, intensity, mat, shape, type);
		}

	}
	else std::cout << "Unable to open file " << worldName << std::endl;

	// Close file and free the device pointers
	readWorldFile.close();

	std::cout << "Done" << std::endl;
}
