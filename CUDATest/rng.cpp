#include "rng.h"
#include <random>
#include <fstream>

// Allocate memory for (width * height * nrPerPixel) random numbers
RNG::RNG(unsigned n) {
	randomValues = new float[n];
	this->n = n;
}

// Deallocate memory
RNG::~RNG() {
	delete[] randomValues;
}

// Generate random numbers
void RNG::GenerateRandomValues() {
	std::mt19937 mt;
	std::uniform_real_distribution<> urd;
	for (unsigned i = 0; i < n; i++)
		randomValues[i] = urd(mt);
}

// Read random numbers from given file
void RNG::ReadRandomValues(const std::string& file) {
	std::ifstream stream(file.c_str());
	for (unsigned i = 0; i < n; i++)
		stream >> randomValues[i];
}

// Write random numbers to given file
void RNG::WriteRandomValues(const std::string& file) const {
	std::ofstream stream(file.c_str());
	for (unsigned i = 0; i < n; i++)
		stream << randomValues[i];
}

// Retrieve all random numbers
float* RNG::GetRandomValues() const {
	return randomValues;
}