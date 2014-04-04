#pragma once

#include <string>

// Random number generator
// Used to precompute a lot of random numbers between 0 and 1
class RNG {
public:
	// Allocate memory for n random numbers
	RNG(unsigned n);
	// Deallocate memory
	~RNG();

	// Generate random numbers
	void		GenerateRandomValues();
	// Read random numbers from given file
	void		ReadRandomValues(const std::string& file);
	// Write random numbers to given file
	void		WriteRandomValues(const std::string& file) const;
	// Retrieve all random numbers
	float*		GetRandomValues() const;

private:
	float*		randomValues;	// All the random values between 0 and 1
	unsigned	n;				// Number of random values
};