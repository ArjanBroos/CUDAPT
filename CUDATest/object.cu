#include "object.h"

__device__ Node* Object::GetParent() {
	return parent;
}

__device__ void Object::SetParent(Node* parent) {
	this->parent = parent;
}