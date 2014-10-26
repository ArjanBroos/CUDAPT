#include "object.h"

Node* Object::GetParent() {
	return parent;
}

void Object::SetParent(Node* parent) {
	this->parent = parent;
}
