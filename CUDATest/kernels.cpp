#include "kernels.h"
#include "sphere.h"
#include "intrec.h"
#include "lambertmaterial.h"
#include "arealight.h"
#include "plane.h"
#include "mirrormaterial.h"
#include "box.h"
#include "point.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>


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

void LaunchLoadBlocks(Scene* scene, std::string worldToRender) {
    // Ask user for the world name to load (for the filename)
    worldToRender += ".wrld";
    std::ifstream readWorldFile(worldToRender);

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
        std::cout << "Loading world file: " << worldToRender << ", this can take a few minutes" << std::endl;
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
    else std::cout << "Unable to open file " << worldToRender << std::endl;

    // Close file and free the device pointers
    readWorldFile.close();

    std::cout << "Done" << std::endl;
}

void PreAddBlock(Point loc, Scene* scene, int type) {
		if(type == 0) {
			Box*				boxShape	= new Box(loc);
			LambertMaterial*	boxMat		= new LambertMaterial(Color(1.f, 0.f, 0.f), 1.f);
			Primitive*			boxPrim		= new Primitive(boxShape, boxMat);
			scene->AddObject(boxPrim);
		}
		if(type == 1) {
			Box*				boxShape	= new Box(loc);
			AreaLight*			boxLight	= new AreaLight(boxShape,Color(1.f,1.f,1.f),10.f);
			scene->AddObject(boxLight);
		}
		if(type == 2) {
			Box*				boxShape	= new Box(loc);
			MirrorMaterial*		boxMat		= new MirrorMaterial(Color(1.f, 1.f, 1.f), .9f);
			Primitive*			boxPrim		= new Primitive(boxShape, boxMat);
			scene->AddObject(boxPrim);
		}
}

void LaunchInitScene(Scene* scene) {
    Plane*				planeShape		= new Plane(Point(0,0,0), Vector(0.f, 1.f, 0.f));
    LambertMaterial*	planeMat		= new LambertMaterial(Color(1.f, 1.f, 1.f), .9f);
    Primitive*			plane			= new Primitive(planeShape, planeMat);
    scene->AddPlane(plane);
}

void LaunchAddBlock(const Camera* cam, Scene* scene, Builder* builder) {
    const Ray ray = cam->GetCenterRay();
    IntRec intRec;
    if (scene->Intersect(ray, intRec)) {
        Point* location;
        if (intRec.prim)
            location = builder->GetPosition(intRec.prim, ray(intRec.t));
        if (intRec.light)
            location = builder->GetPosition((AreaLight*)intRec.light, ray(intRec.t));

        Shape* shape = builder->GetShape(*location);
        Object* object = builder->GetObject(shape, location);
        //scene->AddPrimitive((Primitive*)object);
        scene->AddObject(object);
    }
}
void LaunchSetFocalPoint(Camera* cam, const Scene* scene) {
    const Ray ray = cam->GetCenterRay();
    IntRec intRec;
    if (scene->Intersect(ray, intRec)) {
        cam->fpoint = fmax(1.f,intRec.t - 1.f);
    }
}

void LaunchRemoveBlock(const Camera* cam, Scene* scene) {
    Ray ray = cam->GetCenterRay();
    IntRec intRec;
    if (scene->Intersect(ray, intRec)) {
        if(intRec.light) scene->RemoveObject(intRec.light);
        if(intRec.prim) scene->RemoveObject(intRec.prim);
    }
}

void LaunchSaveBlocks(Scene* scene) {
	// Ask user for a world name (for the filename)
	std::cout << "How would you like to name this world?" << std::endl;
	std::string worldName;
	std::getline(std::cin, worldName);
	worldName += ".wrld";
	if(checkOverwrite(worldName)) return;
	
	//std::ofstream writeWorldFile(worldName);

	std::stringstream contents;
	contents.clear();
	contents.str(std::string());

	std::cout << "Saving the world to file: " << worldName << ", this can take a few minutes" << std::endl;

	// Start saving process

	// Declare pointers and allocate device pointers
    int h_nBlocks;
    Node **d_nextNode;
    Point loc;
    Color col;
    float albedo;
    float intensity;
    MaterialType mat;
    ShapeType shape;
    ObjectType type;
	// Query the number of objects we have to save
    h_nBlocks = scene->GetNumberOfObjects();


	// Reset d_nextNode to root of the octree
    //InitSaveBlocks(scene, d_nextNode);
    Node *octree = (*scene).octree;
    d_nextNode = &octree;
	// Save each object into the world file
    for(int i = 0; i < h_nBlocks; i++) {
		// Fill the variables with the correct block data
        SaveBlock(scene, d_nextNode, loc, col, albedo, intensity, mat, shape, type);
		// Write the block data in the world file
        //if (writeWorldFil.is_open())
        {
        contents << "NewObject" << std::endl;
        contents << "Location: " << loc.x << " " << loc.y << " "  << loc.z << std::endl;
        contents << "Color: " << col.r << " " << col.g << " "  << col.b << std::endl;
        contents << "Albedo: " << albedo << std::endl;
        contents << "Intensity: " << intensity << std::endl;
        contents << "Material: " << mat << std::endl;
        contents << "Shape: " << shape << std::endl;
        contents << "ObjectType: " << type << std::endl;
        }
        //else std::cout << "Unable to open file " << worldName << std::endl;
	}
	
    writeWorldFile(worldName, contents);

	std::cout << "Done" << std::endl; 
}
void InitSaveBlocks(Scene* scene, Node* nextNode) {
    nextNode = scene->octree;
}

void SaveBlock(Scene* scene, Node **nextNode, Point &loc, Color &col, float &albedo, float &intensity, MaterialType &mat, ShapeType &shape, ObjectType &type) {
    // Point to the correct next leaf node
    *nextNode = scene->octree->NextLeaf(*nextNode);
    Object* obj = (*nextNode)->object;
    // Fill the variables with the block data
    ObjectType i = obj->GetType();
    type = i;

	if(obj->GetType() == OBJ_PRIMITIVE) {
        col = ((Primitive*) obj)->GetMaterial()->GetColor();
        albedo = ((Primitive*) obj)->GetMaterial()->GetAlbedo();
        mat = ((Primitive*) obj)->GetMaterial()->GetType();
        shape = ((Primitive*) obj)->GetShape()->GetType();
        loc = *((Primitive*) obj)->GetCornerPoint();
        intensity = -1.f;
	}    
	if(obj->GetType() == OBJ_LIGHT) {
        col = ((AreaLight*) obj)->c;
        intensity = ((AreaLight*) obj)->i;
        shape = ((AreaLight*) obj)->GetShape()->GetType();
        albedo = -1.f;
        mat = (MaterialType) 0;
        loc = *((Primitive*) obj)->GetCornerPoint();
	}
}

void LaunchEmptyScene(Scene* scene) {
    delete scene->octree;
    scene->octree = new Node(Point(0,0,0), Point(scene->size-1,scene->size-1,scene->size-1));
}

void LaunchLoadBlock(Scene* scene, Point loc, Color col, float albedo, float intensity, MaterialType mat, ShapeType shape, ObjectType type) {
    LoadBlock(scene, loc, col, albedo, intensity, mat, shape, type);
}

void LoadBlock(Scene* scene, Point loc, Color col, float albedo, float intensity, MaterialType mat, ShapeType shape, ObjectType type) {
	Object* newObject;
	if(type == OBJ_PRIMITIVE) {
		Shape* shapeObj;
		if(shape == ST_CUBE)
			shapeObj = new Box(loc);
		if(shape == ST_SPHERE)
			shapeObj = new Sphere(loc);
		Material* matObj;
		if(mat == MT_DIFFUSE)
			matObj = new LambertMaterial(col, albedo);
		if(mat == MT_MIRROR)
			matObj = new MirrorMaterial(col, albedo);
		newObject = new Primitive(shapeObj, matObj);
	}

	if(type == OBJ_LIGHT) {
		Shape* shapeObj;
		if(shape == ST_CUBE)
			shapeObj = new Box(loc);
		if(shape == ST_SPHERE)
			shapeObj = new Sphere(loc);
		newObject = new AreaLight(shapeObj,col,intensity);
	}
	scene->AddObject(newObject);
}

int LaunchCountObjects(Scene* scene) {
    return scene->GetNumberOfObjects();
}

void LaunchBuilderNextBuildType(Builder* builder) {
    builder->NextBuildType();
}

void LaunchBuilderNextShapeType(Builder* builder) {
    builder->NextShapeType();
}

void LaunchBuilderNextMaterialType(Builder* builder) {
    builder->NextMaterialType();
}

void LaunchBuilderSetPresetColor(Builder* builder, unsigned index)  {
    builder->SetPresetColor(index);
}

void LaunchBuilderIncrease(Builder* builder)  {
    builder->IncreaseAorI(0.2f);
}

void LaunchBuilderDecrease(Builder* builder)  {
    builder->DecreaseAorI(0.2f);
}

void LaunchIncreaseDayLight(Scene* scene) {
    scene->IncreaseDayLight(.1f);
}

void LaunchDecreaseDayLight(Scene* scene) {
    scene->DecreaseDayLight(.1f);
} 

void LaunchInitResult(Color* result, unsigned width, unsigned height) {
    for(int x=0; x<width; x++)
        for(int y=0; y<height; y++)
            result[y*width + x] = Color();
}

void LaunchTraceRays(const Camera* cam, const Scene* scene, Color* result, unsigned width, unsigned height) {
    for(unsigned x=0; x<width; x++)
        for(unsigned y=0; y<height; y++)
            TraceRays(x,y,cam, scene, result, width);
}

void TraceRays(const unsigned x, const unsigned y, const Camera* cam, const Scene* scene, Color* result, unsigned width) {
	const unsigned i = y * width + x;
	Color black = Color();
	Color env = scene->GetDayLight();
    Ray ray;
    if(cam->anti && cam->dof)
        ray = cam->GetAaDofRay(x, y);
    else if(cam-> anti && !cam->dof)
        ray = cam->GetAaRay(x,y);
    else if(!cam-> anti && cam->dof)
        ray = cam->GetDofRay(x,y);
    else if(!cam-> anti && !cam->dof)
        ray = cam->GetNormalRay(x,y);
	IntRec intRec;
	Color color(1.f, 1.f, 1.f);
	float rr = 1.f;
	for (unsigned depth = 0; depth <= MAX_DEPTH; depth++) {
		// Enforce maximum depth
		if (depth == MAX_DEPTH) {
			color = black;
			break;
		}        
		// If ray leaves the scene
		if (!scene->Intersect(ray, intRec)) {
			color *= env;
			break;
		}        
		// If the ray hits a light
		if (intRec.light) {
            color *= intRec.light->Le();
			break;
		}
		// If ray hit a surface in the scene
		const Material* mat = intRec.prim->GetMaterial();
		const Shape*	shape = intRec.prim->GetShape();
		const Point		p = ray(intRec.t);
		const Vector	n = shape->GetNormal(p);

		//const Vector in = ray.d;
        const Vector out = mat->GetSample(ray.d, n);

		const float multiplier = mat->GetMultiplier(ray.d, out, n);
		ray = Ray(p, out);

		color *= mat->GetColor();
		color *= multiplier;

		// Russian roulette
        if (((double) rand() / (RAND_MAX)) > rr) {
			color = black;
			break;
		}
		rr *= multiplier;
	}

	result[i] += color;
}

void LaunchConvert(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width, unsigned height) {
    for(unsigned x=0; x<width; x++)
        for(unsigned y=0; y<height; y++)
            Convert(x, y, result, pixelData, iteration, width);
}

void LaunchConvertRaw(const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width, unsigned height) {
    for(unsigned x=0; x<width; x++)
        for(unsigned y=0; y<height; y++)
            ConvertRaw(x, y, result, pixelData, iteration, width);
}

void Convert(unsigned x, unsigned y, const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width) {
	const unsigned i = y * width + x;
	const unsigned pdi = i*4;

    const float r = (result[i].r / (float) iteration) * 255.f;
    const float g = (result[i].g / (float)iteration) * 255.f;
    const float b = (result[i].b / (float)iteration) * 255.f;

	pixelData[pdi]		= Clamp255(r);
	pixelData[pdi+1]	= Clamp255(g);
	pixelData[pdi+2]	= Clamp255(b);
    pixelData[pdi+3]	= 255;
}

void ConvertRaw(unsigned x, unsigned y, const Color* result, unsigned char* pixelData, unsigned iteration, unsigned width) {
    const unsigned i = y * width + x;
    const unsigned pdi = i*3;

    const float r = (result[i].r / (float) iteration) * 255.f;
    const float g = (result[i].g / (float)iteration) * 255.f;
    const float b = (result[i].b / (float)iteration) * 255.f;

    pixelData[pdi]		= Clamp255(r);
    pixelData[pdi+1]	= Clamp255(g);
    pixelData[pdi+2]	= Clamp255(b);
}

void LaunchDestroyScene(Scene* scene) {
    delete scene;
    scene = NULL;
}

unsigned char Clamp255(float s) {
	if (s < 0.f) s = 0.f;
	if (s > 255.f) s = 255.f;
	return (unsigned char)s;
}
