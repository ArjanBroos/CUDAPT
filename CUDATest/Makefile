CUDA_INSTALL_PATH ?= /usr/lib/nvidia-cuda-toolkit
CXX := g++
CC := gcc
LINK := nvcc
NVCC := nvcc
# Includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) -rdc=true -arch=compute_30 -code=sm_30 -lcuda -lcudart 
CXXFLAGS += $(COMMONFLAGS) -std=c++11
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib
OBJ_CU_FILES = $(patsubst %.cu,%.cu.o,$(wildcard *.cu))
OBJ_FILES = $(patsubst %.cpp,%.cpp.o,$(wildcard *.cpp))
TARGET = traceMe
LINKLINE = $(LINK) -o $(TARGET) $(OBJ_CU_FILES) $(OBJ_FILES) $(LIB_CUDA) $(NVCCFLAGS) -lsfml-graphics -lsfml-window -lsfml-system
.SUFFIXES: .c .cpp .cu .o
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(TARGET): compilingCuda $(OBJ_CU_FILES) compilingCpp $(OBJ_FILES) 
	@echo "\n\t***Linking object files***"
	$(LINKLINE)	   
compilingCuda:
	@echo "\t***Compiling cuda files***"
compilingCpp:
	@echo "\n\t***Compiling cpp files***"
clean:
	@rm -rf *.o $(TARGET)
