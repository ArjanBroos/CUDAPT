CXX := g++
CC := gcc
LINK := g++
# Includes

# Common flags
COMMONFLAGS += $(INCLUDES)
CXXFLAGS += $(COMMONFLAGS) -std=c++11 -lpthread -g
CFLAGS += $(COMMONFLAGS) -g

OBJ_FILES = $(patsubst %.cpp,%.cpp.o,$(wildcard *.cpp))
TARGET = traceMe
LINKLINE = $(LINK) -o $(TARGET) $(OBJ_FILES)

.SUFFIXES: .c .cpp .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
%.cpp.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)
	
$(TARGET): compilingCpp $(OBJ_FILES) 
	@echo "\n\t***Linking object files***"
	$(LINKLINE)
	
compilingCpp:
	@echo "\n\t***Compiling cpp files***"
	
clean:
	@rm -rf *.o $(TARGET)