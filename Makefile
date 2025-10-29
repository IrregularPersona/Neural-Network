CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
MEMCHECK = -fsanitize=address
TARGET = neural_network_test
SRCDIR = .
OBJDIR = build

# Source files
SOURCES = $(wildcard Utils/*.cpp Models/**/*.cpp)
HEADERS = $(wildcard Utils/*.hpp Models/**/*.hpp)
MAIN = main.cpp

# Since we're using header-only library, we just compile main.cpp
# The headers contain all the implementation

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(MAIN) $(HEADERS)
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(MEMCHECK) -o $(TARGET) $(MAIN)
	@echo "Build complete: $(TARGET)"

run: $(TARGET)
	./$(TARGET)

test: run

clean:
	rm -rf $(OBJDIR) $(TARGET)

print-%:
	@echo $($*)

