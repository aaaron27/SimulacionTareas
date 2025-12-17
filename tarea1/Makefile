CXX = g++

SRC = ejercicio2.cpp ejercicio4.cpp ejercicio5.cpp ejercicio6.cpp ejercicio7.cpp ejercicio8.cpp
BIN = $(SRC:.cpp=)

all: $(BIN)

%: %.cpp
	$(CXX) $< -o $@

clean:
	rm -f $(BIN)