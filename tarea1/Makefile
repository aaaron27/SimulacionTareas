CXX = g++

SRC = ejercicio2.cpp ejercicio4.cpp ejercicio5.cpp ejercicio6.cpp ejercicio7.cpp ejercicio8.cpp ejercicio9.cpp
BIN = $(SRC:.cpp=)

all: $(BIN)

%: %.cpp
	$(CXX) $< -o $@

run: $(BIN)
	@for prog in $(BIN); do \
		echo "=== Ejecutando $$prog ==="; \
		./$$prog; \
		echo ""; \
	done

clean:
	rm -f $(BIN)
