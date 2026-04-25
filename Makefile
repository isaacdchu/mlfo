CC=g++-15
FLAGS=-std=c++23 -Wall -Wextra -Wno-unused-parameter -Werror -O2
BUILD_DIR=build

compile: $(BUILD_DIR)/main

$(BUILD_DIR)/main: $(BUILD_DIR)/main.o
	$(CC) $(FLAGS) $(BUILD_DIR)/main.o -o $(BUILD_DIR)/main

$(BUILD_DIR)/main.o: main.cpp
	$(CC) $(FLAGS) -c main.cpp -o $(BUILD_DIR)/main.o

run: compile
	@clear
	./$(BUILD_DIR)/main

clean:
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

.PHONY: compile run clean
