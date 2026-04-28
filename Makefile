CC=g++-15
FLAGS=-std=c++23 -Wall -Wextra -Wno-unused-parameter -Werror -O1
BUILD_DIR=build
TARGETS=*.cpp *.hpp

$(BUILD_DIR)/main: $(TARGETS)
	$(CC) $(FLAGS) main.cpp -o $(BUILD_DIR)/main

run: $(BUILD_DIR)/main
	@clear
	./$(BUILD_DIR)/main

clean:
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

.PHONY: compile run clean
