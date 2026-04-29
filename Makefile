CC=g++-15
FLAGS=-std=c++23 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Werror -O1
BUILD_DIR=build
TARGETS=*.cpp *.hpp

$(BUILD_DIR)/main: $(TARGETS)
	$(CC) $(FLAGS) main.cpp -o $(BUILD_DIR)/main


# find all test .cpp files recursively under tests/
TEST_SRCS=$(shell find tests -name '*.cpp')

$(BUILD_DIR)/tests: $(TARGETS) $(TEST_SRCS)
	$(CC) $(FLAGS) $(TEST_SRCS) -o $(BUILD_DIR)/tests

test: $(BUILD_DIR)/tests
	./$(BUILD_DIR)/tests

run: $(BUILD_DIR)/main
	@clear
	./$(BUILD_DIR)/main

clean:
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

.PHONY: compile run clean test
