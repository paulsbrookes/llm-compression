CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall
SRC := src
BIN := bin
TOOLS := pmf huffman range rans weights

all: $(addprefix $(BIN)/,$(TOOLS))

$(BIN)/%: $(SRC)/%.cpp $(SRC)/half.h $(SRC)/pmf.h $(SRC)/model.h | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)

.PHONY: all clean
