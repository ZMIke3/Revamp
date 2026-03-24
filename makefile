CC = gcc

TARGET = revamp.dll

SRC_DIR = src
INTERNAL_DIR = internal
PUBLIC_INC_DIR = include
BUILD_DIR = build

PREFIX ?= /usr/local

CFLAGS = -std=c11 -O2 -I$(INTERNAL_DIR) -I$(PUBLIC_INC_DIR)

LDFLAGS = -shared

# Optional BLAS support (user supplies this)
# Example:
#   make BLAS="-lopenblas"
#   make BLAS="-LC:/msys64/mingw64/lib -lopenblas"
BLAS ?=

SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(BLAS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp $(TARGET) $(PREFIX)/lib/
	cp $(PUBLIC_INC_DIR)/*.h $(PREFIX)/include/

uninstall:
	rm -f $(PREFIX)/lib/$(TARGET)
	rm -f $(PREFIX)/include/*.h

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all install uninstall clean