CC ?= gcc
CFLAGS ?= -O3 -g -std=c11 -Wall -Wextra -pthread
LDLIBS ?= -pthread -lm

.PHONY: all clean

all: matmul

matmul: matmul.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

clean:
	$(RM) matmul
