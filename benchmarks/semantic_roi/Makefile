CC ?= gcc
GEM5_ROOT ?= ../../tools/src/gem5
CFLAGS ?= -O3 -g -std=c11 -Wall -Wextra
CPPFLAGS += -I$(GEM5_ROOT)/include

.PHONY: all clean

all: clip_roi.o m5op.o

clip_roi.o: clip_roi.c clip_roi.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

m5op.o: $(GEM5_ROOT)/util/m5/src/abi/x86/m5op.S
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) clip_roi.o m5op.o
