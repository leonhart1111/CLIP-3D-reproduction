CC ?= gcc
CFLAGS ?= -O3 -g -std=c11 -Wall -Wextra -pthread
LDLIBS ?= -pthread -lm
GEM5_ROOT ?= ../../../tools/src/gem5
ROI_DIR := ../../semantic_roi
ROI_OBJECTS := $(ROI_DIR)/clip_roi.o $(ROI_DIR)/m5op.o

.PHONY: all clean

all: stencil

stencil: stencil.c $(ROI_OBJECTS)
	$(CC) $(CFLAGS) $< $(ROI_OBJECTS) -o $@ $(LDLIBS)

$(ROI_OBJECTS):
	$(MAKE) -C $(ROI_DIR) GEM5_ROOT=$(abspath $(GEM5_ROOT))

clean:
	$(RM) stencil
	$(MAKE) -C $(ROI_DIR) clean
