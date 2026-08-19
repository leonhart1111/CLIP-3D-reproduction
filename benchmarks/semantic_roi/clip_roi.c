#define _POSIX_C_SOURCE 200809L

#include "clip_roi.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gem5/m5ops.h>

#define CLIP_ROI_MODE "semantic-work-v1"

static uint64_t
required_u64(const char *name)
{
    const char *text = getenv(name);
    char *end = NULL;
    unsigned long long parsed;

    if (text == NULL || *text == '\0') {
        fprintf(stderr, "semantic ROI requires %s\n", name);
        exit(EXIT_FAILURE);
    }
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) {
        fprintf(stderr, "semantic ROI has invalid %s=%s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)parsed;
}

void
clip_roi_configure(clip_roi_state_t *state, uint64_t available_units,
                   const char *expected_work_unit_type)
{
    const char *mode;
    const char *unit_type;

    if (state == NULL || expected_work_unit_type == NULL) {
        fprintf(stderr, "semantic ROI received an invalid benchmark contract\n");
        exit(EXIT_FAILURE);
    }
    memset(state, 0, sizeof(*state));
    mode = getenv("CLIP_ROI_MODE");
    if (mode == NULL) {
        return;
    }
    if (strcmp(mode, CLIP_ROI_MODE) != 0) {
        fprintf(stderr, "unsupported CLIP_ROI_MODE=%s\n", mode);
        exit(EXIT_FAILURE);
    }
    unit_type = getenv("CLIP_ROI_WORK_UNIT_TYPE");
    if (unit_type == NULL || strcmp(unit_type, expected_work_unit_type) != 0) {
        fprintf(stderr,
                "semantic ROI work-unit mismatch: expected %s, observed %s\n",
                expected_work_unit_type,
                unit_type == NULL ? "<unset>" : unit_type);
        exit(EXIT_FAILURE);
    }

    state->warmup_units = required_u64("CLIP_ROI_WARMUP_UNITS");
    state->measure_units = required_u64("CLIP_ROI_MEASURE_UNITS");
    state->work_id = required_u64("CLIP_ROI_WORK_ID");
    if (UINT64_MAX - state->warmup_units < state->measure_units) {
        fprintf(stderr, "semantic ROI work-unit count overflows uint64_t\n");
        exit(EXIT_FAILURE);
    }
    state->end_boundary = state->warmup_units + state->measure_units;
    if (available_units <= state->end_boundary) {
        fprintf(stderr,
                "semantic ROI needs a sentinel unit after boundary %" PRIu64
                "; benchmark exposes only %" PRIu64 " units\n",
                state->end_boundary, available_units);
        exit(EXIT_FAILURE);
    }
    state->enabled = 1;
    fprintf(stderr,
            "CLIP semantic ROI: unit=%s warmup=%" PRIu64
            " measure=%" PRIu64 " work_id=%" PRIu64 "\n",
            expected_work_unit_type, state->warmup_units,
            state->measure_units, state->work_id);
}

void
clip_roi_boundary(const clip_roi_state_t *state, uint64_t next_unit)
{
    if (state == NULL || !state->enabled) {
        return;
    }
    if (next_unit == state->warmup_units) {
        m5_work_begin(state->work_id, 0);
    } else if (next_unit == state->end_boundary) {
        m5_work_end(state->work_id, 0);
    }
}
