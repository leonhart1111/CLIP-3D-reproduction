#ifndef CLIP_SEMANTIC_ROI_H
#define CLIP_SEMANTIC_ROI_H

#include <stdint.h>

typedef struct {
    uint64_t warmup_units;
    uint64_t measure_units;
    uint64_t end_boundary;
    uint64_t work_id;
    int enabled;
} clip_roi_state_t;

/*
 * Configure a benchmark's semantic ROI from CLIP_ROI_* environment variables.
 * available_units must include one sentinel unit after the measurement end
 * boundary. Without CLIP_ROI_MODE the benchmark keeps its native behaviour.
 */
void clip_roi_configure(clip_roi_state_t *state, uint64_t available_units,
                        const char *expected_work_unit_type);

/* Called by exactly one designated thread while every worker is synchronized. */
void clip_roi_boundary(const clip_roi_state_t *state, uint64_t next_unit);

#endif
