// Records sent from the instrumented kernel to the host (one channel push each).
#pragma once
#include <stdint.h>

enum WpKind : uint8_t {
    WP_BASIC = 0,  // any instruction: just the path
    WP_MBAR = 1,   // mbarrier op: a = smem address, b = first R operand, c = first UR operand
    WP_WAIT = 2,   // mbarrier try_wait (after): a = smem address, b = predicate file, c = parity operand
    WP_MEM = 3,    // ld/st/atom/cp.async/ldsm: 32 lane addresses follow
    WP_TMA = 4,    // TMA: the raw param handle follows
    WP_UREG = 5,   // tcgen05 / misc: a, b, c = uniform register values
    WP_BAR = 6,    // named barrier whose id is in a register: b = barrier id
};

typedef struct {
    uint32_t cta;      // linear CTA id
    uint16_t warp;     // warp index within the CTA
    uint8_t kind;      // WpKind
    uint8_t aux;       // WP_WAIT: destination predicate number
    uint32_t idx;      // static instruction index within the function
    uint32_t active;   // active lanes
    uint32_t predon;   // active lanes whose guard predicate is true
    uint32_t pad;
    uint64_t a, b, c;
    uint64_t clk;      // clock64() of the issuing SM
} wp_rec_t;            // 56 bytes

typedef struct {
    wp_rec_t r;
    uint64_t addrs[32];
} wp_mem_t;

typedef struct {
    wp_rec_t r;
    uint32_t handle_size;
    uint32_t pad2;
    uint64_t handle_ptr;
    uint8_t handle[128];
} wp_tma_t;
