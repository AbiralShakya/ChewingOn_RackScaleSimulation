/*
 * warp_path: record the exact per-warp dynamic SASS path of a few CTAs of one
 * kernel launch, plus what a timing simulator cannot recover from static SASS:
 *
 *   - lane addresses of every load/store/cp.async/ldmatrix,
 *   - the shared-memory address of every mbarrier operation (so the simulator
 *     can match arrivals to waits), the expect-tx byte count, the result and
 *     parity of every try_wait,
 *   - the source runs (address, length) and byte count of every TMA copy,
 *   - uniform-register operands of tcgen05 instructions (the instruction
 *     descriptor that gives the UMMA shape).
 *
 * It also writes the static listing of the function in cuobjdump format with
 * the 128-bit encodings, so the simulator's parser recovers the control bits
 * (stall counts, scoreboards) exactly as the hardware sees them.
 *
 * Environment:
 *   WP_KERNEL   substring of the (mangled) kernel name to trace (required)
 *   WP_LAUNCH   which matching launch to trace, 0-based (default 0)
 *   WP_CTAS     comma-separated linear CTA ids to record, up to 4 (default 0)
 *   WP_OUT      output directory (default .)
 *   WP_TMA_ALL  1 = also record the TMA copies (addresses only) of every CTA, so
 *               persistent kernels replay each CTA's real tile sequence
 *
 * Output per traced launch: <WP_OUT>/<tag>.sass, <tag>.json, <tag>.wp
 * (binary: wp_rec_t / wp_mem_t / TMA records, see common.h and the reader in
 * ../trace_to_rsim.py).
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/channel.hpp"
#include "common.h"
#include "wp_instrument.h"

#define CHANNEL_SIZE (1l << 22)

enum class RecvThreadState { INIT, WORKING, STOP, FINISHED };

struct CTXstate {
    int id;
    ChannelDev *channel_dev = nullptr;
    ChannelHost channel_host;
    CUmodule tool_module;
    CUfunction flush_channel_func;
    volatile RecvThreadState recv_thread_done = RecvThreadState::INIT;
    bool need_sync = false;
};

#include "tool_func/flush_channel.c"

std::recursive_mutex mutex;
std::recursive_mutex cuda_event_mutex;
std::unordered_map<CUcontext, CTXstate *> ctx_state_map;
bool skip_callback_flag = false;

static std::string wp_kernel;
static int wp_launch = 0;
static std::string wp_out = ".";
static uint64_t s01 = 0xffffffffffffffffull, s23 = 0xffffffffffffffffull;
static int matched_launches = 0;
static uint32_t wp_tma_all = 0;

/* the open output file of the launch being traced (written by the recv thread) */
static FILE *out_bin = nullptr;
static CUcontext out_ctx = nullptr;
static std::map<uint32_t, std::string> traced_opcode;  // instr idx -> opcode (traced function)
static uint64_t n_recs = 0;

std::unordered_set<CUfunction> already_instrumented;

void *recv_thread_fun(void *args);

static std::string env_str(const char *k, const char *d) {
    const char *v = getenv(k);
    return v ? std::string(v) : std::string(d);
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    wp_kernel = env_str("WP_KERNEL", "");
    wp_launch = atoi(env_str("WP_LAUNCH", "0").c_str());
    wp_out = env_str("WP_OUT", ".");
    wp_tma_all = atoi(env_str("WP_TMA_ALL", "0").c_str());
    std::vector<uint32_t> ctas;
    std::stringstream ss(env_str("WP_CTAS", "0"));
    std::string tok;
    while (std::getline(ss, tok, ',') && ctas.size() < 4) ctas.push_back((uint32_t)strtoul(tok.c_str(), 0, 0));
    while (ctas.size() < 4) ctas.push_back(0xffffffffu);
    s01 = (uint64_t)ctas[0] | ((uint64_t)ctas[1] << 32);
    s23 = (uint64_t)ctas[2] | ((uint64_t)ctas[3] << 32);
    fprintf(stderr, "WARP_PATH: kernel~'%s' launch %d ctas %u,%u,%u,%u -> %s\n", wp_kernel.c_str(), wp_launch,
            ctas[0], ctas[1], ctas[2], ctas[3], wp_out.c_str());
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    CTXstate *ctx_state = ctx_state_map[ctx];
    std::vector<CUfunction> fs = nvbit_get_related_functions(ctx, func);
    fs.push_back(func);
    for (auto f : fs) {
        if (!already_instrumented.insert(f).second) continue;
        if (f == func)
            for (auto ins : nvbit_get_instrs(ctx, f)) traced_opcode[ins->getIdx()] = ins->getOpcode();
        wp_instrument(ctx, f, s01, s23, wp_tma_all, ctx_state->channel_dev);
    }
}

static void flush_and_close(CUcontext ctx, CTXstate *ctx_state) {
    CUDA_SAFECALL(cudaDeviceSynchronize());
    void *args[] = {&ctx_state->channel_dev};
    nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
    CUDA_SAFECALL(cudaDeviceSynchronize());
    // the recv thread drains asynchronously; wait until it has seen the flush
    usleep(500000);
    mutex.lock();
    if (out_bin) {
        fclose(out_bin);
        out_bin = nullptr;
        fprintf(stderr, "WARP_PATH: wrote %lu records\n", (unsigned long)n_recs);
    }
    mutex.unlock();
}

static std::string sanitize(const char *n) {
    std::string s;
    for (const char *p = n; *p && s.size() < 80; p++) s += isalnum(*p) ? *p : '_';
    return s;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params,
                         CUresult *pStatus) {
    std::lock_guard<std::recursive_mutex> g(cuda_event_mutex);
    if (skip_callback_flag) return;
    if (ctx_state_map.find(ctx) == ctx_state_map.end()) return;
    skip_callback_flag = true;
    CTXstate *ctx_state = ctx_state_map[ctx];

    if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchCooperativeKernel || cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
        CUfunction func;
        unsigned gx, gy, gz, bx, by, bz, dyn_smem;
        unsigned cx = 1, cy = 1, cz = 1;
        if (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
            auto *p = (cuLaunchKernelEx_params *)params;
            func = p->f;
            gx = p->config->gridDimX; gy = p->config->gridDimY; gz = p->config->gridDimZ;
            bx = p->config->blockDimX; by = p->config->blockDimY; bz = p->config->blockDimZ;
            dyn_smem = p->config->sharedMemBytes;
            for (unsigned i = 0; i < p->config->numAttrs; i++) {
                if (p->config->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
                    cx = p->config->attrs[i].value.clusterDim.x;
                    cy = p->config->attrs[i].value.clusterDim.y;
                    cz = p->config->attrs[i].value.clusterDim.z;
                }
            }
        } else {
            auto *p = (cuLaunchKernel_params *)params;
            func = p->f;
            gx = p->gridDimX; gy = p->gridDimY; gz = p->gridDimZ;
            bx = p->blockDimX; by = p->blockDimY; bz = p->blockDimZ;
            dyn_smem = p->sharedMemBytes;
        }
        const char *fname = nvbit_get_func_name(ctx, func);
        bool match = !wp_kernel.empty() && strstr(fname, wp_kernel.c_str()) != nullptr;
        if (!is_exit && getenv("WP_LIST"))
            fprintf(stderr, "WARP_PATH: launch %s grid %u,%u,%u block %u,%u,%u\n", fname, gx, gy, gz, bx, by, bz);
        if (!is_exit) {
            bool trace = false;
            if (match) trace = (matched_launches++ == wp_launch);
            if (trace) {
                instrument_function_if_needed(ctx, func);
                int nregs = func_attr(func, CU_FUNC_ATTRIBUTE_NUM_REGS);
                int smem_static = func_attr(func, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
                std::string tag = sanitize(fname);
                write_listing(ctx, func, wp_out + "/" + tag + ".sass");
                std::string jpath = wp_out + "/" + tag + ".json";
                FILE *jf = fopen(jpath.c_str(), "w");
                if (jf) {
                    fprintf(jf,
                            "{\"kernel\": \"%s\", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], "
                            "\"cluster\": [%u, %u, %u], \"regs\": %d, \"smem_static\": %d, \"smem_dynamic\": %u, "
                            "\"ctas\": [%u, %u, %u, %u]}\n",
                            fname, gx, gy, gz, bx, by, bz, cx, cy, cz, nregs, smem_static, dyn_smem,
                            (uint32_t)s01, (uint32_t)(s01 >> 32), (uint32_t)s23, (uint32_t)(s23 >> 32));
                    fclose(jf);
                }
                mutex.lock();
                out_bin = fopen((wp_out + "/" + tag + ".wp").c_str(), "wb");
                out_ctx = ctx;
                n_recs = 0;
                mutex.unlock();
                ctx_state->need_sync = true;
                fprintf(stderr, "WARP_PATH: tracing %s grid %u,%u,%u block %u,%u,%u cluster %u,%u,%u\n", fname, gx,
                        gy, gz, bx, by, bz, cx, cy, cz);
                nvbit_enable_instrumented(ctx, func, true);
            } else {
                nvbit_enable_instrumented(ctx, func, false);
            }
        } else if (out_bin && match) {
            flush_and_close(ctx, ctx_state);
        }
    }
    skip_callback_flag = false;
}

void *recv_thread_fun(void *args) {
    CUcontext ctx = (CUcontext)args;
    mutex.lock();
    CTXstate *ctx_state = ctx_state_map[ctx];
    ChannelHost *ch_host = &ctx_state->channel_host;
    mutex.unlock();
    char *buf = (char *)malloc(CHANNEL_SIZE);
    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        uint32_t n = ch_host->recv(buf, CHANNEL_SIZE);
        uint32_t off = 0;
        while (off < n) {
            wp_rec_t *r = (wp_rec_t *)&buf[off];
            size_t sz = wp_record_size(r);
            mutex.lock();
            if (out_bin) {
                wp_write_record(ctx, out_bin, r, traced_opcode);
                n_recs++;
            }
            mutex.unlock();
            off += sz;
        }
    }
    free(buf);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void init_context_state(CUcontext ctx) {
    CTXstate *ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = RecvThreadState::WORKING;
    CUDA_SAFECALL(cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev)));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE, ctx_state->channel_dev,
                                 recv_thread_fun, ctx);
    nvbit_set_tool_thread(ctx_state->channel_host.get_thread_id());
}

void nvbit_at_ctx_init(CUcontext ctx) {
    mutex.lock();
    CTXstate *ctx_state = new CTXstate;
    ctx_state_map[ctx] = ctx_state;
    nvbit_load_tool_module(ctx, (const void *)flush_channel_bin, &ctx_state->tool_module);
    nvbit_find_function_by_name(ctx, ctx_state->tool_module, "flush_channel", &ctx_state->flush_channel_func);
    mutex.unlock();
}

void nvbit_tool_init(CUcontext ctx) {
    mutex.lock();
    init_context_state(ctx);
    mutex.unlock();
}

void nvbit_at_ctx_term(CUcontext ctx) {
    mutex.lock();
    skip_callback_flag = true;
    CTXstate *ctx_state = ctx_state_map[ctx];
    if (ctx_state->need_sync) {
        void *args[] = {&ctx_state->channel_dev};
        nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
        CUDA_SAFECALL(cudaDeviceSynchronize());
    }
    if (ctx_state->recv_thread_done != RecvThreadState::INIT) {
        ctx_state->recv_thread_done = RecvThreadState::STOP;
        while (ctx_state->recv_thread_done != RecvThreadState::FINISHED)
            ;
    }
    if (out_bin) {
        fclose(out_bin);
        out_bin = nullptr;
    }
    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    ctx_state_map.erase(ctx);
    mutex.unlock();
}
