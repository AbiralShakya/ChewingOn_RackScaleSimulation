/*
 * rsim_capture: record what a CUDA program runs, for replay in rsim.
 *
 * The tool is injected into an unmodified program (CUDA_INJECTION64_PATH) and
 * works for any code that launches kernels through the driver: PyTorch, vLLM,
 * SGLang, Triton, CuTe DSL, CUTLASS, ThunderKittens, plain CUDA. Two modes:
 *
 *   graph  Every kernel launch (function, grid, block, cluster, shared memory,
 *          registers, stream), event record, stream wait, memcpy, memset and
 *          synchronisation, in host order, with the host time of each call and
 *          the GPU start/end of each kernel and copy (CUDA events recorded on
 *          the launch stream around it). No kernel is instrumented, so the
 *          GPU timeline is native apart from the events.
 *   paths  The per-warp dynamic SASS path (warp_path format, see
 *          wp_instrument.h) of the first in-region launch of every distinct
 *          kernel configuration, for cycle-level replay. A pass records each
 *          function at most once (see on_launch); `rackscale capture` runs
 *          paths passes until every configuration is recorded, passing the
 *          keys already done in RSIM_CAP_DONE.
 *
 * Both modes write <RSIM_CAP_OUT>/p<pid>/events.jsonl. Paths mode also writes
 * one directory per recording, k<NNNN>/, holding <tag>.sass, .json and .wp.
 * `rackscale capture` runs the program once per mode and joins the two by the
 * kernel key (function + launch configuration).
 *
 * The capture region: with RSIM_CAP_REGION=all everything is recorded;
 * otherwise only what runs between the region_begin and region_end marks that
 * the Python side sends through rsim_cap_mark() (python/rackscale/_capture_site).
 *
 * Environment:
 *   RSIM_CAP_OUT        output directory (default .)
 *   RSIM_CAP_MODE       graph (default) | paths
 *   RSIM_CAP_REGION     all | anything else = wait for region marks
 *   RSIM_CAP_KEY        config (default) | func: what makes two launches the same kernel
 *   RSIM_CAP_MAX_PATHS  distinct kernels to record (default 1024)
 *   RSIM_CAP_MAX_MB     stop writing a recording beyond this size (default 2048)
 *   RSIM_CAP_FILTER     only record kernels whose mangled name contains this
 *   RSIM_CAP_TMA_ALL    also record every CTA's TMA sources (default 1)
 *   RSIM_CAP_DONE       file of keys recorded by earlier paths passes
 *   RSIM_CAP_CTAS       "a,b": record CTAs a and b instead of 0 and 1
 *
 * events.jsonl, one JSON object per line ("e" is the record type):
 *   D  device: name, cc, sms, clock_khz, pid, mode, region (all | marked)
 *   F  function: id, mangled m, demangled n, regs r, static smem ss
 *   L  launch: seq i, function f, stream s, grid g, block b, cluster c,
 *      dynamic smem sm, host time h (ns from region begin)
 *   C  copy/memset: seq i, kind k, bytes n, stream s, h
 *   T  GPU time of seq i: start a, end b (ns from the first in-region event)
 *   R  event record: event ev, stream s, h      W  stream wait: s, ev, h
 *   S  host synchronisation: kind k, h          G  CUDA graph launch (not expanded), h
 *   M  mark: kind k, payload p, next seq i, h
 *   P  path recorded: seq i, function f, key, dir, bytes, truncated
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <map>
#include <mutex>
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
    ChannelDev *channel_dev = nullptr;
    ChannelHost channel_host;
    CUmodule tool_module;
    CUfunction flush_channel_func;
    volatile RecvThreadState recv_thread_done = RecvThreadState::INIT;
    bool need_sync = false;
};

#include "tool_func/flush_channel_rc.c"

static std::recursive_mutex mu;
static std::mutex out_mu;  // the open recording, shared with the recv thread
static std::atomic<uint64_t> recv_empty{0};  // recv() calls that found the channel empty
static std::unordered_map<CUcontext, CTXstate *> ctx_state_map;
static thread_local bool in_tool = false;  // our own CUDA calls do not reach the log

/* configuration */
static std::string out_dir = ".";
static bool paths_mode = false;
static bool region_all = false;
static bool key_func = false;
static int max_paths = 1024;
static uint64_t max_bytes = 2048ull << 20;
static std::string filter;
static uint32_t tma_all = 1;

/* state */
static FILE *ev_log = nullptr;
static bool region = false;
static uint64_t seq = 0;
static int64_t t0_host = 0;
static CUcontext last_ctx = nullptr;
static std::unordered_map<CUfunction, int> fid;

/* graph mode: event pairs around launches and copies, drained as they complete */
struct Timed {
    uint64_t seq;
    CUevent a, b;
};
static CUevent ref_ev = nullptr;
static bool ref_recorded = false;
static std::vector<Timed> pending;
static std::vector<CUevent> pool;
static thread_local Timed cur_timed = {0, nullptr, nullptr};

/* paths mode */
static std::unordered_set<std::string> recorded;      // keys recorded by this or an earlier pass
static std::unordered_set<CUfunction> recorded_funcs;  // functions recorded by this pass
static std::unordered_set<CUfunction> instrumented;
static int n_paths = 0;
static std::unordered_map<CUfunction, std::vector<uint32_t>> sampled;  // CTAs baked into each function

static FILE *out_bin = nullptr;
static uint64_t out_bytes = 0;
static bool out_truncated = false;
static std::map<uint32_t, std::string> rec_opcode;
static thread_local bool cur_recording = false;
static std::string cur_dir, cur_key;
static uint64_t cur_seq = 0;
static int cur_fid = 0;

static int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static int64_t host_t() { return now_ns() - t0_host; }

static std::string env_str(const char *k, const char *d) {
    const char *v = getenv(k);
    return v ? std::string(v) : std::string(d);
}

static void mkdirs(const std::string &p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) mkdir(cur.c_str(), 0755);
    }
}

/* JSON string body (without quotes). */
static std::string jesc(const char *s) {
    std::string o;
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        if (*p == '"' || *p == '\\') {
            o += '\\';
            o += (char)*p;
        } else if (*p == '\t') {
            o += "\\t";
        } else if (*p < 0x20) {
            char b[8];
            snprintf(b, sizeof b, "\\u%04x", *p);
            o += b;
        } else {
            o += (char)*p;
        }
    }
    return o;
}

static std::string hex(const void *p) {
    char b[32];
    snprintf(b, sizeof b, "\"0x%lx\"", (unsigned long)(uintptr_t)p);
    return b;
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    out_dir = env_str("RSIM_CAP_OUT", ".") + "/p" + std::to_string(getpid());
    paths_mode = env_str("RSIM_CAP_MODE", "graph") == "paths";
    region_all = env_str("RSIM_CAP_REGION", "marked") == "all";
    key_func = env_str("RSIM_CAP_KEY", "config") == "func";
    max_paths = atoi(env_str("RSIM_CAP_MAX_PATHS", "1024").c_str());
    max_bytes = (uint64_t)atoll(env_str("RSIM_CAP_MAX_MB", "2048").c_str()) << 20;
    filter = env_str("RSIM_CAP_FILTER", "");
    tma_all = atoi(env_str("RSIM_CAP_TMA_ALL", "1").c_str());
    // Keys recorded by earlier paths passes (one per line).
    if (FILE *df = fopen(env_str("RSIM_CAP_DONE", "").c_str(), "r")) {
        char line[8192];
        while (fgets(line, sizeof line, df)) {
            size_t n = strlen(line);
            while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
            if (n) recorded.insert(line);
        }
        fclose(df);
    }
    mkdirs(out_dir);
    ev_log = fopen((out_dir + "/events.jsonl").c_str(), "w");
    if (ev_log) setvbuf(ev_log, nullptr, _IOFBF, 1 << 20);
    region = region_all;
    t0_host = now_ns();
    fprintf(stderr, "RSIM_CAPTURE: %s mode, region %s -> %s\n", paths_mode ? "paths" : "graph",
            region_all ? "all" : "marked", out_dir.c_str());
}

/* ---------------------------------------------------------------- timing */

static CUevent get_event() {
    if (!pool.empty()) {
        CUevent e = pool.back();
        pool.pop_back();
        return e;
    }
    CUevent e = nullptr;
    cuEventCreate(&e, CU_EVENT_DEFAULT);
    return e;
}

static void drain(bool all) {
    if (!ref_recorded) return;
    size_t keep = 0;
    for (size_t j = 0; j < pending.size(); j++) {
        Timed &t = pending[j];
        if (!all && cuEventQuery(t.b) != CUDA_SUCCESS) {
            pending[keep++] = t;
            continue;
        }
        float ma = 0, mb = 0;
        if (cuEventElapsedTime(&ma, ref_ev, t.a) == CUDA_SUCCESS &&
            cuEventElapsedTime(&mb, ref_ev, t.b) == CUDA_SUCCESS && ev_log)
            fprintf(ev_log, "{\"e\":\"T\",\"i\":%lu,\"a\":%.0f,\"b\":%.0f}\n", (unsigned long)t.seq, ma * 1e6,
                    mb * 1e6);
        pool.push_back(t.a);
        pool.push_back(t.b);
    }
    pending.resize(keep);
}

/* Start event of an op about to be issued on `stream` (graph mode, in region). */
static void time_begin(uint64_t s, CUstream stream) {
    if (!ref_recorded) {
        if (!ref_ev) cuEventCreate(&ref_ev, CU_EVENT_DEFAULT);
        cuEventRecord(ref_ev, stream);
        ref_recorded = true;
    }
    cur_timed = {s, get_event(), get_event()};
    cuEventRecord(cur_timed.a, stream);
}

static void time_end(CUstream stream) {
    if (!cur_timed.a) return;
    cuEventRecord(cur_timed.b, stream);
    pending.push_back(cur_timed);
    cur_timed = {0, nullptr, nullptr};
    if (pending.size() >= 4096) drain(false);
}

/* -------------------------------------------------------------- marks API */

static void end_region() {
    if (!paths_mode && ref_recorded) {
        cuCtxSynchronize();
        drain(true);
    }
    if (ev_log) fflush(ev_log);
}

/* Called from Python (ctypes) by the capture site package. */
extern "C" __attribute__((visibility("default"))) void rsim_cap_mark(const char *kind, const char *payload) {
    std::lock_guard<std::recursive_mutex> g(mu);
    bool begin = strcmp(kind, "region_begin") == 0, end = strcmp(kind, "region_end") == 0;
    if (begin && !region) {
        region = true;
        t0_host = now_ns();
    }
    if (!region || !ev_log) return;
    fprintf(ev_log, "{\"e\":\"M\",\"k\":\"%s\",\"p\":\"%s\",\"i\":%lu,\"h\":%ld}\n", jesc(kind).c_str(),
            jesc(payload).c_str(), (unsigned long)seq, (long)host_t());
    if (end) {
        in_tool = true;
        CUcontext cur = nullptr;
        cuCtxGetCurrent(&cur);
        bool pushed = false;
        if (!cur && last_ctx) pushed = cuCtxPushCurrent(last_ctx) == CUDA_SUCCESS;
        if (cur || pushed) end_region();
        if (pushed) cuCtxPopCurrent(&cur);
        in_tool = false;
        region = false;
    }
}

/* ------------------------------------------------------------ path record */

static std::string sanitize(const char *n) {
    std::string s;
    for (const char *p = n; *p && s.size() < 80; p++) s += isalnum(*p) ? *p : '_';
    return s;
}

static int function_id(CUcontext ctx, CUfunction f) {
    auto it = fid.find(f);
    if (it != fid.end()) return it->second;
    int id = (int)fid.size();
    fid[f] = id;
    int regs = func_attr(f, CU_FUNC_ATTRIBUTE_NUM_REGS), ss = func_attr(f, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    if (ev_log)
        fprintf(ev_log, "{\"e\":\"F\",\"id\":%d,\"m\":\"%s\",\"n\":\"%s\",\"r\":%d,\"ss\":%d}\n", id,
                jesc(nvbit_get_func_name(ctx, f, true)).c_str(), jesc(nvbit_get_func_name(ctx, f, false)).c_str(),
                regs, ss);
    return id;
}

static void drain_channel(CUcontext ctx);

/* Begin recording this launch of `func` into a new k<NNNN>/ directory. */
static void start_recording(CUcontext ctx, CUfunction func, CUstream stream, const unsigned g[3],
                            const unsigned b[3], const unsigned c[3], unsigned dyn_smem) {
    CTXstate *cs = ctx_state_map[ctx];
    if (!instrumented.count(func)) {
        // Sampled CTAs. In a cluster, every CTA of the first cluster (up to 4): the
        // ranks take different paths (the leader of a 2-CTA pair issues the MMAs) and
        // unrecorded CTAs replay the recorded CTA of their rank. Otherwise two
        // neighbours along x plus the first CTA one step along y and z, so the
        // converter can infer per-dimension address strides.
        uint32_t ctas[4] = {0, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        uint32_t total = g[0] * g[1] * g[2], csize = c[0] * c[1] * c[2];
        if (csize > 1) {
            // cluster rank r of the first cluster is CTA (r % cx) + (r / cx) * gx along x/y
            for (uint32_t r = 1; r < csize && r < 4; r++) ctas[r] = (r % c[0]) + ((r / c[0]) % c[1]) * g[0];
        } else {
            if (total > 1) ctas[1] = g[0] > 1 ? 1 : g[0];
            if (g[1] > 1 && g[0] > 1) ctas[2] = g[0];
            if (g[2] > 1) ctas[3] = g[0] * g[1];
        }
        const char *ov = getenv("RSIM_CAP_CTAS");  // "a,b": record these two CTAs instead
        if (ov) sscanf(ov, "%u,%u", &ctas[0], &ctas[1]);
        uint64_t s01 = (uint64_t)ctas[0] | ((uint64_t)ctas[1] << 32);
        uint64_t s23 = (uint64_t)ctas[2] | ((uint64_t)ctas[3] << 32);
        sampled[func] = {ctas[0], ctas[1], ctas[2], ctas[3]};
        std::vector<CUfunction> fs = nvbit_get_related_functions(ctx, func);
        fs.push_back(func);
        for (auto f : fs)
            if (instrumented.insert(f).second) wp_instrument(ctx, f, s01, s23, tma_all, cs->channel_dev);
        instrumented.insert(func);
    }
    rec_opcode.clear();
    for (auto ins : nvbit_get_instrs(ctx, func)) rec_opcode[ins->getIdx()] = ins->getOpcode();
    char d[16];
    snprintf(d, sizeof d, "k%04d", n_paths);
    cur_dir = d;
    std::string dir = out_dir + "/" + cur_dir;
    mkdirs(dir);
    const char *fname = nvbit_get_func_name(ctx, func);
    std::string tag = sanitize(fname);
    write_listing(ctx, func, dir + "/" + tag + ".sass");
    int nregs = func_attr(func, CU_FUNC_ATTRIBUTE_NUM_REGS);
    int smem_static = func_attr(func, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    const std::vector<uint32_t> &sc = sampled[func];
    FILE *jf = fopen((dir + "/" + tag + ".json").c_str(), "w");
    if (jf) {
        fprintf(jf,
                "{\"kernel\": \"%s\", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], \"cluster\": [%u, %u, %u], "
                "\"regs\": %d, \"smem_static\": %d, \"smem_dynamic\": %u, \"ctas\": [%u, %u, %u, %u]}\n",
                jesc(fname).c_str(), g[0], g[1], g[2], b[0], b[1], b[2], c[0], c[1], c[2], nregs, smem_static,
                dyn_smem, sc[0], sc[1], sc[2], sc[3]);
        fclose(jf);
    }
    drain_channel(ctx);  // drop anything left in the channel by earlier launches
    {
        std::lock_guard<std::mutex> o(out_mu);
        out_bin = fopen((dir + "/" + tag + ".wp").c_str(), "wb");
        out_bytes = 0;
        out_truncated = false;
    }
    n_paths++;
    cs->need_sync = true;
    nvbit_enable_instrumented(ctx, func, true);
}

/* Flush the device channel and wait until the recv thread has consumed it. */
static void drain_channel(CUcontext ctx) {
    CTXstate *cs = ctx_state_map[ctx];
    if (!cs->channel_dev) return;
    cuCtxSynchronize();
    void *args[] = {&cs->channel_dev};
    nvbit_launch_kernel(ctx, cs->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
    cuCtxSynchronize();
    // The recv thread drains asynchronously and can lag far behind on big
    // recordings: wait until it has found the channel empty twice since the flush.
    uint64_t e0 = recv_empty.load();
    while (recv_empty.load() < e0 + 2) usleep(1000);
}

static void flush_and_close(CUcontext ctx) {
    drain_channel(ctx);
    {
        std::lock_guard<std::mutex> o(out_mu);
        if (out_bin) {
            fclose(out_bin);
            out_bin = nullptr;
        }
    }
    if (ev_log)
        fprintf(ev_log, "{\"e\":\"P\",\"i\":%lu,\"f\":%d,\"key\":\"%s\",\"dir\":\"%s\",\"bytes\":%lu,\"truncated\":%s}\n",
                (unsigned long)cur_seq, cur_fid, jesc(cur_key.c_str()).c_str(), cur_dir.c_str(),
                (unsigned long)out_bytes, out_truncated ? "true" : "false");
}

/* ---------------------------------------------------------- API callbacks */

static bool is_launch(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel ||
           cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
           cbid == API_CUDA_cuLaunchCooperativeKernel || cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz;
}

static bool is_ptsz(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
           cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz || cbid == API_CUDA_cuMemcpyAsync_ptsz ||
           cbid == API_CUDA_cuMemcpyHtoDAsync_v2_ptsz || cbid == API_CUDA_cuMemcpyDtoHAsync_v2_ptsz ||
           cbid == API_CUDA_cuMemcpyDtoDAsync_v2_ptsz || cbid == API_CUDA_cuMemsetD8Async_ptsz ||
           cbid == API_CUDA_cuMemsetD32Async_ptsz || cbid == API_CUDA_cuEventRecord_ptsz ||
           cbid == API_CUDA_cuEventRecordWithFlags_ptsz || cbid == API_CUDA_cuStreamWaitEvent_ptsz;
}

/* The stream a call really uses: stream 0 of a per-thread-default-stream API
 * is the per-thread stream, not the legacy one. */
static CUstream real_stream(nvbit_api_cuda_t cbid, CUstream s) {
    return (s == nullptr && is_ptsz(cbid)) ? CU_STREAM_PER_THREAD : s;
}

static const char *mem_kind(CUdeviceptr p) {
    unsigned int t = 0;
    if (cuPointerGetAttribute(&t, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, p) != CUDA_SUCCESS) return "H";
    return t == CU_MEMORYTYPE_DEVICE ? "D" : "H";
}

static void on_launch(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, void *params) {
    CUfunction func;
    CUstream stream;
    unsigned g[3], b[3], c[3] = {1, 1, 1}, dyn_smem;
    if (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        auto *p = (cuLaunchKernelEx_params *)params;
        func = p->f;
        stream = p->config->hStream;
        g[0] = p->config->gridDimX, g[1] = p->config->gridDimY, g[2] = p->config->gridDimZ;
        b[0] = p->config->blockDimX, b[1] = p->config->blockDimY, b[2] = p->config->blockDimZ;
        dyn_smem = p->config->sharedMemBytes;
        for (unsigned i = 0; i < p->config->numAttrs; i++)
            if (p->config->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
                c[0] = p->config->attrs[i].value.clusterDim.x;
                c[1] = p->config->attrs[i].value.clusterDim.y;
                c[2] = p->config->attrs[i].value.clusterDim.z;
            }
    } else if (cbid == API_CUDA_cuLaunchCooperativeKernel || cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
        auto *p = (cuLaunchCooperativeKernel_params *)params;
        func = p->f;
        stream = p->hStream;
        g[0] = p->gridDimX, g[1] = p->gridDimY, g[2] = p->gridDimZ;
        b[0] = p->blockDimX, b[1] = p->blockDimY, b[2] = p->blockDimZ;
        dyn_smem = p->sharedMemBytes;
    } else {
        auto *p = (cuLaunchKernel_params *)params;
        func = p->f;
        stream = p->hStream;
        g[0] = p->gridDimX, g[1] = p->gridDimY, g[2] = p->gridDimZ;
        b[0] = p->blockDimX, b[1] = p->blockDimY, b[2] = p->blockDimZ;
        dyn_smem = p->sharedMemBytes;
    }
    stream = real_stream(cbid, stream);
    if (is_exit) {
        if (paths_mode) {
            if (cur_recording) {
                flush_and_close(ctx);
                cur_recording = false;
                // Disabled for good: NVBit 1.8 runs a function uninstrumented if it is
                // enabled again after a disable, so each pass records a function once.
                nvbit_enable_instrumented(ctx, func, false);
            }
        } else {
            time_end(stream);
        }
        return;
    }
    if (!region) {
        return;
    }
    uint64_t i = seq++;
    int f = function_id(ctx, func);
    if (ev_log)
        fprintf(ev_log,
                "{\"e\":\"L\",\"i\":%lu,\"f\":%d,\"s\":%s,\"g\":[%u,%u,%u],\"b\":[%u,%u,%u],\"c\":[%u,%u,%u],"
                "\"sm\":%u,\"h\":%ld}\n",
                (unsigned long)i, f, hex(stream).c_str(), g[0], g[1], g[2], b[0], b[1], b[2], c[0], c[1], c[2],
                dyn_smem, (long)host_t());
    if (!paths_mode) {
        time_begin(i, stream);
        return;
    }
    const char *mangled = nvbit_get_func_name(ctx, func, true);
    std::string key = mangled;
    if (!key_func) {
        char cfg[160];
        snprintf(cfg, sizeof cfg, "|%u,%u,%u|%u,%u,%u|%u,%u,%u|%u", g[0], g[1], g[2], b[0], b[1], b[2], c[0], c[1],
                 c[2], dyn_smem);
        key += cfg;
    }
    bool rec = n_paths < max_paths && !recorded.count(key) && !recorded_funcs.count(func) &&
               (filter.empty() || strstr(mangled, filter.c_str()) != nullptr);
    if (rec) {
        recorded.insert(key);
        recorded_funcs.insert(func);
        cur_key = key;
        cur_seq = i;
        cur_fid = f;
        start_recording(ctx, func, stream, g, b, c, dyn_smem);
        cur_recording = true;
        fprintf(stderr, "RSIM_CAPTURE: recording %s grid %u,%u,%u block %u,%u,%u\n", nvbit_get_func_name(ctx, func),
                g[0], g[1], g[2], b[0], b[1], b[2]);
    }
}

static void on_copy(nvbit_api_cuda_t cbid, int is_exit, void *params) {
    const char *kind = nullptr;
    size_t n = 0;
    CUstream s = nullptr;
    switch (cbid) {
        case API_CUDA_cuMemcpyAsync:
        case API_CUDA_cuMemcpyAsync_ptsz: {
            auto *p = (cuMemcpyAsync_params *)params;
            n = p->ByteCount, s = p->hStream;
            kind = nullptr;
            if (!is_exit) {
                static thread_local char k[8];
                snprintf(k, sizeof k, "%sto%s", mem_kind(p->src), mem_kind(p->dst));
                kind = k;
            }
            break;
        }
        case API_CUDA_cuMemcpy: {
            auto *p = (cuMemcpy_params *)params;
            n = p->ByteCount;
            static thread_local char k[8];
            if (!is_exit) snprintf(k, sizeof k, "%sto%s", mem_kind(p->src), mem_kind(p->dst));
            kind = k;
            break;
        }
        case API_CUDA_cuMemcpyHtoDAsync_v2:
        case API_CUDA_cuMemcpyHtoDAsync_v2_ptsz:
            n = ((cuMemcpyHtoDAsync_v2_params *)params)->ByteCount, s = ((cuMemcpyHtoDAsync_v2_params *)params)->hStream;
            kind = "HtoD";
            break;
        case API_CUDA_cuMemcpyDtoHAsync_v2:
        case API_CUDA_cuMemcpyDtoHAsync_v2_ptsz:
            n = ((cuMemcpyDtoHAsync_v2_params *)params)->ByteCount, s = ((cuMemcpyDtoHAsync_v2_params *)params)->hStream;
            kind = "DtoH";
            break;
        case API_CUDA_cuMemcpyDtoDAsync_v2:
        case API_CUDA_cuMemcpyDtoDAsync_v2_ptsz:
            n = ((cuMemcpyDtoDAsync_v2_params *)params)->ByteCount, s = ((cuMemcpyDtoDAsync_v2_params *)params)->hStream;
            kind = "DtoD";
            break;
        case API_CUDA_cuMemcpyHtoD_v2:
            n = ((cuMemcpyHtoD_v2_params *)params)->ByteCount, kind = "HtoD";
            break;
        case API_CUDA_cuMemcpyDtoH_v2:
            n = ((cuMemcpyDtoH_v2_params *)params)->ByteCount, kind = "DtoH";
            break;
        case API_CUDA_cuMemcpyDtoD_v2:
            n = ((cuMemcpyDtoD_v2_params *)params)->ByteCount, kind = "DtoD";
            break;
        case API_CUDA_cuMemsetD8Async:
        case API_CUDA_cuMemsetD8Async_ptsz:
            n = ((cuMemsetD8Async_params *)params)->N, s = ((cuMemsetD8Async_params *)params)->hStream;
            kind = "memset";
            break;
        case API_CUDA_cuMemsetD32Async:
        case API_CUDA_cuMemsetD32Async_ptsz:
            n = 4 * ((cuMemsetD32Async_params *)params)->N, s = ((cuMemsetD32Async_params *)params)->hStream;
            kind = "memset";
            break;
        default:
            return;
    }
    s = real_stream(cbid, s);
    if (is_exit) {
        if (!paths_mode) time_end(s);
        return;
    }
    uint64_t i = seq++;
    if (ev_log)
        fprintf(ev_log, "{\"e\":\"C\",\"i\":%lu,\"k\":\"%s\",\"n\":%lu,\"s\":%s,\"h\":%ld}\n", (unsigned long)i,
                kind ? kind : "copy", (unsigned long)n, hex(s).c_str(), (long)host_t());
    if (!paths_mode) time_begin(i, s);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params,
                         CUresult *pStatus) {
    if (in_tool) return;
    std::lock_guard<std::recursive_mutex> g(mu);
    if (ctx_state_map.find(ctx) == ctx_state_map.end()) return;
    in_tool = true;
    last_ctx = ctx;
    if (is_launch(cbid)) {
        on_launch(ctx, is_exit, cbid, params);
    } else if (region && ev_log) {
        switch (cbid) {
            case API_CUDA_cuEventRecord:
            case API_CUDA_cuEventRecord_ptsz:
            case API_CUDA_cuEventRecordWithFlags:
            case API_CUDA_cuEventRecordWithFlags_ptsz: {
                // cuEventRecord and cuEventRecordWithFlags share the leading fields.
                auto *p = (cuEventRecord_params *)params;
                if (!is_exit)
                    fprintf(ev_log, "{\"e\":\"R\",\"ev\":%s,\"s\":%s,\"h\":%ld}\n", hex(p->hEvent).c_str(),
                            hex(real_stream(cbid, p->hStream)).c_str(), (long)host_t());
                break;
            }
            case API_CUDA_cuStreamWaitEvent:
            case API_CUDA_cuStreamWaitEvent_ptsz: {
                auto *p = (cuStreamWaitEvent_params *)params;
                if (!is_exit)
                    fprintf(ev_log, "{\"e\":\"W\",\"ev\":%s,\"s\":%s,\"h\":%ld}\n", hex(p->hEvent).c_str(),
                            hex(real_stream(cbid, p->hStream)).c_str(), (long)host_t());
                break;
            }
            case API_CUDA_cuStreamSynchronize:
            case API_CUDA_cuStreamSynchronize_ptsz:
            case API_CUDA_cuCtxSynchronize:
            case API_CUDA_cuEventSynchronize:
                if (is_exit)
                    fprintf(ev_log, "{\"e\":\"S\",\"k\":\"%s\",\"h\":%ld}\n", name, (long)host_t());
                break;
            case API_CUDA_cuGraphLaunch:
            case API_CUDA_cuGraphLaunch_ptsz:
                if (!is_exit) fprintf(ev_log, "{\"e\":\"G\",\"h\":%ld}\n", (long)host_t());
                break;
            default:
                on_copy(cbid, is_exit, params);
        }
    }
    in_tool = false;
}

/* ------------------------------------------------------- recv thread, ctx */

static void *recv_thread_fun(void *args) {
    CUcontext ctx = (CUcontext)args;
    mu.lock();
    CTXstate *cs = ctx_state_map[ctx];
    ChannelHost *ch_host = &cs->channel_host;
    mu.unlock();
    char *buf = (char *)malloc(CHANNEL_SIZE);
    while (cs->recv_thread_done == RecvThreadState::WORKING) {
        uint32_t n = ch_host->recv(buf, CHANNEL_SIZE);
        if (n == 0) recv_empty++;
        uint32_t off = 0;
        while (off < n) {
            wp_rec_t *r = (wp_rec_t *)&buf[off];
            size_t sz = wp_record_size(r);
            std::lock_guard<std::mutex> o(out_mu);
            if (out_bin && !out_truncated) {
                out_bytes += wp_write_record(ctx, out_bin, r, rec_opcode);
                if (out_bytes > max_bytes) out_truncated = true;
            }
            off += sz;
        }
    }
    free(buf);
    cs->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    std::lock_guard<std::recursive_mutex> g(mu);
    CTXstate *cs = new CTXstate;
    ctx_state_map[ctx] = cs;
    if (paths_mode) {
        nvbit_load_tool_module(ctx, (const void *)flush_channel_bin, &cs->tool_module);
        nvbit_find_function_by_name(ctx, cs->tool_module, "flush_channel", &cs->flush_channel_func);
    }
}

void nvbit_tool_init(CUcontext ctx) {
    std::lock_guard<std::recursive_mutex> g(mu);
    CTXstate *cs = ctx_state_map[ctx];
    if (ev_log) {
        CUdevice dev;
        cuCtxGetDevice(&dev);
        char nm[256] = {0};
        cuDeviceGetName(nm, sizeof nm, dev);
        int sms = 0, clk = 0, major = 0, minor = 0;
        cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
        cuDeviceGetAttribute(&clk, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        fprintf(ev_log, "{\"e\":\"D\",\"name\":\"%s\",\"cc\":[%d,%d],\"sms\":%d,\"clock_khz\":%d,\"pid\":%d,\"mode\":\"%s\",\"region\":\"%s\"}\n",
                jesc(nm).c_str(), major, minor, sms, clk, getpid(), paths_mode ? "paths" : "graph",
                region_all ? "all" : "marked");
    }
    if (!paths_mode) return;
    cs->recv_thread_done = RecvThreadState::WORKING;
    cudaMallocManaged(&cs->channel_dev, sizeof(ChannelDev));
    cs->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE, cs->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_thread(cs->channel_host.get_thread_id());
}

void nvbit_at_ctx_term(CUcontext ctx) {
    std::lock_guard<std::recursive_mutex> g(mu);
    in_tool = true;
    CTXstate *cs = ctx_state_map[ctx];
    if (!paths_mode && ref_recorded) {
        cuCtxSynchronize();
        drain(true);
    }
    if (paths_mode && cs->need_sync) {
        void *args[] = {&cs->channel_dev};
        nvbit_launch_kernel(ctx, cs->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
        cuCtxSynchronize();
    }
    if (cs->recv_thread_done != RecvThreadState::INIT) {
        cs->recv_thread_done = RecvThreadState::STOP;
        while (cs->recv_thread_done != RecvThreadState::FINISHED)
            ;
        cs->channel_host.destroy(false);
        cudaFree(cs->channel_dev);
    }
    {
        std::lock_guard<std::mutex> o(out_mu);
        if (out_bin) {
            fclose(out_bin);
            out_bin = nullptr;
        }
    }
    if (ev_log) fflush(ev_log);
    delete cs;
    ctx_state_map.erase(ctx);
    in_tool = false;
}

void nvbit_at_term() {
    std::lock_guard<std::recursive_mutex> g(mu);
    if (ev_log) {
        fclose(ev_log);
        ev_log = nullptr;
    }
}
