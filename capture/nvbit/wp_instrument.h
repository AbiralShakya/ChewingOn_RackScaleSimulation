// Instrumentation shared by the warp_path and rsim_capture NVBit tools: which
// device function records each SASS instruction (see inject_funcs.cu), and the
// static listing written next to every recording.
#pragma once
#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "nvbit.h"

static bool starts_with(const char *s, const char *p) { return strncmp(s, p, strlen(p)) == 0; }

/* mbarrier-style operations whose memory operand is a barrier in shared memory */
static bool is_mbar_op(const char *op) {
    return starts_with(op, "SYNCS") || starts_with(op, "UTCBAR") || starts_with(op, "ARRIVES") ||
           (starts_with(op, "LDGSTS") && strstr(op, "ARV"));
}

static int ureg_in_generic(const InstrType::operand_t *o) {
    // e.g. "idesc[UR11]" or "tmem[UR8]"; return the UR number or -1
    const char *p = strstr(o->str, "UR");
    if (!p || !isdigit(p[2])) return -1;
    return atoi(p + 2);
}

/* A function attribute. The runtime launches library kernels (CUkernel) through
 * cuLaunchKernel, and cuFuncGetAttribute rejects those handles. */
static int func_attr(CUfunction f, CUfunction_attribute a) {
    int v = 0;
    if (cuFuncGetAttribute(&v, a, f) == CUDA_SUCCESS) return v;
    CUfunction g;
    if (cuKernelGetFunction(&g, (CUkernel)f) == CUDA_SUCCESS && cuFuncGetAttribute(&v, a, g) == CUDA_SUCCESS) return v;
    CUdevice d;
    if (cuCtxGetDevice(&d) == CUDA_SUCCESS && cuKernelGetAttribute(&v, a, (CUkernel)f, d) == CUDA_SUCCESS) return v;
    return 0;
}

/* The static listing in cuobjdump format, with both 64-bit halves of every encoding. */
static void write_listing(CUcontext ctx, CUfunction f, const std::string &path) {
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    FILE *fp = fopen(path.c_str(), "w");
    if (!fp) return;
    fprintf(fp, "\t\tFunction : %s\n", nvbit_get_func_name(ctx, f, true));
    for (auto ins : instrs) {
        std::vector<uint8_t> enc;
        ins->getSassBinary([](uint8_t b, void *u) { ((std::vector<uint8_t> *)u)->push_back(b); }, &enc);
        uint64_t lo = 0, hi = 0;
        for (int i = 0; i < 8 && i < (int)enc.size(); i++) lo |= (uint64_t)enc[i] << (8 * i);
        for (int i = 8; i < 16 && i < (int)enc.size(); i++) hi |= (uint64_t)enc[i] << (8 * (i - 8));
        fprintf(fp, "        /*%04x*/  %s ;  /* 0x%016lx */\n", ins->getOffset(), ins->getSass(), lo);
        fprintf(fp, "                    /* 0x%016lx */\n", hi);
    }
    fclose(fp);
}

/* Insert the recording calls into every instruction of f. The sampled CTA ids
 * (s01, s23), the TMA-for-all-CTAs flag and the channel are baked in as constants. */
static void wp_instrument(CUcontext ctx, CUfunction f, uint64_t s01, uint64_t s23, uint32_t wp_tma_all,
                          void *channel_dev) {
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    for (auto ins : instrs) {
        const char *op = ins->getOpcode();
        uint32_t idx = ins->getIdx();
        auto ms = ins->getMemorySpace();
        int mref = -1, first_reg = -1, first_ureg = -1, second_ureg = -1, dst_pred = -1, idesc_ur = -1;
        int nreg_seen = 0;
        for (int i = 0; i < ins->getNumOperands(); i++) {
            const InstrType::operand_t *o = ins->getOperand(i);
            switch (o->type) {
                case InstrType::OperandType::MREF:
                    if (mref < 0) mref = i;
                    break;
                case InstrType::OperandType::PRED:
                    if (i == 0 && dst_pred < 0) dst_pred = o->u.pred.num;
                    break;
                case InstrType::OperandType::REG:
                    // skip the destination (operand 0) and RZ
                    if (i > 0 && o->u.reg.num != 255 && first_reg < 0 && (mref < 0 || i > mref))
                        first_reg = o->u.reg.num;
                    nreg_seen++;
                    break;
                case InstrType::OperandType::UREG:
                    if (o->u.reg.num != 255 && o->u.reg.num != 63) {
                        if (first_ureg < 0) first_ureg = o->u.reg.num;
                        else if (second_ureg < 0) second_ureg = o->u.reg.num;
                    }
                    break;
                case InstrType::OperandType::MEM_DESC:
                    if (first_ureg < 0) first_ureg = o->u.mem_desc.ureg_num;
                    else if (second_ureg < 0) second_ureg = o->u.mem_desc.ureg_num;
                    break;
                case InstrType::OperandType::GENERIC:
                    if (strstr(o->str, "idesc")) idesc_ur = ureg_in_generic(o);
                    break;
                default:
                    break;
            }
        }
        (void)nreg_seen;
        if (starts_with(op, "UTC")) {
            // tcgen05 operands print as gdesc[URn] / tmem[URn] / idesc[URn]; take the
            // register numbers from the text, which is stable across NVBit versions.
            const char *sass = ins->getSass();
            const char *p = strstr(sass, "idesc[UR");
            if (p) idesc_ur = atoi(p + 8);
            std::vector<int> urs;
            for (const char *q = sass; (q = strstr(q, "UR")) != nullptr; q += 2)
                if (isdigit(q[2])) urs.push_back(atoi(q + 2));
            if (first_ureg < 0 && urs.size() > 0) first_ureg = urs[0];
            if (second_ureg < 0 && urs.size() > 1) second_ureg = urs.back();
        }
        if (ins->isTMAMem()) {
            if (strstr(op, "MULTICAST")) {
                // The CTA mask of a multicast copy is its last uniform-register operand:
                // record it (as a WP_UREG record of the same instruction) ahead of the copy.
                int mask_ur = -1;
                for (int i = 0; i < ins->getNumOperands(); i++)
                    if (ins->getOperand(i)->type == InstrType::OperandType::UREG) mask_ur = ins->getOperand(i)->u.reg.num;
                if (mask_ur >= 0) {
                    nvbit_insert_call(ins, "wp_ureg", IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(ins);
                    nvbit_add_call_arg_const_val32(ins, idx);
                    nvbit_add_call_arg_ureg_val(ins, mask_ur);
                    nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
                    nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
                    nvbit_add_call_arg_const_val64(ins, s01);
                    nvbit_add_call_arg_const_val64(ins, s23);
                    nvbit_add_call_arg_const_val64(ins, (uint64_t)channel_dev);
                }
            }
            nvbit_insert_call(ins, "wp_tma", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(ins);
            nvbit_add_call_arg_const_val32(ins, idx);
            nvbit_add_call_arg_tma_param_handle_and_size(ins, ctx);
            nvbit_add_call_arg_const_val32(ins, wp_tma_all);
        } else if (starts_with(op, "SYNCS") && (strstr(op, "TRYWAIT") || strstr(op, "PHASECHK")) && mref >= 0) {
            nvbit_insert_call(ins, "wp_vals", IPOINT_AFTER);
            nvbit_add_call_arg_guard_pred_val(ins);
            nvbit_add_call_arg_const_val32(ins, idx);
            nvbit_add_call_arg_const_val32(ins, WP_WAIT);
            nvbit_add_call_arg_const_val32(ins, dst_pred < 0 ? 0 : dst_pred);
            nvbit_add_call_arg_mref_addr64(ins, 0);
            nvbit_add_call_arg_pred_reg(ins);
            if (first_reg >= 0) nvbit_add_call_arg_reg_val(ins, first_reg);
            else nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
        } else if (is_mbar_op(op) && mref >= 0 && starts_with(op, "SYNCS.ARRIVE") &&
                   strstr(ins->getSass(), "[R") != nullptr) {
            nvbit_insert_call(ins, "wp_mbarv", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(ins);
            nvbit_add_call_arg_const_val32(ins, idx);
            nvbit_add_call_arg_mref_addr64(ins, 0);
            if (first_reg >= 0) nvbit_add_call_arg_reg_val(ins, first_reg);
            else nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
            if (first_ureg >= 0) nvbit_add_call_arg_ureg_val(ins, first_ureg);
            else nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
        } else if (is_mbar_op(op) && mref >= 0) {
            nvbit_insert_call(ins, "wp_vals", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(ins);
            nvbit_add_call_arg_const_val32(ins, idx);
            nvbit_add_call_arg_const_val32(ins, WP_MBAR);
            nvbit_add_call_arg_const_val32(ins, 0);
            nvbit_add_call_arg_mref_addr64(ins, 0);
            if (first_reg >= 0) nvbit_add_call_arg_reg_val(ins, first_reg);
            else nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
            if (first_ureg >= 0) nvbit_add_call_arg_ureg_val(ins, first_ureg);
            else nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
        } else if (starts_with(op, "UTC") && (first_ureg >= 0 || idesc_ur >= 0)) {
            nvbit_insert_call(ins, "wp_ureg", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(ins);
            nvbit_add_call_arg_const_val32(ins, idx);
            if (idesc_ur >= 0) nvbit_add_call_arg_ureg_val(ins, idesc_ur);
            else nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
            if (first_ureg >= 0) nvbit_add_call_arg_ureg_val(ins, first_ureg);
            else nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
            if (second_ureg >= 0) nvbit_add_call_arg_ureg_val(ins, second_ureg);
            else nvbit_add_call_arg_const_val32(ins, 0xffffffffu);
        } else if (starts_with(op, "BAR.") && ins->getNumOperands() > 0 &&
                   ins->getOperand(0)->type == InstrType::OperandType::REG) {
            nvbit_insert_call(ins, "wp_vals", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(ins);
            nvbit_add_call_arg_const_val32(ins, idx);
            nvbit_add_call_arg_const_val32(ins, WP_BAR);
            nvbit_add_call_arg_const_val32(ins, 0);
            nvbit_add_call_arg_const_val64(ins, 0);
            nvbit_add_call_arg_reg_val(ins, ins->getOperand(0)->u.reg.num);
            nvbit_add_call_arg_const_val32(ins, 0);
        } else if (mref >= 0 && ms != InstrType::MemorySpace::NONE && ms != InstrType::MemorySpace::CONSTANT) {
            // cp.async (LDGSTS) has the shared destination first; record the global source.
            int nmref = 0;
            for (int i = 0; i < ins->getNumOperands(); i++)
                nmref += ins->getOperand(i)->type == InstrType::OperandType::MREF;
            bool g2s = ms == InstrType::MemorySpace::GLOBAL_TO_SHARED && nmref > 1;
            nvbit_insert_call(ins, "wp_mem", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(ins);
            nvbit_add_call_arg_const_val32(ins, idx);
            nvbit_add_call_arg_mref_addr64(ins, g2s ? 1 : 0);
        } else {
            nvbit_insert_call(ins, "wp_basic", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(ins);
            nvbit_add_call_arg_const_val32(ins, idx);
        }
        nvbit_add_call_arg_const_val64(ins, s01);
        nvbit_add_call_arg_const_val64(ins, s23);
        nvbit_add_call_arg_const_val64(ins, (uint64_t)channel_dev);
    }
}

/* Bytes one channel record occupies. */
static size_t wp_record_size(const wp_rec_t *r) {
    return (r->kind == WP_MEM || r->kind == WP_MBARV) ? sizeof(wp_mem_t)
           : r->kind == WP_TMA                         ? sizeof(wp_tma_t)
                                                       : sizeof(wp_rec_t);
}

/* Write one channel record to the .wp file. A TMA record's raw parameter
 * handle is replaced by what the simulator needs: the total bytes and the
 * contiguous source runs in global memory (then the handle itself). Returns
 * the bytes written. `opcode` maps instruction index to opcode for the
 * recorded function. */
static size_t wp_write_record(CUcontext ctx, FILE *out, wp_rec_t *r, std::map<uint32_t, std::string> &opcode) {
    size_t sz = wp_record_size(r);
    if (r->kind == WP_TMA) {
        wp_tma_t *t = (wp_tma_t *)r;
        const char *op = opcode[r->idx].c_str();
        uint64_t bytes = 0;
        std::vector<TMAElementAddress_t> src;
        uint32_t esz = 0;
        if (t->r.predon) {
            TMATransferInfo_t info = nvbit_parse_tma_transfer_info(ctx, op, t->handle, t->handle_size);
            bytes = info.byte_count;
            esz = info.transfer_size;
            nvbit_parse_tma_src_addrs(
                ctx, op, t->handle, t->handle_size,
                [](const TMAElementAddress_t *a, size_t c, void *u) {
                    ((std::vector<TMAElementAddress_t> *)u)->assign(a, a + c);
                },
                &src);
        }
        std::vector<std::pair<uint64_t, uint64_t>> runs;
        for (auto &e : src) {
            if (e.is_oob) continue;
            if (!runs.empty() && runs.back().first + runs.back().second == e.address)
                runs.back().second += esz;
            else
                runs.push_back({e.address, esz});
        }
        t->r.a = bytes;
        t->r.b = runs.size();
        t->r.c = t->handle_size;
        fwrite(&t->r, sizeof(wp_rec_t), 1, out);
        sz = sizeof(wp_rec_t) + 16 * runs.size() + 128;
        for (auto &x : runs) fwrite(&x, 16, 1, out);
        fwrite(t->handle, 128, 1, out);
    } else {
        fwrite(r, sz, 1, out);
    }
    return sz;
}
