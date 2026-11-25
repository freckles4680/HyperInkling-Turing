// HyperInkling, Turing (sm75): relative-bias paged attention on the fp16 tensor cores.
// Triton emits no tensor-core MMA for sm75, so `_rel_attention_kernel`
// (ops/attention_sm90.py) runs its two dots as FMA through shared memory there. On a
// 2080 Ti that is 74% of an 8k prefill and, at 90k tokens, ~43 ms of a decode step whose KV
// reads take ~3 ms at the card's bandwidth.
//
// Same contract as `_rel_attention_kernel` over a uint8 E4M3 cache, for two of its modes:
// - MODE_PARTIAL: fp32 split partials with log2 LSEs (decode and verify, merged by
//   `_rel_decode_combine_kernel`);
// - MODE_OUT_LSE: the fp32 output and natural-log LSE (prefill, a single split).
// Full and sliding-window layers, with the sheared relative logits or without (the draft).
//
// A block owns `NW * 16` rows (tokens x GQA group) of one request and KV head: one warp
// for decode, four for prefill, so a prefill tile reads its keys once for 64 rows. Each
// warp keeps its 16 rows' Q and output accumulators in tensor-core fragments; the block
// shares the decoded keys and values, 32 keys an iteration. A key's 128 bytes are read by
// a whole warp (lane `l` takes word `l`): one contiguous transaction a key. (A lane
// reading its own key scatters every load over 32 rows 2 KB apart: 85 GB/s, measured.)
//
// Everything the threads exchange goes through shared memory between `PHASE`s, and
// fragments only move to and from shared memory, so the same phases run on the CPU
// (HOST_EMU, one thread after the other) against the Triton kernel:
// tests/test_sm75_attention.py.
//
//   scripts/build-sm75-kernels.sh        # venv/hyperinkling_sm75.so, about a minute
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef HOST_EMU
#define DEV
static inline float half_bits_to_float(uint16_t bits) {
  _Float16 h;
  memcpy(&h, &bits, 2);
  return (float)h;
}
static inline uint16_t float_to_half_bits(float x) {
  _Float16 h = (_Float16)x;
  uint16_t bits;
  memcpy(&bits, &h, 2);
  return bits;
}
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
#define DEV __device__ __forceinline__
static DEV float half_bits_to_float(uint16_t bits) {
  return __half2float(__ushort_as_half(bits));
}
static DEV uint16_t float_to_half_bits(float x) {
  return __half_as_ushort(__float2half_rn(x));
}
#endif

#define D 128      // head dim
#define ROWS 16    // rows of a warp
#define BN 32      // keys per iteration
#define DT (D / 16)
#define NT (BN / 16)
#define MAX_NW 4
#define LOG2E 1.4426950408889634f
#define LN2 0.6931471805599453f
#define MODE_OUT_LSE 0
#define MODE_PARTIAL 2

struct HiSm75AttnArgs {
  const uint16_t* q;        // fp16 (T, H, D), head stride D
  const uint8_t* k;         // E4M3 bytes, paged
  const uint8_t* v;
  const uint16_t* rel;      // fp16 sheared relative logits (T + 128, H, E + 256); unused if draft
  const int32_t* block_table;
  const int32_t* seq_lens;
  const int32_t* cu_seqlens_q;
  float* out;               // fp32: (S, T, H, D) partials, or (T, H, D) by strides
  float* lse;               // fp32: (S, T, H) log2 partials, or (H, T) natural log by strides
  int64_t stride_q_t;
  int64_t stride_kv_blk;
  int64_t stride_kv_slot;
  int64_t stride_kv_h;
  int64_t stride_rel_t;
  int64_t stride_bt;
  int64_t stride_out_t;     // MODE_OUT_LSE
  int64_t stride_out_h;
  int64_t stride_lse_h;
  int64_t stride_lse_t;
  float softmax_scale;
  int32_t num_tokens;
  int32_t num_splits;
  int32_t num_kv_heads;
  int32_t num_heads;
  int32_t group_shift;      // GQA group, a power of two
  int32_t extent;           // E
  int32_t page;             // a power of two
  int32_t page_shift;
  int32_t window;           // keys a query sees, itself included; 0: all
  int32_t draft;
  int32_t mode;
};
typedef struct HiSm75AttnArgs Args;

// A warp's part of the block's shared memory.
struct alignas(32) WarpShared {
  float s[ROWS * BN];       // scaled, biased, masked logits (log2 units)
  uint16_t p[ROWS * BN];    // fp16 probabilities relative to the running max
  float scratch[ROWS * 16]; // one accumulator fragment
  float part_max[32];
  float part_sum[32];
  float m[ROWS];            // running max (log2 units)
  float l[ROWS];            // running sum
  float alpha[ROWS];
};

// 12.7 KB for one warp (five blocks per SM in Turing's 64 KB), 26.4 KB for four (two).
template <int NW>
struct alignas(32) Shared {
  uint16_t kv[BN * D];      // fp16 bits: a warp's Q at start, then K and V of the iteration
  WarpShared w[NW];
  int32_t rescale;
};

// Per-thread registers: word `lane` of the V bytes of this thread's keys, decoded after
// the QK product.
struct Lane {
  uint32_t vraw[BN];
};

#ifdef HOST_EMU
// A fragment is the whole 16x16 tile, owned by lane 0 of its warp.
struct FragH { float e[16][16]; };
struct FragF { float e[16][16]; };
#define ONCE if (lane == 0)
static inline void frag_zero(FragF& f, int lane) { ONCE memset(&f, 0, sizeof f); }
static inline void load_a_row(FragH& f, const uint16_t* ptr, int ldm, int lane) {
  ONCE for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++)
    f.e[i][j] = half_bits_to_float(ptr[i * ldm + j]);
}
static inline void mma_col_b(FragF& c, const FragH& a, const uint16_t* b, int ldm, int lane) {
  // b: (k, n) at b[n * ldm + k]
  ONCE for (int i = 0; i < 16; i++) for (int n = 0; n < 16; n++) {
    float acc = c.e[i][n];
    for (int k = 0; k < 16; k++) acc += a.e[i][k] * half_bits_to_float(b[n * ldm + k]);
    c.e[i][n] = acc;
  }
}
static inline void mma_row_b(FragF& c, const FragH& a, const uint16_t* b, int ldm, int lane) {
  // b: (k, n) at b[k * ldm + n]
  ONCE for (int i = 0; i < 16; i++) for (int n = 0; n < 16; n++) {
    float acc = c.e[i][n];
    for (int k = 0; k < 16; k++) acc += a.e[i][k] * half_bits_to_float(b[k * ldm + n]);
    c.e[i][n] = acc;
  }
}
static inline void store_row(float* ptr, const FragF& f, int ldm, int lane) {
  ONCE for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) ptr[i * ldm + j] = f.e[i][j];
}
static inline void load_acc_row(FragF& f, const float* ptr, int ldm, int lane) {
  ONCE for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) f.e[i][j] = ptr[i * ldm + j];
}
#else
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> FragH;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> FragF;
static DEV void frag_zero(FragF& f, int) { wmma::fill_fragment(f, 0.0f); }
static DEV void load_a_row(FragH& f, const uint16_t* ptr, int ldm, int) {
  wmma::load_matrix_sync(f, (const __half*)ptr, ldm);
}
static DEV void mma_col_b(FragF& c, const FragH& a, const uint16_t* b, int ldm, int) {
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
  wmma::load_matrix_sync(fb, (const __half*)b, ldm);
  wmma::mma_sync(c, a, fb, c);
}
static DEV void mma_row_b(FragF& c, const FragH& a, const uint16_t* b, int ldm, int) {
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
  wmma::load_matrix_sync(fb, (const __half*)b, ldm);
  wmma::mma_sync(c, a, fb, c);
}
static DEV void store_row(float* ptr, const FragF& f, int ldm, int) {
  wmma::store_matrix_sync(ptr, f, ldm, wmma::mem_row_major);
}
static DEV void load_acc_row(FragF& f, const float* ptr, int ldm, int) {
  wmma::load_matrix_sync(f, ptr, ldm, wmma::mem_row_major);
}
#endif

// A warp's fragments.
struct Frags {
  FragH q[DT];
  FragF acc[DT];
  FragH pa[NT];
};

// What a block derives from its ids; the same in every thread.
struct Block {
  int32_t kv_head, split, req;
  int32_t q_start, q_len, seqlen_k, past, first_tok, tokens;
  int32_t n_lo, num_iters;
  float qk_scale;
};

static DEV bool block_init(const Args& a, int nw, int bx, int by, int bz, Block& b) {
  b.kv_head = by / a.num_splits;
  b.split = by % a.num_splits;
  b.req = bz;
  b.q_start = a.cu_seqlens_q[bz];
  b.q_len = a.cu_seqlens_q[bz + 1] - b.q_start;
  b.tokens = (nw * ROWS) >> a.group_shift;
  b.first_tok = bx * b.tokens;
  if (b.first_tok >= b.q_len) return false;
  b.seqlen_k = a.seq_lens[bz];
  b.past = b.seqlen_k - b.q_len;
  int32_t last = b.first_tok + b.tokens < b.q_len ? b.first_tok + b.tokens : b.q_len;
  int32_t pos_lo = a.draft ? b.seqlen_k - 1 : b.past + b.first_tok;
  int32_t pos_hi = a.draft ? b.seqlen_k - 1 : b.past + last - 1;
  b.n_lo = 0;
  if (a.window > 0 && pos_lo - (a.window - 1) > 0) b.n_lo = (pos_lo - (a.window - 1)) / BN;
  int32_t n_hi = pos_hi + 1 < b.seqlen_k ? pos_hi + 1 : b.seqlen_k;
  n_hi = (n_hi + BN - 1) / BN;
  int32_t iters = n_hi - b.n_lo - b.split + a.num_splits - 1;
  b.num_iters = iters > 0 ? iters / a.num_splits : 0;
  // The cache bytes moved into fp16 hold value * 2^-8 (`e4m3_f16_scaled`).
  b.qk_scale = a.softmax_scale * LOG2E * 256.0f;
  return true;
}

struct Row {
  int32_t tok, head;
  int64_t token;
  bool valid;
};

// Row `r` of warp `w`.
static DEV Row row_of(const Args& a, const Block& b, int w, int r) {
  int row_in_block = w * ROWS + r;
  Row row;
  row.tok = b.first_tok + (row_in_block >> a.group_shift);
  row.valid = row_in_block < (b.tokens << a.group_shift) && row.tok < b.q_len;
  row.head = (b.kv_head << a.group_shift) + (row_in_block & ((1 << a.group_shift) - 1));
  row.token = (int64_t)b.q_start + row.tok;
  return row;
}

// Two E4M3 bytes -> two fp16 holding value * 2^-8: exponent and mantissa moved into the
// fp16 fields, the sign to bit 15 (exact, subnormals included).
static DEV uint32_t decode2(uint32_t two_bytes) {
  uint32_t x = (two_bytes & 0xFFu) | ((two_bytes & 0xFF00u) << 8);
  return ((x & 0x007F007Fu) << 7) | ((x & 0x00800080u) << 8);
}

// Word `lane` of key row `j` (4 bytes) as 4 fp16 in `kv`.
static DEV void store_decoded(uint16_t* kv, int j, int lane, uint32_t word) {
  uint32_t* out = (uint32_t*)(kv + j * D) + 2 * lane;
  out[0] = decode2(word & 0xFFFFu);
  out[1] = decode2(word >> 16);
}

// The block's threads copy the Q rows of warp `w` into `kv` (zero past the request's
// tokens), and that warp resets its running max and sum.
template <int NW>
static DEV void phase_q(const Args& a, const Block& b, Shared<NW>& sm, int w, int t) {
  const int per_thread = ROWS * D / (NW * 32);
  for (int i = t * per_thread; i < (t + 1) * per_thread; i++) {
    int r = i / D, d = i % D;
    Row row = row_of(a, b, w, r);
    sm.kv[i] = row.valid ? a.q[row.token * a.stride_q_t + (int64_t)row.head * D + d] : (uint16_t)0;
  }
  if (t / 32 == w && t % 32 < ROWS) {
    sm.w[w].m[t % 32] = -INFINITY;
    sm.w[w].l[t % 32] = 0.0f;
  }
}

template <int NW>
static DEV void phase_q_frags(Shared<NW>& sm, Frags& f, int w, int t) {
  if (t / 32 != w) return;
  for (int dt = 0; dt < DT; dt++) {
    load_a_row(f.q[dt], sm.kv + dt * 16, D, t % 32);
    frag_zero(f.acc[dt], t % 32);
  }
}

// Keys `n * BN ..`, a key at a time for a warp: K decoded into `kv`, V kept as bytes.
template <int NW>
static DEV void phase_load(const Args& a, const Block& b, Shared<NW>& sm, Lane& ln, int n, int t) {
  const int keys = BN / NW, lane = t % 32;
  for (int i = 0; i < keys; i++) {
    int j = (t / 32) * keys + i;
    int32_t key = n * BN + j;
    uint32_t kw = 0, vw = 0;
    if (key < b.seqlen_k) {
      int32_t page = a.block_table[(int64_t)b.req * a.stride_bt + (key >> a.page_shift)];
      int64_t off = (int64_t)page * a.stride_kv_blk +
                    (int64_t)(key & (a.page - 1)) * a.stride_kv_slot +
                    (int64_t)b.kv_head * a.stride_kv_h;
      kw = ((const uint32_t*)(a.k + off))[lane];
      vw = ((const uint32_t*)(a.v + off))[lane];
    }
    store_decoded(sm.kv, j, lane, kw);
    ln.vraw[i] = vw;
  }
}

// S = Q K^T of the warp's rows for the 32 keys.
template <int NW>
static DEV void phase_qk(Shared<NW>& sm, Frags& f, int t) {
  WarpShared& ws = sm.w[t / 32];
  for (int j = 0; j < NT; j++) {
    FragF c;
    frag_zero(c, t % 32);
    for (int dt = 0; dt < DT; dt++)
      mma_col_b(c, f.q[dt], sm.kv + j * 16 * D + dt * 16, D, t % 32);
    store_row(ws.s + j * 16, c, BN, t % 32);
  }
}

// V replaces K in `kv`. Lane `c` takes key column `c` of its warp's 16 rows: scale, bias
// (a row's logits are contiguous along the keys) and mask.
template <int NW>
static DEV void phase_logits(const Args& a, const Block& b, Shared<NW>& sm, const Lane& ln,
                             int n, int t) {
  const int keys = BN / NW, w = t / 32, c = t % 32;
  for (int i = 0; i < keys; i++) store_decoded(sm.kv, w * keys + i, c, ln.vraw[i]);
  WarpShared& ws = sm.w[w];
  const int32_t key = n * BN + c;
  const int32_t e_pad = a.extent + 256;
  for (int r = 0; r < ROWS; r++) {
    Row row = row_of(a, b, w, r);
    int32_t pos = a.draft ? b.seqlen_k - 1 : b.past + row.tok;
    int32_t dist = pos - key;
    bool visible = dist >= 0 && key < b.seqlen_k;
    if (a.window > 0) visible = visible && dist < a.window;
    float s = ws.s[r * BN + c] * b.qk_scale;
    if (!a.draft && visible && dist < a.extent && row.valid) {
      // Sheared layout of attention_prologue (128-row query tiles).
      int32_t tile_end = (row.tok / 128 + 1) * 128;
      if (tile_end > b.q_len) tile_end = b.q_len;
      int32_t rel_base = (b.past + tile_end + 127) / 128 * 128 - e_pad;
      const uint16_t* rel_row =
          a.rel + row.token * a.stride_rel_t + (int64_t)row.head * e_pad - rel_base;
      s += half_bits_to_float(rel_row[key]) * LOG2E;
    }
    ws.s[r * BN + c] = visible ? s : -INFINITY;
  }
  if (t == 0) sm.rescale = 0;
}

// Lane `l`: the max of 16 logits of row `l % 16`.
template <int NW>
static DEV void phase_max(Shared<NW>& sm, int t) {
  WarpShared& ws = sm.w[t / 32];
  int lane = t % 32, r = lane % ROWS, c0 = (lane / ROWS) * 16;
  float best = -INFINITY;
  for (int c = c0; c < c0 + 16; c++) best = fmaxf(best, ws.s[r * BN + c]);
  ws.part_max[lane] = best;
}

static DEV float new_max(const WarpShared& ws, int r) {
  return fmaxf(ws.m[r], fmaxf(ws.part_max[r], ws.part_max[r + ROWS]));
}

// Probabilities relative to the new running max, as fp16.
template <int NW>
static DEV void phase_probs(Shared<NW>& sm, int t) {
  WarpShared& ws = sm.w[t / 32];
  int lane = t % 32, r = lane % ROWS, c0 = (lane / ROWS) * 16;
  float m_new = new_max(ws, r);
  float m_safe = m_new == -INFINITY ? 0.0f : m_new;
  float sum = 0.0f;
  for (int c = c0; c < c0 + 16; c++) {
    float p = exp2f(ws.s[r * BN + c] - m_safe);
    sum += p;
    ws.p[r * BN + c] = float_to_half_bits(p);
  }
  ws.part_sum[lane] = sum;
}

template <int NW>
static DEV void phase_state(Shared<NW>& sm, int t) {
  WarpShared& ws = sm.w[t / 32];
  int r = t % 32;
  if (r >= ROWS) return;
  float m_new = new_max(ws, r);
  float m_safe = m_new == -INFINITY ? 0.0f : m_new;
  float alpha = exp2f(ws.m[r] - m_safe);
  ws.alpha[r] = alpha;
  ws.l[r] = ws.l[r] * alpha + (ws.part_sum[r] + ws.part_sum[r + ROWS]);
  ws.m[r] = m_new;
  if (alpha != 1.0f) sm.rescale = 1;  // every writer stores the same value
}

template <int NW>
static DEV void phase_scale(Shared<NW>& sm, int t) {
  WarpShared& ws = sm.w[t / 32];
  int lane = t % 32;
  for (int i = lane * 8; i < lane * 8 + 8; i++) ws.scratch[i] *= ws.alpha[i / 16];
}

// acc += P V.
template <int NW>
static DEV void phase_pv(Shared<NW>& sm, Frags& f, int t) {
  WarpShared& ws = sm.w[t / 32];
  for (int kt = 0; kt < NT; kt++) load_a_row(f.pa[kt], ws.p + kt * 16, BN, t % 32);
  for (int dt = 0; dt < DT; dt++)
    for (int kt = 0; kt < NT; kt++)
      mma_row_b(f.acc[dt], f.pa[kt], sm.kv + kt * 16 * D + dt * 16, D, t % 32);
}

template <int NW>
static DEV void phase_out(const Args& a, const Block& b, Shared<NW>& sm, int dt, int t) {
  int w = t / 32, lane = t % 32;
  WarpShared& ws = sm.w[w];
  for (int i = lane * 8; i < lane * 8 + 8; i++) {
    int r = i / 16;
    Row row = row_of(a, b, w, r);
    if (!row.valid) continue;
    float l = ws.l[r];
    float o = ws.scratch[i] * 256.0f / (l == 0.0f ? 1.0f : l);
    int d = dt * 16 + i % 16;
    if (a.mode == MODE_PARTIAL) {
      int64_t part = ((int64_t)b.split * a.num_tokens + row.token) * a.num_heads + row.head;
      a.out[part * D + d] = o;
    } else {
      a.out[row.token * a.stride_out_t + row.head * a.stride_out_h + d] = o;
    }
  }
}

template <int NW>
static DEV void phase_lse(const Args& a, const Block& b, Shared<NW>& sm, int t) {
  int w = t / 32, r = t % 32;
  if (r >= ROWS) return;
  Row row = row_of(a, b, w, r);
  if (!row.valid) return;
  float l = sm.w[w].l[r];
  float lse2 = l == 0.0f ? -INFINITY : sm.w[w].m[r] + log2f(l);
  if (a.mode == MODE_PARTIAL) {
    a.lse[((int64_t)b.split * a.num_tokens + row.token) * a.num_heads + row.head] = lse2;
  } else {
    a.lse[row.head * a.stride_lse_h + row.token * a.stride_lse_t] = lse2 * LN2;
  }
}

#ifdef HOST_EMU
#define PHASE(call) \
  for (int t = 0; t < NW * 32; t++) { Lane& ln = lanes[t]; Frags& f = frags[t / 32]; \
    (void)ln; (void)f; call; }
#define KERNEL_HEAD template <int NW> static void run_block(const Args& a, int bx, int by, int bz)
#define KERNEL_STATE \
  static Shared<NW> sm; static Lane lanes[NW * 32]; static Frags frags[NW];
#else
#define PHASE(call) { call; __syncthreads(); }
#define KERNEL_HEAD template <int NW> __global__ void rel_attention_kernel(const Args a)
#define KERNEL_STATE \
  __shared__ Shared<NW> sm; Lane ln; Frags f; \
  const int t = threadIdx.x, bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
#endif

KERNEL_HEAD {
  KERNEL_STATE
  Block b;
  if (!block_init(a, NW, bx, by, bz, b)) return;
  for (int w = 0; w < NW; w++) {
    PHASE(phase_q(a, b, sm, w, t))
    PHASE(phase_q_frags(sm, f, w, t))
  }
  for (int i = 0; i < b.num_iters; i++) {
    const int n = b.n_lo + b.split + i * a.num_splits;
    PHASE(phase_load(a, b, sm, ln, n, t))
    PHASE(phase_qk(sm, f, t))
    PHASE(phase_logits(a, b, sm, ln, n, t))
    PHASE(phase_max(sm, t))
    PHASE(phase_probs(sm, t))
    PHASE(phase_state(sm, t))
    if (sm.rescale) {
      for (int dt = 0; dt < DT; dt++) {
        PHASE(store_row(sm.w[t / 32].scratch, f.acc[dt], 16, t % 32))
        PHASE(phase_scale(sm, t))
        PHASE(load_acc_row(f.acc[dt], sm.w[t / 32].scratch, 16, t % 32))
      }
    }
    PHASE(phase_pv(sm, f, t))
  }
  for (int dt = 0; dt < DT; dt++) {
    PHASE(store_row(sm.w[t / 32].scratch, f.acc[dt], 16, t % 32))
    PHASE(phase_out(a, b, sm, dt, t))
  }
  PHASE(phase_lse(a, b, sm, t))
}

extern "C" {
// `tiles`: blocks along the query tokens (`warps * 16 / group` tokens each); `warps` 1 or 4.
#ifdef HOST_EMU
int hi_sm75_rel_attention(const Args* a, int num_reqs, int tiles, int warps, void*) {
  for (int bz = 0; bz < num_reqs; bz++)
    for (int by = 0; by < a->num_kv_heads * a->num_splits; by++)
      for (int bx = 0; bx < tiles; bx++) {
        if (warps == 1) run_block<1>(*a, bx, by, bz);
        else run_block<MAX_NW>(*a, bx, by, bz);
      }
  return 0;
}
#else
int hi_sm75_rel_attention(const Args* a, int num_reqs, int tiles, int warps, void* stream) {
  dim3 grid(tiles, a->num_kv_heads * a->num_splits, num_reqs);
  if (warps == 1)
    rel_attention_kernel<1><<<grid, 32, 0, (cudaStream_t)stream>>>(*a);
  else
    rel_attention_kernel<MAX_NW><<<grid, MAX_NW * 32, 0, (cudaStream_t)stream>>>(*a);
  return (int)cudaGetLastError();
}
#endif
int hi_sm75_abi(void) { return 3; }
}
