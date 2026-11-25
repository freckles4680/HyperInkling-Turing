// HyperInkling, Turing (sm75): split-KV decode/verify attention of the full layers on the
// fp16 tensor cores. Triton emits no tensor-core MMA for sm75, so `_rel_attention_kernel`
// (ops/attention_sm90.py) runs its two dots as FMA through shared memory there: ~43 ms a
// step at 90k tokens on a 2080 Ti, against ~3 ms of KV reads at the card's bandwidth.
//
// Same contract as `_rel_attention_kernel` in MODE_PARTIAL with WINDOW 0 and a uint8 E4M3
// cache: a block owns the 16 rows (tokens x GQA group) of one request and KV head and one
// KV split, and writes fp32 partials and log2 LSEs for `_rel_decode_combine_kernel`.
//
// One warp per block, 32 keys per iteration. A key's 128 bytes are read by the whole warp
// (lane `l` takes word `l`): one contiguous transaction a key. A lane reading its own key
// instead scatters every load over 32 rows 2 KB apart (85 GB/s on a 2080 Ti, measured).
// Everything the lanes exchange
// goes through shared memory between `PHASE`s, and tensor-core fragments only move to and
// from shared memory, so the same phases run on the CPU (HOST_EMU, one lane after the
// other) against the Triton kernel: tests/test_sm75_attention.py.
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
#define ROWS 16    // tokens x GQA group of a block
#define BN 32      // keys per iteration
#define DT (D / 16)
#define NT (BN / 16)
#define LOG2E 1.4426950408889634f

struct HiSm75AttnArgs {
  const uint16_t* q;        // fp16 (T, H, D), head stride D
  const uint8_t* k;         // E4M3 bytes, paged
  const uint8_t* v;
  const uint16_t* rel;      // fp16 sheared relative logits (T + 128, H, E + 256); unused if draft
  const int32_t* block_table;
  const int32_t* seq_lens;
  const int32_t* cu_seqlens_q;
  float* out;               // fp32 (S, T, H, D)
  float* lse;               // fp32 (S, T, H), log2
  int64_t stride_q_t;
  int64_t stride_kv_blk;
  int64_t stride_kv_slot;
  int64_t stride_kv_h;
  int64_t stride_rel_t;
  int64_t stride_bt;
  float softmax_scale;
  int32_t num_tokens;
  int32_t num_splits;
  int32_t num_kv_heads;
  int32_t num_heads;
  int32_t group;
  int32_t extent;           // E
  int32_t page;             // a power of two
  int32_t page_shift;
  int32_t draft;
};
typedef struct HiSm75AttnArgs Args;

// Shared memory of a block (~12.3 KB: five blocks per SM in Turing's 64 KB).
struct alignas(32) Shared {
  uint16_t kv[BN * D];      // fp16 bits: Q at start, then K and V of the iteration
  float s[ROWS * BN];       // scaled, biased, masked logits (log2 units)
  uint16_t p[ROWS * BN];    // fp16 probabilities relative to the running max
  float scratch[ROWS * 16]; // one accumulator fragment
  float part_max[32];
  float part_sum[32];
  float m[ROWS];            // running max (log2 units)
  float l[ROWS];            // running sum
  float alpha[ROWS];
  int32_t rescale;
};

// Per-lane registers.
struct Lane {
  uint32_t vraw[BN];        // word `lane` of each key's V bytes, decoded after the QK product
};

#ifdef HOST_EMU
// A fragment is the whole 16x16 tile, owned by lane 0.
struct FragH { float e[16][16]; };
struct FragF { float e[16][16]; };
struct Frags {
  FragH q[DT];
  FragF acc[DT];
  FragH pa[NT];
};
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
struct Frags {
  FragH q[DT];
  FragF acc[DT];
  FragH pa[NT];
};
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

// What a block derives from its ids; the same in every lane.
struct Block {
  int32_t kv_head, split, req;
  int32_t q_start, q_len, seqlen_k, past, tokens;
  int32_t num_iters;
  float qk_scale;
};

static DEV bool block_init(const Args& a, int by, int bz, Block& b) {
  b.kv_head = by / a.num_splits;
  b.split = by % a.num_splits;
  b.req = bz;
  b.q_start = a.cu_seqlens_q[bz];
  b.q_len = a.cu_seqlens_q[bz + 1] - b.q_start;
  if (b.q_len <= 0) return false;
  b.seqlen_k = a.seq_lens[bz];
  b.past = b.seqlen_k - b.q_len;
  b.tokens = ROWS / a.group;
  int32_t last = b.tokens < b.q_len ? b.tokens : b.q_len;
  int32_t pos_hi = a.draft ? b.seqlen_k - 1 : b.past + last - 1;
  int32_t n_hi = pos_hi + 1 < b.seqlen_k ? pos_hi + 1 : b.seqlen_k;
  n_hi = (n_hi + BN - 1) / BN;
  int32_t iters = n_hi - b.split + a.num_splits - 1;
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

static DEV Row row_of(const Args& a, const Block& b, int r) {
  Row row;
  row.tok = r / a.group;
  row.valid = r < b.tokens * a.group && row.tok < b.q_len;
  row.head = b.kv_head * a.group + r % a.group;
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

// Q of the block's rows into `kv` (zero for rows past the request's tokens).
static DEV void phase_q(const Args& a, const Block& b, Shared& sm, int lane) {
  int r = lane % ROWS, d0 = (lane / ROWS) * (D / 2);
  Row row = row_of(a, b, r);
  const uint16_t* src = a.q + row.token * a.stride_q_t + (int64_t)row.head * D + d0;
  for (int d = 0; d < D / 2; d++) sm.kv[r * D + d0 + d] = row.valid ? src[d] : (uint16_t)0;
  if (lane < ROWS) {
    sm.m[r] = -INFINITY;
    sm.l[r] = 0.0f;
  }
}

static DEV void phase_q_frags(Shared& sm, Frags& f, int lane) {
  for (int dt = 0; dt < DT; dt++) {
    load_a_row(f.q[dt], sm.kv + dt * 16, D, lane);
    frag_zero(f.acc[dt], lane);
  }
}

// The warp loads keys `n * BN ..`, a key at a time: K decoded into `kv`, V kept as bytes.
static DEV void phase_load(const Args& a, const Block& b, Shared& sm, Lane& ln, int n, int lane) {
  for (int j = 0; j < BN; j++) {
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
    ln.vraw[j] = vw;
  }
}

// S = Q K^T for the 32 keys.
static DEV void phase_qk(Shared& sm, Frags& f, int lane) {
  for (int j = 0; j < NT; j++) {
    FragF c;
    frag_zero(c, lane);
    for (int dt = 0; dt < DT; dt++) mma_col_b(c, f.q[dt], sm.kv + j * 16 * D + dt * 16, D, lane);
    store_row(sm.s + j * 16, c, BN, lane);
  }
}

// V replaces K in `kv`; the lane's 16 logits of its row get scale, bias and mask.
static DEV void phase_logits(const Args& a, const Block& b, Shared& sm, const Lane& ln, int n,
                             int lane) {
  for (int j = 0; j < BN; j++) store_decoded(sm.kv, j, lane, ln.vraw[j]);
  int r = lane % ROWS, c0 = (lane / ROWS) * 16;
  Row row = row_of(a, b, r);
  int32_t pos = a.draft ? b.seqlen_k - 1 : b.past + row.tok;
  const uint16_t* rel_row = a.rel;
  if (!a.draft) {
    int32_t e_pad = a.extent + 256;
    // Sheared layout of attention_prologue; decode rows sit in the first 128-row tile.
    int32_t tile_end = (row.tok / 128 + 1) * 128;
    if (tile_end > b.q_len) tile_end = b.q_len;
    int32_t rel_base = (b.past + tile_end + 127) / 128 * 128 - e_pad;
    rel_row = a.rel + row.token * a.stride_rel_t + (int64_t)row.head * e_pad - rel_base;
  }
  float best = -INFINITY;
  for (int c = c0; c < c0 + 16; c++) {
    int32_t key = n * BN + c;
    int32_t dist = pos - key;
    bool visible = dist >= 0 && key < b.seqlen_k;
    float s = sm.s[r * BN + c] * b.qk_scale;
    if (!a.draft && visible && dist < a.extent && row.valid)
      s += half_bits_to_float(rel_row[key]) * LOG2E;
    s = visible ? s : -INFINITY;
    sm.s[r * BN + c] = s;
    best = fmaxf(best, s);
  }
  sm.part_max[lane] = best;
  if (lane == 0) sm.rescale = 0;
}

static DEV float safe_max(const Shared& sm, int r) {
  float m_new = fmaxf(sm.m[r], fmaxf(sm.part_max[r], sm.part_max[r + ROWS]));
  return m_new == -INFINITY ? 0.0f : m_new;
}

// Probabilities relative to the new running max, as fp16.
static DEV void phase_probs(Shared& sm, int lane) {
  int r = lane % ROWS, c0 = (lane / ROWS) * 16;
  float m_safe = safe_max(sm, r);
  float sum = 0.0f;
  for (int c = c0; c < c0 + 16; c++) {
    float p = exp2f(sm.s[r * BN + c] - m_safe);
    sum += p;
    sm.p[r * BN + c] = float_to_half_bits(p);
  }
  sm.part_sum[lane] = sum;
}

static DEV void phase_state(Shared& sm, int lane) {
  if (lane >= ROWS) return;
  int r = lane;
  float m_new = fmaxf(sm.m[r], fmaxf(sm.part_max[r], sm.part_max[r + ROWS]));
  float m_safe = m_new == -INFINITY ? 0.0f : m_new;
  float alpha = exp2f(sm.m[r] - m_safe);
  sm.alpha[r] = alpha;
  sm.l[r] = sm.l[r] * alpha + (sm.part_sum[r] + sm.part_sum[r + ROWS]);
  sm.m[r] = m_new;
  if (alpha != 1.0f) sm.rescale = 1;  // every writer stores the same value
}

static DEV void phase_scale(Shared& sm, int lane) {
  for (int i = lane * 8; i < lane * 8 + 8; i++) sm.scratch[i] *= sm.alpha[i / 16];
}

// acc += P V.
static DEV void phase_pv(Shared& sm, Frags& f, int lane) {
  for (int kt = 0; kt < NT; kt++) load_a_row(f.pa[kt], sm.p + kt * 16, BN, lane);
  for (int dt = 0; dt < DT; dt++)
    for (int kt = 0; kt < NT; kt++)
      mma_row_b(f.acc[dt], f.pa[kt], sm.kv + kt * 16 * D + dt * 16, D, lane);
}

static DEV void phase_out(const Args& a, const Block& b, Shared& sm, int dt, int lane) {
  for (int i = lane * 8; i < lane * 8 + 8; i++) {
    int r = i / 16;
    Row row = row_of(a, b, r);
    if (!row.valid) continue;
    float l = sm.l[r];
    float o = sm.scratch[i] * 256.0f / (l == 0.0f ? 1.0f : l);
    int64_t part = ((int64_t)b.split * a.num_tokens + row.token) * a.num_heads + row.head;
    a.out[part * D + dt * 16 + i % 16] = o;
  }
}

static DEV void phase_lse(const Args& a, const Block& b, Shared& sm, int lane) {
  if (lane >= ROWS) return;
  Row row = row_of(a, b, lane);
  if (!row.valid) return;
  float l = sm.l[lane];
  int64_t part = ((int64_t)b.split * a.num_tokens + row.token) * a.num_heads + row.head;
  a.lse[part] = l == 0.0f ? -INFINITY : sm.m[lane] + log2f(l);
}

#ifdef HOST_EMU
#define PHASE(call) for (int lane = 0; lane < 32; lane++) { Lane& ln = lanes[lane]; (void)ln; call; }
#define KERNEL_HEAD static void run_block(const Args& a, int by, int bz)
#define KERNEL_STATE Shared sm_storage; Shared& sm = sm_storage; Lane lanes[32]; Frags f;
#else
#define PHASE(call) { call; __syncwarp(); }
#define KERNEL_HEAD __global__ void hi_sm75_rel_attention_kernel(const Args a)
#define KERNEL_STATE \
  __shared__ Shared sm; Lane ln; Frags f; \
  const int lane = threadIdx.x, by = blockIdx.y, bz = blockIdx.z;
#endif

KERNEL_HEAD {
  KERNEL_STATE
  Block b;
  if (!block_init(a, by, bz, b)) return;
  PHASE(phase_q(a, b, sm, lane))
  PHASE(phase_q_frags(sm, f, lane))
  for (int i = 0; i < b.num_iters; i++) {
    const int n = b.split + i * a.num_splits;
    PHASE(phase_load(a, b, sm, ln, n, lane))
    PHASE(phase_qk(sm, f, lane))
    PHASE(phase_logits(a, b, sm, ln, n, lane))
    PHASE(phase_probs(sm, lane))
    PHASE(phase_state(sm, lane))
    if (sm.rescale) {
      for (int dt = 0; dt < DT; dt++) {
        PHASE(store_row(sm.scratch, f.acc[dt], 16, lane))
        PHASE(phase_scale(sm, lane))
        PHASE(load_acc_row(f.acc[dt], sm.scratch, 16, lane))
      }
    }
    PHASE(phase_pv(sm, f, lane))
  }
  for (int dt = 0; dt < DT; dt++) {
    PHASE(store_row(sm.scratch, f.acc[dt], 16, lane))
    PHASE(phase_out(a, b, sm, dt, lane))
  }
  PHASE(phase_lse(a, b, sm, lane))
}

extern "C" {
#ifdef HOST_EMU
int hi_sm75_rel_attention(const Args* a, int num_reqs, void*) {
  for (int bz = 0; bz < num_reqs; bz++)
    for (int by = 0; by < a->num_kv_heads * a->num_splits; by++) run_block(*a, by, bz);
  return 0;
}
#else
int hi_sm75_rel_attention(const Args* a, int num_reqs, void* stream) {
  dim3 grid(1, a->num_kv_heads * a->num_splits, num_reqs);
  hi_sm75_rel_attention_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(*a);
  return (int)cudaGetLastError();
}
#endif
int hi_sm75_abi(void) { return 2; }
}
