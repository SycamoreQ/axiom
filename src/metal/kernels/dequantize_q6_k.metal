#include <metal_stdlib>
using namespace metal;

kernel void dequantize_q6_k_f32(
    device const uchar* data      [[buffer(0)]],
    device float* out             [[buffer(1)]],
    constant uint& num_blocks     [[buffer(2)]],
    constant uint& numel          [[buffer(3)]],
    uint block_idx [[thread_position_in_grid]]
) {
    if (block_idx >= num_blocks) return;

    device const uchar* block = data + block_idx * 210;
    uint block_base = block_idx * 256;

    half d_h = as_type<half>(ushort(ushort(block[208]) | (ushort(block[209]) << 8)));
    float d = float(d_h);

    device const uchar* ql_all = block;
    device const uchar* qh_all = block + 128;
    device const char* sc_all = (device const char*)(block + 192);

    for (uint g = 0; g < 2; g++) {
        device const uchar* ql = ql_all + g * 64;
        device const uchar* qh = qh_all + g * 32;
        device const char* sc = sc_all + g * 8;
        uint base = block_base + g * 128;

        for (uint l = 0; l < 32; l++) {
            uint is = l / 16;

            int q1 = int((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = int((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            float s1 = float(sc[is]);
            float s2 = float(sc[is + 2]);
            float s3 = float(sc[is + 4]);
            float s4 = float(sc[is + 6]);

            if (base + l < numel) {
                out[base + l] = d * s1 * float(q1);
            }
            if (base + l + 32 < numel) {
                out[base + l + 32] = d * s2 * float(q2);
            }
            if (base + l + 64 < numel) {
                out[base + l + 64] = d * s3 * float(q3);
            }
            if (base + l + 96 < numel) {
                out[base + l + 96] = d * s4 * float(q4);
            }
        }
    }
}
