#include <metal_stdlib>
using namespace metal;

kernel void dequantize_q2_k_f32(
    device const uchar* data      [[buffer(0)]],
    device float* out             [[buffer(1)]],
    constant uint& num_blocks     [[buffer(2)]],
    constant uint& numel          [[buffer(3)]],
    uint block_idx [[thread_position_in_grid]]
) {
    if (block_idx >= num_blocks) return;

    device const uchar* block = data + block_idx * 84;
    uint out_base = block_idx * 256;

    device const uchar* sc = block;
    device const uchar* q = block + 16;
    half d_h = as_type<half>(ushort(ushort(block[80]) | (ushort(block[81]) << 8)));
    half min_h = as_type<half>(ushort(ushort(block[82]) | (ushort(block[83]) << 8)));
    float d = float(d_h);
    float min = float(min_h);

    uint is = 0;
    uint out_idx = out_base;

    for (uint n = 0; n < 256; n += 128) {
        uint shift = 0;
        for (uint j = 0; j < 4; ++j) {
            uchar sc_val1 = sc[is++];
            float dl1 = d * float(sc_val1 & 0x0F);
            float ml1 = min * float(sc_val1 >> 4);
            for (uint l = 0; l < 16; ++l) {
                if (out_idx >= numel) return;
                out[out_idx++] = dl1 * float(char((q[l] >> shift) & 3)) - ml1;
            }

            uchar sc_val2 = sc[is++];
            float dl2 = d * float(sc_val2 & 0x0F);
            float ml2 = min * float(sc_val2 >> 4);
            for (uint l = 0; l < 16; ++l) {
                if (out_idx >= numel) return;
                out[out_idx++] = dl2 * float(char((q[l + 16] >> shift) & 3)) - ml2;
            }

            shift += 2;
        }
        q += 32;
    }
}
