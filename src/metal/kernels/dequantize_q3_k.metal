#include <metal_stdlib>
using namespace metal;

kernel void dequantize_q3_k_f32(
    device const uchar* data      [[buffer(0)]],
    device float* out             [[buffer(1)]],
    constant uint& num_blocks     [[buffer(2)]],
    constant uint& numel          [[buffer(3)]],
    uint block_idx [[thread_position_in_grid]]
) {
    if (block_idx >= num_blocks) return;

    device const uchar* block = data + block_idx * 110;
    uint out_base = block_idx * 256;

    device const uchar* hm = block;
    device const uchar* q = block + 32;
    device const uchar* sc = block + 96;
    half d_h = as_type<half>(ushort(ushort(block[108]) | (ushort(block[109]) << 8)));
    float d_all = float(d_h);

    uint kmask1 = 0x03030303;
    uint kmask2 = 0x0f0f0f0f;

    uint aux[4];
    aux[0] = uint(sc[0]) | (uint(sc[1]) << 8) | (uint(sc[2]) << 16) | (uint(sc[3]) << 24);
    aux[1] = uint(sc[4]) | (uint(sc[5]) << 8) | (uint(sc[6]) << 16) | (uint(sc[7]) << 24);
    aux[2] = uint(sc[8]) | (uint(sc[9]) << 8) | (uint(sc[10]) << 16) | (uint(sc[11]) << 24);

    uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    char scales[16];
    for (uint i = 0; i < 4; i++) {
        scales[i * 4 + 0] = char(aux[i] & 0xFF);
        scales[i * 4 + 1] = char((aux[i] >> 8) & 0xFF);
        scales[i * 4 + 2] = char((aux[i] >> 16) & 0xFF);
        scales[i * 4 + 3] = char((aux[i] >> 24) & 0xFF);
    }

    uint is = 0;
    uint m = 1;
    uint out_idx = out_base;

    for (uint n = 0; n < 256; n += 128) {
        uint shift = 0;
        for (uint j = 0; j < 4; ++j) {
            float dl1 = d_all * float(scales[is++] - 32);
            for (uint l = 0; l < 16; ++l) {
                if (out_idx >= numel) return;
                int val = int((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                out[out_idx++] = dl1 * float(val);
            }

            float dl2 = d_all * float(scales[is++] - 32);
            for (uint l = 0; l < 16; ++l) {
                if (out_idx >= numel) return;
                int val = int((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                out[out_idx++] = dl2 * float(val);
            }

            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}
