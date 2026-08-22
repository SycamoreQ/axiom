kernel void dequantize_q4_k_f32(
    device const uchar* data      [[buffer(0)]],
    device float* out             [[buffer(1)]],
    constant uint& num_blocks     [[buffer(2)]],
    constant uint& numel          [[buffer(3)]],
    uint block_idx [[thread_position_in_grid]]
) {
    if (block_idx >= num_blocks) return;

    device const uchar* block = data + block_idx * 144;
    uint out_base = block_idx * 256;

    half d_h = as_type<half>(ushort(ushort(block[0]) | (ushort(block[1]) << 8)));
    half dmin_h = as_type<half>(ushort(ushort(block[2]) | (ushort(block[3]) << 8)));
    float d = float(d_h);
    float dmin = float(dmin_h);

    device const uchar* sc = block + 4;
    device const uchar* qs = block + 16;

    uchar scales[8];
    uchar mins[8];
    scales[0] = sc[0] & 0x3F;
    scales[1] = sc[1] & 0x3F;
    scales[2] = sc[2] & 0x3F;
    scales[3] = sc[3] & 0x3F;
    scales[4] = (sc[8] & 0x0F) | ((sc[0] >> 6) << 4);
    scales[5] = (sc[9] & 0x0F) | ((sc[1] >> 6) << 4);
    scales[6] = (sc[10] & 0x0F) | ((sc[2] >> 6) << 4);
    scales[7] = (sc[11] & 0x0F) | ((sc[3] >> 6) << 4);

    mins[0] = sc[4] & 0x3F;
    mins[1] = sc[5] & 0x3F;
    mins[2] = sc[6] & 0x3F;
    mins[3] = sc[7] & 0x3F;
    mins[4] = (sc[8] >> 4) | ((sc[4] >> 6) << 4);
    mins[5] = (sc[9] >> 4) | ((sc[5] >> 6) << 4);
    mins[6] = (sc[10] >> 4) | ((sc[6] >> 6) << 4);
    mins[7] = (sc[11] >> 4) | ((sc[7] >> 6) << 4);

    uint out_idx = out_base;
    uint is = 0;
    for (uint c = 0; c < 4; c++) {
        device const uchar* q = qs + c * 32;
        float d1 = d * float(scales[is]);
        float m1 = dmin * float(mins[is]);
        float d2 = d * float(scales[is + 1]);
        float m2 = dmin * float(mins[is + 1]);

        for (uint l = 0; l < 32; l++) {
            if (out_idx >= numel) return;
            out[out_idx] = d1 * float(q[l] & 0x0F) - m1;
            out_idx++;
        }
        for (uint l = 0; l < 32; l++) {
            if (out_idx >= numel) return;
            out[out_idx] = d2 * float(q[l] >> 4) - m2;
            out_idx++;
        }
        is += 2;
    }
}
