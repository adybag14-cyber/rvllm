import struct, torch

def rust_fp8_encode(v):
    v = float(v)
    if v != v: return 0x7f
    bits_i = struct.unpack("I", struct.pack("f", v))[0]
    s_bit = 0x80 if (bits_i >> 31) else 0
    a = abs(v)
    if a == 0: return s_bit
    if a > 448: return s_bit | 0x7e
    bits = struct.unpack("I", struct.pack("f", a))[0]
    exp32 = ((bits >> 23) & 0xff) - 127
    mant32 = bits & 0x7fffff
    exp8 = exp32 + 7
    if exp8 <= 0:
        shift = 1 - exp8
        full = mant32 | (1 << 23)
        rshift = 20 + shift
        m = full >> rshift
        rb = (full >> (rshift - 1)) & 1 if rshift > 0 else 0
        sticky = 1 if rshift > 1 and (full & ((1 << (rshift - 1)) - 1)) else 0
        m += rb & (sticky | (m & 1))
        if m >= 8: return s_bit | 0x08
        return s_bit | (m & 0x07)
    trunc = mant32 >> 20
    rb = (mant32 >> 19) & 1
    sticky = 1 if (mant32 & 0x7ffff) else 0
    m = trunc + (rb & (sticky | (trunc & 1)))
    if m >= 8:
        exp8 += 1
        if exp8 > 15: return s_bit | 0x7e
        return s_bit | ((exp8 & 0x0f) << 3)
    if exp8 > 15: return s_bit | 0x7e
    return s_bit | ((exp8 & 0x0f) << 3) | (m & 0x07)

vals = [-10.071, -80.569, 9.352, -74.814, -63.304, -25.897, -4.316, -20.142]
mismatches = 0
for v in vals:
    rust_b = rust_fp8_encode(v)
    nv_b = torch.tensor([v], device="cuda").to(torch.float8_e4m3fn).view(torch.uint8).item()
    match = "OK" if rust_b == nv_b else "MISMATCH"
    if rust_b != nv_b: mismatches += 1
    print(f"  {v:8.3f}: rust=0x{rust_b:02x}({rust_b:3d}) nvidia=0x{nv_b:02x}({nv_b:3d}) {match}")

print(f"\nMismatches: {mismatches}/{len(vals)}")
print(f"\nGPU bytes from rvLLM probe:  [209, 233, 81, 233, 230, 220, 200, 217]")
print(f"Expected (NVIDIA encode):    [210, 234, 81, 233, 232, 221, 201, 218]")
print(f"Rust encode (Python):        [{', '.join(str(rust_fp8_encode(v)) for v in vals)}]")
