import numpy as np

def pack_int2(data: np.ndarray) -> np.ndarray:
    flat = data.flatten()
    pad_size = (4 - (len(flat) % 4)) % 4
    if pad_size > 0:
        flat = np.append(flat, np.zeros(pad_size, dtype=np.int8))
    unsigned = (flat.astype(np.int8) + 2).astype(np.uint8) & 0x03
    packed = (unsigned[0::4] << 6) | (unsigned[1::4] << 4) | \
             (unsigned[2::4] << 2) | unsigned[3::4]
    return packed.astype(np.uint8)

def unpack_int2(packed: np.ndarray, original_size: int) -> np.ndarray:
    v0 = (packed >> 6) & 0x03
    v1 = (packed >> 4) & 0x03
    v2 = (packed >> 2) & 0x03
    v3 = packed & 0x03
    unpacked = np.empty(len(packed) * 4, dtype=np.int8)
    unpacked[0::4] = v0
    unpacked[1::4] = v1
    unpacked[2::4] = v2
    unpacked[3::4] = v3
    unpacked = unpacked.astype(np.int8) - 2
    return unpacked[:original_size]

# Test
original = np.array([-2, -1, 0, 1, 1, 0, -1, -2, 0, 0, 1, -2], dtype=np.int8)
print(f"Original: {original}")

packed = pack_int2(original)
print(f"Packed (hex): {[hex(b) for b in packed]}")

unpacked = unpack_int2(packed, len(original))
print(f"Unpacked: {unpacked}")

assert np.array_equal(original, unpacked)
print("✅ Test passed!")
