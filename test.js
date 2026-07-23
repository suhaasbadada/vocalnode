const leftover = new Uint8Array([1, 2]);
const value = new Uint8Array([3, 4, 5]);
const combined = new Uint8Array(leftover.length + value.length);
combined.set(leftover);
combined.set(value, leftover.length);
const safeLength = combined.length - (combined.length % 2);
const int16Array = new Int16Array(combined.buffer, 0, safeLength / 2);
console.log(int16Array.length, combined.buffer.byteLength, safeLength);
