///! SIMD Vector Math -- Gate 1
///!
///! Provides Vec3 and Vec4 types backed by @Vector(4, f32) for SIMD
///! acceleration on both WASM (SIMD128) and native targets.
///!
///! Vec3 is stored as @Vector(4, f32) with w=0 for alignment. This lets
///! the same SIMD lanes handle both 3D and 4D operations without
///! type-punning or extra copies.
const std = @import("std");
const math = std.math;

/// A 4-component SIMD vector: [x, y, z, w].
/// Used for both homogeneous coordinates (w=1 for points, w=0 for directions)
/// and as the column type inside Mat4.
pub const Vec4 = @Vector(4, f32);

/// Alias for clarity -- a 3D vector stored in 4 lanes with w=0.
/// Using the same underlying type avoids conversion overhead and lets
/// dot/cross share SIMD hardware with Mat4 operations.
pub const Vec3 = @Vector(4, f32);

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

/// Create a Vec3 (direction/free vector) with w=0.
pub fn vec3(x: f32, y: f32, z: f32) Vec3 {
    return .{ x, y, z, 0 };
}

/// Create a Vec4 (homogeneous point or direction) with explicit w.
pub fn vec4(x: f32, y: f32, z: f32, w: f32) Vec4 {
    return .{ x, y, z, w };
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

/// Dot product of the xyz components only (ignores w).
/// Uses SIMD multiply + horizontal add pattern.
pub fn dot3(a: Vec3, b: Vec3) f32 {
    const prod = a * b;
    return prod[0] + prod[1] + prod[2];
}

/// Cross product of two Vec3 values. Result has w=0.
///
/// Formula: a × b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
///
/// Implemented via swizzle: we rearrange lanes so a single SIMD
/// multiply-subtract produces all three components at once.
pub fn cross(a: Vec3, b: Vec3) Vec3 {
    // Swizzle both inputs into yzx and zxy orderings so a single
    // lane-wise multiply-subtract produces the cross product directly
    // in xyz order -- no final re-shuffle needed.
    //   a_yzx * b_zxy = [a.y*b.z, a.z*b.x, a.x*b.y, _]
    //   a_zxy * b_yzx = [a.z*b.y, a.x*b.z, a.y*b.x, _]
    //   difference    = cross product in xyz order
    const a_yzx = @shuffle(f32, a, undefined, [4]i32{ 1, 2, 0, 3 });
    const b_yzx = @shuffle(f32, b, undefined, [4]i32{ 1, 2, 0, 3 });
    const a_zxy = @shuffle(f32, a, undefined, [4]i32{ 2, 0, 1, 3 });
    const b_zxy = @shuffle(f32, b, undefined, [4]i32{ 2, 0, 1, 3 });

    const result = a_yzx * b_zxy - a_zxy * b_yzx;
    // Zero out w to keep it clean
    return .{ result[0], result[1], result[2], 0 };
}

/// Euclidean length of the xyz components.
pub fn length3(v: Vec3) f32 {
    return @sqrt(dot3(v, v));
}

/// Normalize the xyz components to unit length. Returns zero vector if
/// the input length is near zero (avoids division by zero / NaN).
pub fn normalize3(v: Vec3) Vec3 {
    const len = length3(v);
    if (len < 1e-10) return .{ 0, 0, 0, 0 };
    const inv: Vec4 = @splat(1.0 / len);
    // Multiply then zero out w to keep it clean
    const result = v * inv;
    return .{ result[0], result[1], result[2], 0 };
}

/// Scalar multiply: v * s (broadcast scalar across all 4 lanes).
pub fn scale(v: Vec4, s: f32) Vec4 {
    const sv: Vec4 = @splat(s);
    return v * sv;
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "dot3 basic" {
    const a = vec3(1, 2, 3);
    const b = vec3(4, 5, 6);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dot3(a, b), 1e-6);
}

test "dot3 ignores w" {
    const a = vec4(1, 2, 3, 100);
    const b = vec4(4, 5, 6, 200);
    // w components (100, 200) should be ignored
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dot3(a, b), 1e-6);
}

test "cross x cross y = z" {
    const x = vec3(1, 0, 0);
    const y = vec3(0, 1, 0);
    const result = cross(x, y);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[3], 1e-6);
}

test "cross y cross z = x" {
    const y = vec3(0, 1, 0);
    const z = vec3(0, 0, 1);
    const result = cross(y, z);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[2], 1e-6);
}

test "cross anti-commutativity" {
    const a = vec3(1, 2, 3);
    const b = vec3(4, 5, 6);
    const ab = cross(a, b);
    const ba = cross(b, a);
    // a×b = -(b×a)
    try std.testing.expectApproxEqAbs(-ab[0], ba[0], 1e-6);
    try std.testing.expectApproxEqAbs(-ab[1], ba[1], 1e-6);
    try std.testing.expectApproxEqAbs(-ab[2], ba[2], 1e-6);
}

test "normalize (3,4,0) -> (0.6,0.8,0)" {
    const v = vec3(3, 4, 0);
    const n = normalize3(v);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), n[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), n[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), n[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), n[3], 1e-6);
}

test "normalize zero vector returns zero" {
    const v = vec3(0, 0, 0);
    const n = normalize3(v);
    try std.testing.expectApproxEqAbs(@as(f32, 0), n[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), n[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), n[2], 1e-6);
}

test "length of (3,4,0) is 5" {
    const v = vec3(3, 4, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), length3(v), 1e-6);
}

test "length of unit vector is 1" {
    const v = normalize3(vec3(1, 1, 1));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), length3(v), 1e-6);
}

test "scale multiplies all components" {
    const v = vec4(1, 2, 3, 4);
    const result = scale(v, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 3), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12), result[3], 1e-6);
}
