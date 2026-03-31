///! SIMD Column-Major Mat4 -- Gate 1
///!
///! A 4x4 matrix stored as four column Vec4 values. Column-major layout
///! matches OpenGL/WebGL uniform conventions directly, so the JS host can
///! read the raw f32 array without transposing.
///!
///! SIMD strategy: each column is a @Vector(4, f32), so matrix-vector
///! multiply becomes four @splat broadcasts + lane-wise multiply-add.
///! On WASM this maps to i32x4.splat + f32x4.mul + f32x4.add.
const std = @import("std");
const math = std.math;
const vec = @import("vec.zig");
const Vec4 = vec.Vec4;
const Vec3 = vec.Vec3;

/// A 4x4 column-major matrix backed by four SIMD column vectors.
///
/// Memory layout (matches OpenGL column-major convention):
///   cols[0] = [m00, m10, m20, m30]  -- first column
///   cols[1] = [m01, m11, m21, m31]  -- second column
///   cols[2] = [m02, m12, m22, m32]  -- third column
///   cols[3] = [m03, m13, m23, m33]  -- fourth column
pub const Mat4 = struct {
    cols: [4]Vec4,

    // -------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------

    /// The 4x4 identity matrix.
    pub const identity = Mat4{
        .cols = .{
            .{ 1, 0, 0, 0 },
            .{ 0, 1, 0, 0 },
            .{ 0, 0, 1, 0 },
            .{ 0, 0, 0, 1 },
        },
    };

    // -------------------------------------------------------------------
    // Core operations
    // -------------------------------------------------------------------

    /// Matrix-vector multiply: m * v.
    ///
    /// Each column of m is scaled by the corresponding component of v,
    /// then the four scaled columns are summed. This maps cleanly to
    /// four SIMD broadcasts + four lane-wise multiplies + three adds.
    pub fn mulVec(m: Mat4, v: Vec4) Vec4 {
        // Broadcast each component of v across all 4 lanes
        const x: Vec4 = @splat(v[0]);
        const y: Vec4 = @splat(v[1]);
        const z: Vec4 = @splat(v[2]);
        const w: Vec4 = @splat(v[3]);
        return m.cols[0] * x + m.cols[1] * y + m.cols[2] * z + m.cols[3] * w;
    }

    /// Matrix-matrix multiply: a * b.
    ///
    /// Transforms each column of b by a. Result column i = a * b.cols[i].
    pub fn mul(a: Mat4, b: Mat4) Mat4 {
        return .{
            .cols = .{
                a.mulVec(b.cols[0]),
                a.mulVec(b.cols[1]),
                a.mulVec(b.cols[2]),
                a.mulVec(b.cols[3]),
            },
        };
    }

    // -------------------------------------------------------------------
    // Transform builders
    // -------------------------------------------------------------------

    /// Translation matrix: shifts points by (tx, ty, tz).
    pub fn translate(tx: f32, ty: f32, tz: f32) Mat4 {
        return .{
            .cols = .{
                .{ 1, 0, 0, 0 },
                .{ 0, 1, 0, 0 },
                .{ 0, 0, 1, 0 },
                .{ tx, ty, tz, 1 },
            },
        };
    }

    /// Uniform scale matrix: scales equally in all axes.
    pub fn scaleUniform(s: f32) Mat4 {
        return .{
            .cols = .{
                .{ s, 0, 0, 0 },
                .{ 0, s, 0, 0 },
                .{ 0, 0, s, 0 },
                .{ 0, 0, 0, 1 },
            },
        };
    }

    /// Rotation around the Y axis by `angle` radians.
    ///
    /// Right-hand rule: looking down -Y, positive angle rotates from +X
    /// toward +Z.
    pub fn rotateY(angle: f32) Mat4 {
        const c = @cos(angle);
        const s = @sin(angle);
        return .{
            .cols = .{
                .{ c, 0, -s, 0 },
                .{ 0, 1, 0, 0 },
                .{ s, 0, c, 0 },
                .{ 0, 0, 0, 1 },
            },
        };
    }

    /// Rotation around the X axis by `angle` radians.
    ///
    /// Right-hand rule: looking down -X, positive angle rotates from +Y
    /// toward +Z.
    pub fn rotateX(angle: f32) Mat4 {
        const c = @cos(angle);
        const s = @sin(angle);
        return .{
            .cols = .{
                .{ 1, 0, 0, 0 },
                .{ 0, c, s, 0 },
                .{ 0, -s, c, 0 },
                .{ 0, 0, 0, 1 },
            },
        };
    }

    /// Symmetric perspective projection matrix.
    ///
    /// Produces a standard OpenGL-style perspective matrix where:
    ///   - fov_y is the vertical field of view in radians
    ///   - aspect is width / height
    ///   - near/far are the clip plane distances (both positive)
    ///
    /// The resulting clip-space has z in [-1, 1] (OpenGL convention).
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) Mat4 {
        const f = 1.0 / @tan(fov_y / 2.0);
        const range_inv = 1.0 / (near - far);
        return .{
            .cols = .{
                .{ f / aspect, 0, 0, 0 },
                .{ 0, f, 0, 0 },
                .{ 0, 0, (far + near) * range_inv, -1 },
                .{ 0, 0, 2 * far * near * range_inv, 0 },
            },
        };
    }

    /// Standard look-at view matrix.
    ///
    /// Constructs an orthonormal basis from the eye position, target point,
    /// and world-up vector. The resulting matrix transforms world-space
    /// points into camera/view space.
    ///
    /// Implementation:
    ///   forward = normalize(eye - target)   (camera looks along -forward)
    ///   right   = normalize(up × forward)
    ///   cam_up  = forward × right
    ///   Result  = rotation * translation(-eye)
    pub fn lookAt(eye: Vec3, target: Vec3, up: Vec3) Mat4 {
        const forward = vec.normalize3(eye - target);
        const right = vec.normalize3(vec.cross(up, forward));
        const cam_up = vec.cross(forward, right);

        // The view matrix is the inverse of the camera's model matrix.
        // For an orthonormal rotation R and translation T:
        //   inverse = R^T * (-T)
        // We combine this into a single matrix where the last column
        // contains the dot products with -eye.
        const neg_dot_r = -vec.dot3(right, eye);
        const neg_dot_u = -vec.dot3(cam_up, eye);
        const neg_dot_f = -vec.dot3(forward, eye);

        return .{
            .cols = .{
                .{ right[0], cam_up[0], forward[0], 0 },
                .{ right[1], cam_up[1], forward[1], 0 },
                .{ right[2], cam_up[2], forward[2], 0 },
                .{ neg_dot_r, neg_dot_u, neg_dot_f, 1 },
            },
        };
    }

    // -------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------

    /// Flatten to a 16-element f32 array in column-major order.
    /// Useful for writing directly into a WebGL uniform buffer.
    pub fn toArray(m: Mat4) [16]f32 {
        // Each @Vector(4,f32) column is laid out contiguously,
        // so we just concatenate the four columns.
        var result: [16]f32 = undefined;
        inline for (0..4) |col| {
            inline for (0..4) |row| {
                result[col * 4 + row] = m.cols[col][row];
            }
        }
        return result;
    }

    /// Write the matrix into a mutable f32 buffer at a byte offset.
    /// `offset` is the index into `buf` (in f32 elements, not bytes).
    pub fn writeTo(m: Mat4, buf: []f32, offset: usize) void {
        const arr = m.toArray();
        @memcpy(buf[offset..][0..16], &arr);
    }
};

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "identity * vec = vec" {
    const v = vec.vec4(1, 2, 3, 1);
    const result = Mat4.identity.mulVec(v);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[3], 1e-6);
}

test "identity * identity = identity" {
    const result = Mat4.identity.mul(Mat4.identity);
    const arr = result.toArray();
    const id_arr = Mat4.identity.toArray();
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(id_arr[i], arr[i], 1e-6);
    }
}

test "translate point" {
    const t = Mat4.translate(10, 20, 30);
    const point = vec.vec4(1, 2, 3, 1); // w=1 means it's a point
    const result = t.mulVec(point);
    try std.testing.expectApproxEqAbs(@as(f32, 11), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 33), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[3], 1e-6);
}

test "translate does not affect directions (w=0)" {
    const t = Mat4.translate(10, 20, 30);
    const dir = vec.vec4(1, 0, 0, 0); // w=0 means direction
    const result = t.mulVec(dir);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[2], 1e-6);
}

test "perspective has -1 at [2][3]" {
    const p = Mat4.perspective(math.pi / 4.0, 1.333, 0.1, 100.0);
    // [2][3] is the third column, fourth row -> cols[2][3]
    try std.testing.expectApproxEqAbs(@as(f32, -1), p.cols[2][3], 1e-6);
    // [3][3] should be 0 for a perspective matrix
    try std.testing.expectApproxEqAbs(@as(f32, 0), p.cols[3][3], 1e-6);
}

test "perspective non-zero focal terms" {
    const p = Mat4.perspective(math.pi / 4.0, 1.333, 0.1, 100.0);
    try std.testing.expect(p.cols[0][0] != 0);
    try std.testing.expect(p.cols[1][1] != 0);
}

test "lookAt camera at z=3" {
    const eye = vec.vec3(0, 0, 3);
    const target = vec.vec3(0, 0, 0);
    const up = vec.vec3(0, 1, 0);
    const v = Mat4.lookAt(eye, target, up);

    // Transform the eye point -- should map to origin in view space
    const eye_h = vec.vec4(0, 0, 3, 1);
    const result = v.mulVec(eye_h);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[2], 1e-5);

    // Transform the origin -- should be at (0, 0, -3) in view space
    const origin_h = vec.vec4(0, 0, 0, 1);
    const origin_result = v.mulVec(origin_h);
    try std.testing.expectApproxEqAbs(@as(f32, 0), origin_result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), origin_result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -3), origin_result[2], 1e-5);
}

test "toArray column-major layout" {
    // Build a matrix with distinct values to verify ordering
    const m = Mat4{
        .cols = .{
            .{ 1, 2, 3, 4 },
            .{ 5, 6, 7, 8 },
            .{ 9, 10, 11, 12 },
            .{ 13, 14, 15, 16 },
        },
    };
    const arr = m.toArray();
    // Column 0
    try std.testing.expectEqual(@as(f32, 1), arr[0]);
    try std.testing.expectEqual(@as(f32, 2), arr[1]);
    try std.testing.expectEqual(@as(f32, 3), arr[2]);
    try std.testing.expectEqual(@as(f32, 4), arr[3]);
    // Column 1
    try std.testing.expectEqual(@as(f32, 5), arr[4]);
    try std.testing.expectEqual(@as(f32, 6), arr[5]);
    try std.testing.expectEqual(@as(f32, 7), arr[6]);
    try std.testing.expectEqual(@as(f32, 8), arr[7]);
    // Column 3
    try std.testing.expectEqual(@as(f32, 13), arr[12]);
    try std.testing.expectEqual(@as(f32, 14), arr[13]);
    try std.testing.expectEqual(@as(f32, 15), arr[14]);
    try std.testing.expectEqual(@as(f32, 16), arr[15]);
}

test "rotateY at 0 is identity" {
    const r = Mat4.rotateY(0);
    const id = Mat4.identity;
    const r_arr = r.toArray();
    const id_arr = id.toArray();
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(id_arr[i], r_arr[i], 1e-6);
    }
}

test "rotateX at 0 is identity" {
    const r = Mat4.rotateX(0);
    const id = Mat4.identity;
    const r_arr = r.toArray();
    const id_arr = id.toArray();
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(id_arr[i], r_arr[i], 1e-6);
    }
}

test "scaleUniform scales all axes" {
    const s = Mat4.scaleUniform(2);
    const point = vec.vec4(1, 2, 3, 1);
    const result = s.mulVec(point);
    try std.testing.expectApproxEqAbs(@as(f32, 2), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[3], 1e-6);
}

test "writeTo writes at offset" {
    var buf: [32]f32 = .{0} ** 32;
    const m = Mat4.translate(1, 2, 3);
    m.writeTo(&buf, 8);
    // Translation is in column 3 (indices 12..15 relative to start)
    // At offset 8, that's buf[8+12..8+15] = buf[20..23]
    try std.testing.expectApproxEqAbs(@as(f32, 1), buf[20], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), buf[21], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), buf[22], 1e-6);
}

test "mul with translation" {
    const t = Mat4.translate(1, 2, 3);
    const result = Mat4.identity.mul(t);
    const arr = result.toArray();
    // Translation components in column 3
    try std.testing.expectApproxEqAbs(@as(f32, 1), arr[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), arr[13], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), arr[14], 1e-6);
}
