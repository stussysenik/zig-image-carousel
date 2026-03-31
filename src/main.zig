///! Carousel WASM Module -- Gate 1
///!
///! Computes all transform matrices on the Zig side and writes them into
///! linear memory. The JS host reads these matrices each frame to set
///! WebGL2 uniforms.
///!
///! Exported functions (WASM boundary):
///!   init()                   -- one-time setup
///!   frame(dt: f32)           -- called every RAF tick; dt in seconds
///!   resize(w: u32, h: u32)  -- called on canvas resize
///!   getTransformBufferPtr()  -- pointer to model matrix (4x4 f32)
///!   getViewMatrixPtr()       -- pointer to view matrix
///!   getProjMatrixPtr()       -- pointer to projection matrix
///!   getCardCount()           -- number of active cards
///!
///! Architecture: "Maximum Zig" -- JS is a thin GPU driver that reads
///! shared memory and issues WebGL2 draw calls.
const std = @import("std");
const builtin = @import("builtin");
const math = std.math;

// SIMD math modules (Gate 1)
const vec = @import("vec.zig");
const m4 = @import("mat4.zig");
const Mat4 = m4.Mat4;

// ---------------------------------------------------------------------------
// Dual-target abstraction: WASM uses extern "env" imports; native uses stubs.
// This lets `zig build test` run on the host without any WASM runtime.
// ---------------------------------------------------------------------------
const is_wasm = builtin.cpu.arch == .wasm32;

const env = if (is_wasm) struct {
    extern "env" fn consoleLog(ptr: [*]const u8, len: u32) void;
} else struct {
    fn consoleLog(ptr: [*]const u8, len: u32) void {
        // On native, forward to stderr so tests can observe log output.
        std.debug.print("{s}\n", .{ptr[0..len]});
    }
};

fn log(msg: []const u8) void {
    env.consoleLog(msg.ptr, @intCast(msg.len));
}

// ---------------------------------------------------------------------------
// Shared state written into WASM linear memory
// ---------------------------------------------------------------------------
const MAX_CARDS = 16;

/// Model transforms -- one [16]f32 per card (only index 0 used in Gate 0).
/// Stored as flat f32 arrays in column-major order for direct WebGL upload.
var transform_buffer: [MAX_CARDS][16]f32 = undefined;

/// View matrix (camera) -- flat f32 array for JS to read directly.
var view_matrix: [16]f32 = Mat4.identity.toArray();

/// Projection matrix -- flat f32 array for JS to read directly.
var proj_matrix: [16]f32 = Mat4.identity.toArray();

/// How many cards are currently active.
var card_count: u32 = 1;

/// Accumulated time for animation.
var elapsed: f32 = 0;

/// Current canvas dimensions.
var canvas_width: u32 = 800;
var canvas_height: u32 = 600;

// ---------------------------------------------------------------------------
// Exported API
// ---------------------------------------------------------------------------

/// One-time initialisation. Sets up the view and projection matrices and
/// the initial model transform for the single Gate 0 card.
export fn init() void {
    log("carousel: init");

    // Camera: slightly back along Z so the card is visible.
    // Proper lookAt with forward/right/up basis vectors.
    const eye = vec.vec3(0, 0, 3.0);
    const target = vec.vec3(0, 0, 0);
    const up = vec.vec3(0, 1, 0);
    view_matrix = Mat4.lookAt(eye, target, up).toArray();

    // Default projection (will be recalculated on resize)
    const aspect: f32 = @as(f32, @floatFromInt(canvas_width)) /
        @as(f32, @floatFromInt(canvas_height));
    proj_matrix = Mat4.perspective(math.pi / 4.0, aspect, 0.1, 100.0).toArray();

    // Card starts at identity
    transform_buffer[0] = Mat4.identity.toArray();
    card_count = 1;
    elapsed = 0;

    log("carousel: init complete");
}

/// Called every requestAnimationFrame. `dt` is the time delta in seconds.
/// Gate 0: slowly rotates the single card around the Y axis.
export fn frame(dt: f32) void {
    elapsed += dt;

    // Gentle continuous Y rotation
    const model = Mat4.rotateY(elapsed * 0.5);
    transform_buffer[0] = model.toArray();
}

/// Called when the canvas is resized. Updates the projection matrix.
export fn resize(w: u32, h: u32) void {
    canvas_width = w;
    canvas_height = h;
    const aspect: f32 = @as(f32, @floatFromInt(w)) /
        @as(f32, @floatFromInt(if (h == 0) 1 else h));
    proj_matrix = Mat4.perspective(math.pi / 4.0, aspect, 0.1, 100.0).toArray();
}

/// Returns a pointer to the transform buffer so JS can read model matrices
/// directly from WASM linear memory.
export fn getTransformBufferPtr() [*]const f32 {
    return @ptrCast(&transform_buffer);
}

/// Returns a pointer to the view matrix.
export fn getViewMatrixPtr() [*]const f32 {
    return @ptrCast(&view_matrix);
}

/// Returns a pointer to the projection matrix.
export fn getProjMatrixPtr() [*]const f32 {
    return @ptrCast(&proj_matrix);
}

/// Returns the number of active cards.
export fn getCardCount() u32 {
    return card_count;
}

// ---------------------------------------------------------------------------
// Tests -- run on native target via `zig build test`
// ---------------------------------------------------------------------------

// Re-export tests from submodules so `zig build test` runs them all.
test {
    _ = @import("vec.zig");
    _ = @import("mat4.zig");
}

test "identity matrix is correct" {
    const id = Mat4.identity.toArray();
    // Diagonal should be 1
    try std.testing.expectEqual(@as(f32, 1), id[0]);
    try std.testing.expectEqual(@as(f32, 1), id[5]);
    try std.testing.expectEqual(@as(f32, 1), id[10]);
    try std.testing.expectEqual(@as(f32, 1), id[15]);
    // Off-diagonal should be 0
    try std.testing.expectEqual(@as(f32, 0), id[1]);
    try std.testing.expectEqual(@as(f32, 0), id[4]);
}

test "rotationY at 0 is identity" {
    const r = Mat4.rotateY(0).toArray();
    try std.testing.expectApproxEqAbs(@as(f32, 1), r[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), r[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), r[10], 1e-6);
}

test "mat4Mul identity * identity = identity" {
    const result = Mat4.identity.mul(Mat4.identity).toArray();
    const id = Mat4.identity.toArray();
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(id[i], result[i], 1e-6);
    }
}

test "mat4Mul with translation" {
    const t = Mat4.translate(1, 2, 3);
    const result = Mat4.identity.mul(t).toArray();
    // Translation components in column 3
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), result[13], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), result[14], 1e-6);
}

test "perspective produces non-zero matrix" {
    const p = Mat4.perspective(math.pi / 4.0, 1.333, 0.1, 100.0).toArray();
    // [0] and [5] should be non-zero (focal length terms)
    try std.testing.expect(p[0] != 0);
    try std.testing.expect(p[5] != 0);
    // [15] should be 0 for a perspective matrix
    try std.testing.expectApproxEqAbs(@as(f32, 0), p[15], 1e-6);
    // [11] should be -1 (perspective divide)
    try std.testing.expectApproxEqAbs(@as(f32, -1), p[11], 1e-6);
}

test "init sets card_count to 1" {
    init();
    try std.testing.expectEqual(@as(u32, 1), getCardCount());
}

test "frame advances rotation" {
    init();
    frame(1.0);
    // After 1 second at 0.5 rad/s, the model[0] should be cos(0.5)
    const ptr = getTransformBufferPtr();
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 0.5)), ptr[0], 1e-5);
}

test "resize updates projection" {
    init();
    const old_proj = getProjMatrixPtr()[0];
    resize(1920, 1080);
    const new_proj = getProjMatrixPtr()[0];
    // Different aspect ratio should change projection[0]
    try std.testing.expect(old_proj != new_proj);
}
