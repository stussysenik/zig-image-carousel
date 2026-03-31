///! Carousel WASM Module -- Gate 1
///!
///! Computes all transform matrices on the Zig side and writes them into
///! linear memory. The JS host reads these matrices each frame to set
///! WebGL2 instanced attributes for rendering a 12-card depth stack.
///!
///! Exported functions (WASM boundary):
///!   init()                   -- one-time setup
///!   frame(dt: f32)           -- called every RAF tick; dt in seconds
///!   resize(w: u32, h: u32)  -- called on canvas resize
///!   getTransformBufferPtr()  -- pointer to model matrices (MAX_CARDS x 4x4 f32)
///!   getOpacityBufferPtr()    -- pointer to per-card opacities
///!   getTexIndexBufferPtr()   -- pointer to per-card texture layer indices
///!   getViewMatrixPtr()       -- pointer to view matrix
///!   getProjMatrixPtr()       -- pointer to projection matrix
///!   getCardCount()           -- number of visible cards
///!
///! Architecture: "Maximum Zig" -- JS is a thin GPU driver that reads
///! shared memory and issues WebGL2 instanced draw calls.
const std = @import("std");
const builtin = @import("builtin");
const math = std.math;

// SIMD math modules (Gate 1)
const vec = @import("vec.zig");
const m4 = @import("mat4.zig");
const Mat4 = m4.Mat4;

// Depth stack layout engine (Gate 1)
const stack_mod = @import("stack.zig");
const Stack = stack_mod.Stack;
const MAX_CARDS = stack_mod.MAX_CARDS;

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

/// Model transforms -- one [16]f32 per card, flat for direct WebGL upload.
var transform_buffer: [MAX_CARDS * 16]f32 = undefined;

/// Per-card opacity values (one f32 per visible card).
var opacity_buffer: [MAX_CARDS]f32 = undefined;

/// Per-card texture array layer indices (as f32 for vertex attrib).
var tex_index_buffer: [MAX_CARDS]f32 = undefined;

/// View matrix (camera) -- flat f32 array for JS to read directly.
var view_matrix: [16]f32 = Mat4.identity.toArray();

/// Projection matrix -- flat f32 array for JS to read directly.
var proj_matrix: [16]f32 = Mat4.identity.toArray();

/// Number of currently visible cards (set by computeLayout each frame).
var visible_count: u32 = 0;

/// The depth stack instance -- manages layout for all cards.
var depth_stack: Stack = Stack.init(MAX_CARDS);

/// Current canvas dimensions.
var canvas_width: u32 = 800;
var canvas_height: u32 = 600;

// ---------------------------------------------------------------------------
// Exported API
// ---------------------------------------------------------------------------

/// One-time initialisation. Sets up the camera, projection, and computes
/// the initial depth stack layout for 12 cards.
export fn init() void {
    log("carousel: init");

    // Camera: slightly elevated, looking at origin -- gives a nice
    // top-down perspective on the stack.
    const eye = vec.vec3(0, 0.5, 4.0);
    const target = vec.vec3(0, 0, 0);
    const up = vec.vec3(0, 1, 0);
    view_matrix = Mat4.lookAt(eye, target, up).toArray();

    // Default projection (will be recalculated on resize)
    const aspect: f32 = @as(f32, @floatFromInt(canvas_width)) /
        @as(f32, @floatFromInt(canvas_height));
    proj_matrix = Mat4.perspective(math.pi / 4.0, aspect, 0.1, 100.0).toArray();

    // Initialise depth stack with 12 cards
    depth_stack = Stack.init(MAX_CARDS);
    depth_stack.computeLayout();
    visible_count = depth_stack.writeTransforms(&transform_buffer);
    depth_stack.writeOpacities(&opacity_buffer);
    depth_stack.writeTexIndices(&tex_index_buffer);

    log("carousel: init complete");
}

/// Called every requestAnimationFrame. `dt` is the time delta in seconds.
/// Gate 1: static depth stack (no animation yet -- scrolling comes in Gate 2).
export fn frame(dt: f32) void {
    _ = dt;

    // Recompute layout each frame (will matter once scroll_offset animates)
    depth_stack.computeLayout();
    visible_count = depth_stack.writeTransforms(&transform_buffer);
    depth_stack.writeOpacities(&opacity_buffer);
    depth_stack.writeTexIndices(&tex_index_buffer);
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
/// directly from WASM linear memory (MAX_CARDS * 16 floats).
export fn getTransformBufferPtr() [*]const f32 {
    return @ptrCast(&transform_buffer);
}

/// Returns a pointer to the per-card opacity buffer.
export fn getOpacityBufferPtr() [*]const f32 {
    return @ptrCast(&opacity_buffer);
}

/// Returns a pointer to the per-card texture index buffer.
export fn getTexIndexBufferPtr() [*]const f32 {
    return @ptrCast(&tex_index_buffer);
}

/// Returns a pointer to the view matrix.
export fn getViewMatrixPtr() [*]const f32 {
    return @ptrCast(&view_matrix);
}

/// Returns a pointer to the projection matrix.
export fn getProjMatrixPtr() [*]const f32 {
    return @ptrCast(&proj_matrix);
}

/// Returns the number of visible cards (instance count for draw call).
export fn getCardCount() u32 {
    return visible_count;
}

// ---------------------------------------------------------------------------
// Tests -- run on native target via `zig build test`
// ---------------------------------------------------------------------------

// Re-export tests from submodules so `zig build test` runs them all.
test {
    _ = @import("vec.zig");
    _ = @import("mat4.zig");
    _ = @import("stack.zig");
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

test "init sets visible_count to 12" {
    init();
    try std.testing.expectEqual(@as(u32, MAX_CARDS), getCardCount());
}

test "frame does not crash" {
    init();
    frame(1.0);
    // After frame, all cards should still be visible
    try std.testing.expectEqual(@as(u32, MAX_CARDS), getCardCount());
}

test "resize updates projection" {
    init();
    const old_proj = getProjMatrixPtr()[0];
    resize(1920, 1080);
    const new_proj = getProjMatrixPtr()[0];
    // Different aspect ratio should change projection[0]
    try std.testing.expect(old_proj != new_proj);
}

test "opacity buffer is populated after init" {
    init();
    const ptr = getOpacityBufferPtr();
    // First card should have opacity 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ptr[0], 1e-5);
    // Last card should have opacity > 0
    try std.testing.expect(ptr[MAX_CARDS - 1] > 0);
}

test "tex index buffer is populated after init" {
    init();
    const ptr = getTexIndexBufferPtr();
    // First card should be texture 0
    try std.testing.expectApproxEqAbs(@as(f32, 0), ptr[0], 1e-5);
    // Last card should be texture 11
    try std.testing.expectApproxEqAbs(@as(f32, 11), ptr[MAX_CARDS - 1], 1e-5);
}
