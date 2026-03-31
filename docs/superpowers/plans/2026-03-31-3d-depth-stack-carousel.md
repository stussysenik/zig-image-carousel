# 3D Depth Stack Carousel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-performance 3D depth stack image carousel in Zig → wasm32-freestanding → WebGL2, with touch/mouse gesture physics, progressive image loading, and Facebook-designer-quality polish.

**Architecture:** Maximum Zig — Zig owns all compute (physics, transforms, culling, SIMD math). JS is a ~200 LOC GPU driver that reads shared memory and issues WebGL2 calls. Zero allocations in the hot path. Single instanced draw call per frame.

**Tech Stack:** Zig 0.15.2, wasm32-freestanding + SIMD128, WebGL2, raw JS (no frameworks)

**Spec:** `docs/superpowers/specs/2026-03-31-3d-depth-stack-carousel-design.md`

---

## File Structure

```
zig-image-carousel/
├── build.zig              # wasm32-freestanding target, ReleaseSmall, SIMD128
├── build.zig.zon          # package manifest
├── src/
│   ├── main.zig           # WASM exports: init(), frame(dt), resize(w,h)
│   ├── vec.zig            # Vec3/Vec4 via @Vector(4, f32), SIMD ops
│   ├── mat4.zig           # 4×4 column-major matrices, perspective, lookAt
│   ├── stack.zig          # Depth stack layout: 12 card transforms + opacity
│   ├── input.zig          # Ring buffer: JS writes touch events, Zig reads
│   ├── gesture.zig        # Touch FSM: idle→pressed→dragging→fling→snap
│   ├── physics.zig        # Friction decel + critically damped spring
│   └── ease.zig           # Easing functions for animations
├── web/
│   ├── index.html         # Canvas + minimal markup
│   ├── host.js            # WebGL2 driver + event marshaling (~200 LOC)
│   ├── carousel.wasm      # Build output (auto-copied by build step)
│   └── shaders/
│       ├── card.vert      # Instanced vertex shader
│       └── card.frag      # Texture array sampling + depth effects
└── tests/                 # (tests embedded in source via Zig test blocks)
```

---

## Task 1: Project Scaffold + build.zig

**Files:**
- Create: `build.zig`
- Create: `build.zig.zon`
- Create: `src/main.zig` (minimal)
- Create: `web/index.html`
- Create: `web/shaders/card.vert`
- Create: `web/shaders/card.frag`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/s3nik/Desktop/zig-image-carousel
mkdir -p src web/shaders web/images
```

- [ ] **Step 2: Write build.zig with wasm32-freestanding + SIMD128**

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    // --- WASM target ---
    var wasm_query: std.Target.Query = .{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    };
    wasm_query.cpu_features_add = std.Target.wasm.featureSet(&.{.simd128});

    const wasm_target = b.resolveTargetQuery(wasm_query);

    const wasm_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
        .strip = true,
    });

    const wasm = b.addExecutable(.{
        .name = "carousel",
        .root_module = wasm_module,
    });

    wasm.entry = .disabled;
    wasm.rdynamic = true;
    wasm.export_memory = true;
    wasm.initial_memory = 4 * 65536;  // 256KB
    wasm.max_memory = 64 * 65536;     // 4MB

    b.installArtifact(wasm);

    // Copy WASM to web/ for dev serving
    const install_to_web = b.addInstallFile(
        wasm.getEmittedBin(),
        "../web/carousel.wasm",
    );
    b.getInstallStep().dependOn(&install_to_web.step);

    // --- Native tests ---
    const native_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = native_target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = test_module,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
```

- [ ] **Step 3: Write build.zig.zon**

```zig
.{
    .name = .@"zig-image-carousel",
    .version = .{ 0, 1, 0 },
    .fingerprint = 0x0,
    .minimum_zig_version = .{ 0, 15, 0 },
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
        "web",
    },
}
```

- [ ] **Step 4: Write minimal src/main.zig that exports init/frame/resize**

```zig
const std = @import("std");
const math = std.math;
const builtin = @import("builtin");

const is_wasm = builtin.cpu.arch == .wasm32;

// JS host imports (wasm only, native fallback for tests)
const env = if (is_wasm) struct {
    extern "env" fn consoleLog(ptr: [*]const u8, len: u32) void;
} else struct {
    fn consoleLog(ptr: [*]const u8, len: u32) void {
        std.debug.print("{s}\n", .{ptr[0..len]});
    }
};

fn log(comptime fmt: []const u8, args: anytype) void {
    var buf: [256]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, fmt, args) catch return;
    env.consoleLog(slice.ptr, @intCast(slice.len));
}

// --- Shared memory ---
const MAX_CARDS = 12;
var transform_buffer: [MAX_CARDS * 16]f32 = undefined;
var view_matrix_buf: [16]f32 = undefined;
var proj_matrix_buf: [16]f32 = undefined;
var card_count: u32 = 1;
var angle: f32 = 0.0;

export fn getTransformBufferPtr() [*]f32 {
    return &transform_buffer;
}
export fn getViewMatrixPtr() [*]f32 {
    return &view_matrix_buf;
}
export fn getProjMatrixPtr() [*]f32 {
    return &proj_matrix_buf;
}
export fn getCardCount() u32 {
    return card_count;
}

export fn init() void {
    // Identity matrix for single card
    transform_buffer[0..16].* = .{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    // Simple view: camera at z=3 looking at origin
    view_matrix_buf = .{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -3, 1 };
    // Placeholder projection (will be set by resize)
    proj_matrix_buf = .{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    log("carousel init: buffer ready", .{});
}

export fn frame(dt: f32) void {
    angle += dt * 0.001;
    if (angle > 2.0 * math.pi) angle -= 2.0 * math.pi;
    const c = @cos(angle);
    const s = @sin(angle);
    // Rotation-Y matrix, column-major
    var m = transform_buffer[0..16];
    m.* = .{ c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1 };
}

export fn resize(width: u32, height: u32) void {
    const aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
    const fov_y = math.pi / 4.0;
    const f = 1.0 / @tan(fov_y / 2.0);
    const near: f32 = 0.1;
    const far: f32 = 100.0;
    const nf = 1.0 / (near - far);
    proj_matrix_buf = .{
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2.0 * far * near * nf, 0,
    };
}

test "identity matrix check" {
    var m: [16]f32 = .{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    try std.testing.expectEqual(@as(f32, 1.0), m[0]);
    try std.testing.expectEqual(@as(f32, 0.0), m[1]);
}
```

- [ ] **Step 5: Write web/shaders/card.vert (single-card version)**

```glsl
#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 v_texcoord;

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
}
```

- [ ] **Step 6: Write web/shaders/card.frag (solid color version)**

```glsl
#version 300 es
precision highp float;

in vec2 v_texcoord;

uniform vec4 u_color;

out vec4 fragColor;

void main() {
    fragColor = u_color;
}
```

- [ ] **Step 7: Write web/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>3D Depth Stack Carousel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #111; overflow: hidden; touch-action: none; }
        canvas { display: block; width: 100vw; height: 100vh; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script src="host.js" type="module"></script>
</body>
</html>
```

- [ ] **Step 8: Write web/host.js (Gate 0: single card rendering)**

```javascript
"use strict";

const canvas = document.getElementById("canvas");
const gl = canvas.getContext("webgl2", { alpha: false, antialias: true });
if (!gl) throw new Error("WebGL2 not supported");

function compileShader(gl, type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        throw new Error("Shader: " + gl.getShaderInfoLog(s));
    return s;
}

function createProgram(gl, vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compileShader(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compileShader(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
        throw new Error("Link: " + gl.getProgramInfoLog(p));
    return p;
}

// 3:2 aspect quad
const QUAD = new Float32Array([
    -0.75,-0.5, 0,0,  0.75,-0.5, 1,0,  0.75,0.5, 1,1,
    -0.75,-0.5, 0,0,  0.75,0.5, 1,1,  -0.75,0.5, 0,1,
]);

let wasm, mem, transformPtr, viewPtr, projPtr;
let program, vao, uniforms, lastTime = 0;

async function main() {
    const importObj = {
        env: {
            consoleLog: (ptr, len) => {
                const bytes = new Uint8Array(wasm.instance.exports.memory.buffer, ptr, len);
                console.log("[zig]", new TextDecoder().decode(bytes));
            },
        },
    };

    wasm = await WebAssembly.instantiate(
        await (await fetch("carousel.wasm")).arrayBuffer(), importObj
    );
    mem = wasm.instance.exports.memory;

    resizeCanvas();
    wasm.instance.exports.init();

    transformPtr = wasm.instance.exports.getTransformBufferPtr();
    viewPtr = wasm.instance.exports.getViewMatrixPtr();
    projPtr = wasm.instance.exports.getProjMatrixPtr();

    const [vs, fs] = await Promise.all([
        fetch("shaders/card.vert").then(r => r.text()),
        fetch("shaders/card.frag").then(r => r.text()),
    ]);
    program = createProgram(gl, vs, fs);
    uniforms = {
        u_model: gl.getUniformLocation(program, "u_model"),
        u_view: gl.getUniformLocation(program, "u_view"),
        u_projection: gl.getUniformLocation(program, "u_projection"),
        u_color: gl.getUniformLocation(program, "u_color"),
    };

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    gl.bindVertexArray(null);

    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.067, 0.067, 0.067, 1.0);

    window.addEventListener("resize", () => {
        resizeCanvas();
        wasm.instance.exports.resize(canvas.width, canvas.height);
    });

    lastTime = performance.now();
    requestAnimationFrame(loop);
}

function resizeCanvas() {
    const dpr = devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    if (wasm) wasm.instance.exports.resize(canvas.width, canvas.height);
}

function loop(now) {
    const dt = now - lastTime;
    lastTime = now;
    wasm.instance.exports.frame(dt);

    const model = new Float32Array(mem.buffer, transformPtr, 16);
    const view = new Float32Array(mem.buffer, viewPtr, 16);
    const proj = new Float32Array(mem.buffer, projPtr, 16);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(program);
    gl.uniformMatrix4fv(uniforms.u_model, false, model);
    gl.uniformMatrix4fv(uniforms.u_view, false, view);
    gl.uniformMatrix4fv(uniforms.u_projection, false, proj);
    gl.uniform4f(uniforms.u_color, 0.2, 0.6, 1.0, 1.0);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);

    requestAnimationFrame(loop);
}

main().catch(console.error);
```

- [ ] **Step 9: Build and verify Gate 0**

```bash
cd /Users/s3nik/Desktop/zig-image-carousel
zig build
zig build test
ls -la web/carousel.wasm
python3 -m http.server 8080 --directory web
```

Expected: Blue 3:2 quad rotating on dark background. Console shows `[zig] carousel init: buffer ready`. WASM < 50KB. 60fps.

- [ ] **Step 10: Commit Gate 0**

```bash
git init
git add build.zig build.zig.zon src/main.zig web/
git commit -m "gate 0: foundation - zig wasm webgl2 pipeline proven"
```

---

## Task 2: SIMD Vector + Matrix Math

**Files:**
- Create: `src/vec.zig`
- Create: `src/mat4.zig`
- Modify: `src/main.zig` (use mat4 for camera)

- [ ] **Step 1: Write src/vec.zig with @Vector(4, f32) SIMD**

```zig
const std = @import("std");

pub const Vec4 = @Vector(4, f32);
pub const Vec3 = @Vector(4, f32); // w=0 for SIMD alignment

pub fn vec3(x: f32, y: f32, z: f32) Vec3 {
    return .{ x, y, z, 0.0 };
}

pub fn vec4(x: f32, y: f32, z: f32, w: f32) Vec4 {
    return .{ x, y, z, w };
}

pub fn dot3(a: Vec3, b: Vec3) f32 {
    const prod = a * b;
    return prod[0] + prod[1] + prod[2];
}

pub fn cross(a: Vec3, b: Vec3) Vec3 {
    const a_yzx: Vec4 = .{ a[1], a[2], a[0], 0.0 };
    const a_zxy: Vec4 = .{ a[2], a[0], a[1], 0.0 };
    const b_yzx: Vec4 = .{ b[1], b[2], b[0], 0.0 };
    const b_zxy: Vec4 = .{ b[2], b[0], b[1], 0.0 };
    return a_yzx * b_zxy - a_zxy * b_yzx;
}

pub fn length3(v: Vec3) f32 {
    return @sqrt(dot3(v, v));
}

pub fn normalize3(v: Vec3) Vec3 {
    const len = length3(v);
    if (len < 1e-10) return .{ 0, 0, 0, 0 };
    const inv: Vec4 = @splat(1.0 / len);
    return v * inv;
}

pub fn scale(v: Vec4, s: f32) Vec4 {
    const sv: Vec4 = @splat(s);
    return v * sv;
}

test "dot product" {
    const a = vec3(1, 2, 3);
    const b = vec3(4, 5, 6);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dot3(a, b), 1e-6);
}

test "cross product x cross y = z" {
    const x = vec3(1, 0, 0);
    const y = vec3(0, 1, 0);
    const z = cross(x, y);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), z[2], 1e-6);
}

test "normalize" {
    const v = vec3(3, 4, 0);
    const n = normalize3(v);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), n[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), n[1], 1e-6);
}

test "length" {
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), length3(vec3(3, 4, 0)), 1e-6);
}
```

- [ ] **Step 2: Write src/mat4.zig with column-major SIMD matrices**

```zig
const std = @import("std");
const math = std.math;
const vec = @import("vec.zig");
const Vec4 = vec.Vec4;
const Vec3 = vec.Vec3;

pub const Mat4 = struct {
    cols: [4]Vec4,

    pub const identity: Mat4 = .{ .cols = .{
        .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 }, .{ 0, 0, 0, 1 },
    } };

    pub fn mul(a: Mat4, b: Mat4) Mat4 {
        var r: Mat4 = undefined;
        inline for (0..4) |i| {
            r.cols[i] = a.mulVec(b.cols[i]);
        }
        return r;
    }

    pub fn mulVec(m: Mat4, v: Vec4) Vec4 {
        const vx: Vec4 = @splat(v[0]);
        const vy: Vec4 = @splat(v[1]);
        const vz: Vec4 = @splat(v[2]);
        const vw: Vec4 = @splat(v[3]);
        return m.cols[0] * vx + m.cols[1] * vy + m.cols[2] * vz + m.cols[3] * vw;
    }

    pub fn translate(tx: f32, ty: f32, tz: f32) Mat4 {
        return .{ .cols = .{
            .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 }, .{ tx, ty, tz, 1 },
        } };
    }

    pub fn scaleUniform(s: f32) Mat4 {
        return .{ .cols = .{
            .{ s, 0, 0, 0 }, .{ 0, s, 0, 0 }, .{ 0, 0, s, 0 }, .{ 0, 0, 0, 1 },
        } };
    }

    pub fn rotateY(angle_rad: f32) Mat4 {
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);
        return .{ .cols = .{
            .{ c, 0, -s, 0 }, .{ 0, 1, 0, 0 }, .{ s, 0, c, 0 }, .{ 0, 0, 0, 1 },
        } };
    }

    pub fn rotateX(angle_rad: f32) Mat4 {
        const c = @cos(angle_rad);
        const s = @sin(angle_rad);
        return .{ .cols = .{
            .{ 1, 0, 0, 0 }, .{ 0, c, s, 0 }, .{ 0, -s, c, 0 }, .{ 0, 0, 0, 1 },
        } };
    }

    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) Mat4 {
        const f = 1.0 / @tan(fov_y / 2.0);
        const nf = 1.0 / (near - far);
        return .{ .cols = .{
            .{ f / aspect, 0, 0, 0 },
            .{ 0, f, 0, 0 },
            .{ 0, 0, (far + near) * nf, -1 },
            .{ 0, 0, 2.0 * far * near * nf, 0 },
        } };
    }

    pub fn lookAt(eye: Vec3, target: Vec3, world_up: Vec3) Mat4 {
        const forward = vec.normalize3(target - eye);
        const right = vec.normalize3(vec.cross(forward, world_up));
        const up = vec.cross(right, forward);
        return .{ .cols = .{
            .{ right[0], up[0], -forward[0], 0 },
            .{ right[1], up[1], -forward[1], 0 },
            .{ right[2], up[2], -forward[2], 0 },
            .{ -vec.dot3(right, eye), -vec.dot3(up, eye), vec.dot3(forward, eye), 1 },
        } };
    }

    pub fn toArray(self: Mat4) [16]f32 {
        var r: [16]f32 = undefined;
        inline for (0..4) |i| {
            r[i * 4 + 0] = self.cols[i][0];
            r[i * 4 + 1] = self.cols[i][1];
            r[i * 4 + 2] = self.cols[i][2];
            r[i * 4 + 3] = self.cols[i][3];
        }
        return r;
    }

    pub fn writeTo(self: Mat4, buf: []f32, offset: usize) void {
        const arr = self.toArray();
        @memcpy(buf[offset..][0..16], &arr);
    }
};

test "identity * vec = vec" {
    const v: Vec4 = .{ 1, 2, 3, 1 };
    const r = Mat4.identity.mulVec(v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), r[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), r[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), r[2], 1e-6);
}

test "translate moves point" {
    const t = Mat4.translate(10, 20, 30);
    const r = t.mulVec(.{ 1, 2, 3, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), r[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), r[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), r[2], 1e-6);
}

test "perspective produces valid matrix" {
    const p = Mat4.perspective(math.pi / 4.0, 16.0 / 9.0, 0.1, 100.0);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), p.cols[2][3], 1e-5);
}

test "lookAt camera at z=3" {
    const v = Mat4.lookAt(vec.vec3(0, 0, 3), vec.vec3(0, 0, 0), vec.vec3(0, 1, 0));
    const r = v.mulVec(.{ 0, 0, 0, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), r[2], 1e-5);
}

test "toArray column-major layout" {
    const t = Mat4.translate(5, 6, 7);
    const arr = t.toArray();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), arr[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), arr[13], 1e-6);
}
```

- [ ] **Step 3: Update main.zig to use mat4/vec for camera**

Replace the inline projection/view computation in `resize()` and `init()` with calls to `Mat4.perspective()` and `Mat4.lookAt()`. Import mat4 and vec modules. Add test re-exports:

```zig
test {
    _ = @import("vec.zig");
    _ = @import("mat4.zig");
}
```

- [ ] **Step 4: Run tests and verify**

```bash
zig build test    # All vec, mat4 tests pass
zig build         # WASM builds with mat4/vec included
```

- [ ] **Step 5: Commit**

```bash
git add src/vec.zig src/mat4.zig src/main.zig
git commit -m "gate 1 (partial): SIMD vec4 and mat4 math library"
```

---

## Task 3: Depth Stack Layout + Instanced Rendering

**Files:**
- Create: `src/stack.zig`
- Modify: `src/main.zig` (12-card layout)
- Modify: `web/shaders/card.vert` (instanced)
- Modify: `web/shaders/card.frag` (per-instance opacity + texture array)
- Modify: `web/host.js` (instanced rendering + placeholder textures)

- [ ] **Step 1: Write src/stack.zig — depth stack layout engine**

```zig
const std = @import("std");
const mat4_mod = @import("mat4.zig");
const Mat4 = mat4_mod.Mat4;

pub const MAX_CARDS = 12;

pub const CardState = struct {
    transform: Mat4,
    opacity: f32,
    texture_index: u32,
    visible: bool,
};

pub const Stack = struct {
    cards: [MAX_CARDS]CardState = undefined,
    num_cards: u32 = MAX_CARDS,
    scroll_offset: f32 = 0.0,

    // Layout params
    depth_step_base: f32 = 0.4,
    depth_step_exp: f32 = 1.15,
    scale_decay: f32 = 0.08,
    opacity_decay: f32 = 0.075,
    y_gap: f32 = 0.12,

    pub fn computeLayout(self: *Stack) void {
        for (0..self.num_cards) |idx| {
            const i = @as(f32, @floatFromInt(idx));
            const eff = i - self.scroll_offset;

            const depth = self.depth_step_base *
                (std.math.pow(f32, self.depth_step_exp, eff) - 1.0) /
                (self.depth_step_exp - 1.0);

            const y = eff * self.y_gap;
            const s = @max(0.3, 1.0 - eff * self.scale_decay);
            const opacity = @max(0.1, 1.0 - eff * self.opacity_decay);

            self.cards[idx] = .{
                .transform = Mat4.translate(0, y, -depth).mul(Mat4.scaleUniform(s)),
                .opacity = opacity,
                .texture_index = @intCast(idx),
                .visible = eff >= -1.0 and eff < @as(f32, @floatFromInt(MAX_CARDS)),
            };
        }
    }

    pub fn writeTransforms(self: *const Stack, buf: []f32) u32 {
        var count: u32 = 0;
        for (0..self.num_cards) |idx| {
            if (!self.cards[idx].visible) continue;
            self.cards[idx].transform.writeTo(buf, count * 16);
            count += 1;
        }
        return count;
    }

    pub fn writeOpacities(self: *const Stack, buf: []f32) u32 {
        var count: u32 = 0;
        for (0..self.num_cards) |idx| {
            if (!self.cards[idx].visible) continue;
            buf[count] = self.cards[idx].opacity;
            count += 1;
        }
        return count;
    }

    pub fn writeTexIndices(self: *const Stack, buf: []f32) u32 {
        var count: u32 = 0;
        for (0..self.num_cards) |idx| {
            if (!self.cards[idx].visible) continue;
            buf[count] = @floatFromInt(self.cards[idx].texture_index);
            count += 1;
        }
        return count;
    }
};

test "card 0 at scroll=0 is near origin" {
    var s = Stack{};
    s.computeLayout();
    try std.testing.expect(s.cards[0].visible);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), s.cards[0].opacity, 0.01);
}

test "cards recede in Z" {
    var s = Stack{};
    s.computeLayout();
    const z0 = s.cards[0].transform.cols[3][2];
    const z1 = s.cards[1].transform.cols[3][2];
    try std.testing.expect(z1 < z0);
}

test "opacity decreases" {
    var s = Stack{};
    s.computeLayout();
    try std.testing.expect(s.cards[1].opacity < s.cards[0].opacity);
    try std.testing.expect(s.cards[11].opacity >= 0.1);
}
```

- [ ] **Step 2: Update main.zig for 12-card stack with shared buffers**

Wire up Stack into main.zig. Add `opacity_buffer`, `tex_index_buffer`. The `frame()` function calls `stack.computeLayout()` and writes all buffers. Export getters for JS to find buffer pointers.

- [ ] **Step 3: Update card.vert for instanced rendering**

```glsl
#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;
layout(location = 2) in vec4 a_model_col0;
layout(location = 3) in vec4 a_model_col1;
layout(location = 4) in vec4 a_model_col2;
layout(location = 5) in vec4 a_model_col3;
layout(location = 6) in float a_opacity;
layout(location = 7) in float a_tex_layer;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 v_texcoord;
out float v_opacity;
out float v_tex_layer;
out float v_depth;

void main() {
    mat4 model = mat4(a_model_col0, a_model_col1, a_model_col2, a_model_col3);
    vec4 viewPos = u_view * model * vec4(a_position, 0.0, 1.0);
    gl_Position = u_projection * viewPos;
    v_texcoord = a_texcoord;
    v_opacity = a_opacity;
    v_tex_layer = a_tex_layer;
    v_depth = -viewPos.z;
}
```

- [ ] **Step 4: Update card.frag for texture array + opacity**

```glsl
#version 300 es
precision highp float;

in vec2 v_texcoord;
in float v_opacity;
in float v_tex_layer;
in float v_depth;

uniform highp sampler2DArray u_textures;

out vec4 fragColor;

void main() {
    vec4 color = texture(u_textures, vec3(v_texcoord, v_tex_layer));
    fragColor = vec4(color.rgb, color.a * v_opacity);
}
```

- [ ] **Step 5: Rewrite host.js for instanced rendering**

Major update: Create instance buffers for mat4 columns (locations 2-5), opacity (location 6), tex layer (location 7). Use `vertexAttribDivisor(loc, 1)` for all instance attributes. Generate placeholder colored textures via `TEXTURE_2D_ARRAY` + `texSubImage3D`. Render with single `drawArraysInstanced(gl.TRIANGLES, 0, 6, cardCount)`.

- [ ] **Step 6: Build, test, verify Gate 1**

```bash
zig build test    # All tests pass (vec, mat4, stack)
zig build         # WASM < 50KB
python3 -m http.server 8080 --directory web
```

Expected: 12 colored numbered cards in depth stack. Cards recede, shrink, and fade. Single draw call. 60fps.

- [ ] **Step 7: Commit Gate 1**

```bash
git add src/ web/
git commit -m "gate 1: 12-card depth stack with instanced rendering"
```

---

## Task 4: Input Pipeline — Ring Buffer

**Files:**
- Create: `src/input.zig`
- Modify: `web/host.js` (add touch/mouse → ring buffer writing)
- Modify: `src/main.zig` (import input)

- [ ] **Step 1: Write src/input.zig — shared ring buffer**

```zig
//! Input ring buffer: JS writes, Zig reads. Lock-free SPSC.

pub const EventType = enum(u8) {
    none = 0,
    start = 1,
    move = 2,
    end = 3,
};

pub const InputEvent = extern struct {
    event_type: u8,
    _pad: [3]u8,
    x: f32,
    y: f32,
    timestamp: f32,
};

comptime {
    if (@sizeOf(InputEvent) != 16) @compileError("InputEvent must be 16 bytes");
}

const RING_SIZE: u32 = 64;
const RING_MASK: u32 = RING_SIZE - 1;

export var input_ring: [RING_SIZE]InputEvent = [_]InputEvent{.{
    .event_type = 0, ._pad = .{ 0, 0, 0 }, .x = 0, .y = 0, .timestamp = 0,
}} ** RING_SIZE;

export var input_head: u32 = 0;
export var input_tail: u32 = 0;

export fn getInputRingPtr() usize { return @intFromPtr(&input_ring); }
export fn getInputHeadPtr() usize { return @intFromPtr(&input_head); }
export fn getInputTailPtr() usize { return @intFromPtr(&input_tail); }

pub fn poll() ?InputEvent {
    if (input_tail == input_head) return null;
    const event = input_ring[input_tail & RING_MASK];
    input_tail +%= 1;
    return event;
}
```

- [ ] **Step 2: Add touch/mouse event marshaling to host.js**

After WASM loads, call `initInput(exports, memory)` which sets up `touchstart/move/end` and `mousedown/move/up/leave` listeners. Each listener writes to the ring buffer via `DataView` at the exported pointer offsets. Ring-full check: `(head - tail) >>> 0 >= 64`.

- [ ] **Step 3: Verify in browser**

Touch/click canvas → ring buffer head increments → no JS errors.

- [ ] **Step 4: Commit**

```bash
git add src/input.zig web/host.js
git commit -m "gate 2 (partial): input ring buffer - JS touch/mouse to WASM"
```

---

## Task 5: Gesture State Machine

**Files:**
- Create: `src/gesture.zig`
- Modify: `src/main.zig` (wire gesture into frame loop)

- [ ] **Step 1: Write src/gesture.zig — 5-state FSM**

States: `idle, pressed, dragging, flinging, snapping`. Transitions per spec. Velocity sampling: circular buffer of last 5 move deltas, smoothed average. Dead zone: 8px. Scroll position in card-units (200px per card). `tap_fired` flag for tap detection. Interrupt: fling/snap → touch_start → dragging.

Key methods: `processEvent(InputEvent)`, `computeSmoothedVelocity()`, `nearestCardPosition()`, `beginSnap()`, `settle()`.

- [ ] **Step 2: Write gesture unit tests**

```zig
test "idle to pressed on touch start" {
    var g = Gesture.init(12);
    g.processEvent(.{ .event_type = 1, ._pad = .{0,0,0}, .x = 100, .y = 200, .timestamp = 0 });
    try std.testing.expectEqual(State.pressed, g.state);
}

test "pressed to dragging after dead zone" {
    var g = Gesture.init(12);
    g.processEvent(.{ .event_type = 1, ._pad = .{0,0,0}, .x = 100, .y = 200, .timestamp = 0 });
    g.processEvent(.{ .event_type = 2, ._pad = .{0,0,0}, .x = 100, .y = 220, .timestamp = 16 });
    try std.testing.expectEqual(State.dragging, g.state);
}

test "tap detection" {
    var g = Gesture.init(12);
    g.processEvent(.{ .event_type = 1, ._pad = .{0,0,0}, .x = 100, .y = 200, .timestamp = 0 });
    g.processEvent(.{ .event_type = 3, ._pad = .{0,0,0}, .x = 100, .y = 200, .timestamp = 100 });
    try std.testing.expect(g.tap_fired);
    try std.testing.expectEqual(State.idle, g.state);
}

test "interrupt fling with touch" {
    var g = Gesture.init(12);
    g.state = .flinging;
    g.scroll_velocity = 500;
    g.processEvent(.{ .event_type = 1, ._pad = .{0,0,0}, .x = 100, .y = 200, .timestamp = 0 });
    try std.testing.expectEqual(State.dragging, g.state);
    try std.testing.expectEqual(@as(f32, 0), g.scroll_velocity);
}
```

- [ ] **Step 3: Wire gesture into main.zig frame loop**

In `frame(dt)`: drain `input.poll()` → `gesture.processEvent()`. Pass `gesture.scroll_position` to `stack.scroll_offset`.

- [ ] **Step 4: Run tests, verify**

```bash
zig build test
```

- [ ] **Step 5: Commit**

```bash
git add src/gesture.zig src/main.zig
git commit -m "gate 2 (partial): gesture FSM - touch state machine"
```

---

## Task 6: Physics Engine — Friction + Spring

**Files:**
- Create: `src/physics.zig`
- Modify: `src/main.zig` (call physics.update in frame)

- [ ] **Step 1: Write src/physics.zig**

Frame-rate independent friction: `velocity *= pow(0.94, dt / (1/60))`. Critically damped spring: `a = -k*(x-target) - 2*zeta*sqrt(k)*v`. Semi-implicit Euler. Settle epsilon: 0.005 card-units.

Triggers state transitions: flinging→snapping when velocity < 50px/s. snapping→idle when converged.

- [ ] **Step 2: Write physics unit tests**

```zig
test "friction reduces velocity" {
    var g = gesture_mod.Gesture.init(12);
    g.state = .flinging;
    g.scroll_velocity = 1000;
    physics.update(&g, 1.0 / 60.0);
    try std.testing.expect(g.scroll_velocity < 1000);
    try std.testing.expect(g.scroll_velocity > 0);
}

test "spring converges to target" {
    var g = gesture_mod.Gesture.init(12);
    g.state = .snapping;
    g.scroll_position = 1.5;
    g.snap_target = 2.0;
    g.scroll_velocity = 0;
    // Run 600 frames (~10 seconds at 60fps)
    for (0..600) |_| {
        physics.update(&g, 1.0 / 60.0);
        if (g.state == .idle) break;
    }
    try std.testing.expectEqual(gesture_mod.State.idle, g.state);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), g.scroll_position, 0.01);
}

test "no NaN in physics" {
    var g = gesture_mod.Gesture.init(12);
    g.state = .flinging;
    g.scroll_velocity = 8000;
    for (0..1000) |_| {
        physics.update(&g, 1.0 / 60.0);
        try std.testing.expect(!std.math.isNan(g.scroll_position));
        try std.testing.expect(!std.math.isNan(g.scroll_velocity));
    }
}
```

- [ ] **Step 3: Wire physics into main.zig**

In `frame(dt_ms)`: call `physics.update(&gesture, dt_ms / 1000.0)` after draining input.

- [ ] **Step 4: Build, test, verify Gate 2 in browser**

```bash
zig build test
zig build
python3 -m http.server 8080 --directory web
```

Expected: Drag cards with mouse/touch. Fling momentum. Magnetic snap to nearest card. Interruptible. iOS-quality feel.

- [ ] **Step 5: Commit Gate 2**

```bash
git add src/physics.zig src/main.zig
git commit -m "gate 2: interaction - friction/spring physics, gesture FSM"
```

---

## Task 7: Progressive Image Loading

**Files:**
- Modify: `web/host.js` (replace placeholder textures with progressive loader)
- Modify: `src/main.zig` (export scroll position for LOD decisions)

- [ ] **Step 1: Add progressive image loading to host.js**

Replace placeholder texture generation with a 4-tier progressive loader:
- Tier 0: Dominant color (1×1 canvas drawImage extraction)
- Tier 1: 64×64 thumbnail via `createImageBitmap` with resize
- Tier 2: 512×512 medium
- Tier 3: Full resolution (only for focused card)

Use `fetch()` + `createImageBitmap()` for off-thread decode. Upload via `texSubImage3D` to the texture array. Track loading state per card. Only load Tier 2+ when card is near viewport (check `getScrollPosition()` from Zig).

- [ ] **Step 2: Add LRU texture eviction**

Track VRAM budget. When exceeding limit, downgrade far cards from Tier 2 back to Tier 1.

- [ ] **Step 3: Add idle-frame scheduling**

Only upload textures during frames with < 12ms spent. Check `performance.now()` delta before/after Zig frame call. If headroom exists, upload one pending texture.

- [ ] **Step 4: Add sample test images**

Download 12 high-quality photos to `web/images/` for testing. Provide multiple resolutions (64px, 512px, full).

- [ ] **Step 5: Test with 50+ images, rapid scroll**

Verify: no jank during rapid scroll, progressive reveal visible, memory stays under 100MB.

- [ ] **Step 6: Commit Gate 3**

```bash
git add web/host.js web/images/
git commit -m "gate 3: progressive image loading with LOD + LRU eviction"
```

---

## Task 8: Visibility Culling

**Files:**
- Modify: `src/stack.zig` (skip culled cards)
- Modify: `src/main.zig` (export visible count)

- [ ] **Step 1: Enhance stack.zig culling logic**

Cards with `effective_i < -1.0` or `effective_i >= MAX_CARDS` are marked invisible. `writeTransforms()` already skips invisible cards. Add margin-based culling for cards barely off-screen.

- [ ] **Step 2: Verify culling reduces draw count**

When scrolled to card 5, only ~12 nearest cards should be drawn. Export `visible_count` for JS to use as instance count.

- [ ] **Step 3: Commit**

```bash
git add src/stack.zig src/main.zig
git commit -m "gate 3: visibility culling - only render visible cards"
```

---

## Task 9: Visual Polish — Depth Effects

**Files:**
- Modify: `web/shaders/card.frag` (depth blur, shadows, glow)
- Modify: `web/shaders/card.vert` (slight X rotation for receding cards)
- Modify: `web/host.js` (rounded corners, background)

- [ ] **Step 1: Add depth-based blur to fragment shader**

Far cards get progressively blurred using a simple box blur approximation: sample texture at multiple offsets scaled by depth, average results. This simulates depth-of-field cheaply.

```glsl
// In card.frag: depth-based blur
float blur_amount = smoothstep(2.0, 8.0, v_depth) * 0.003;
vec4 color = vec4(0.0);
color += texture(u_textures, vec3(v_texcoord + vec2(-blur_amount, 0), v_tex_layer));
color += texture(u_textures, vec3(v_texcoord + vec2(blur_amount, 0), v_tex_layer));
color += texture(u_textures, vec3(v_texcoord + vec2(0, -blur_amount), v_tex_layer));
color += texture(u_textures, vec3(v_texcoord + vec2(0, blur_amount), v_tex_layer));
color += texture(u_textures, vec3(v_texcoord, v_tex_layer));
color /= 5.0;
```

- [ ] **Step 2: Add focused card glow**

For the front card (v_depth closest to camera), add a subtle edge glow using distance from card edges:

```glsl
float edge_dist = min(min(v_texcoord.x, 1.0 - v_texcoord.x), min(v_texcoord.y, 1.0 - v_texcoord.y));
float glow = smoothstep(0.0, 0.05, edge_dist) * (1.0 - smoothstep(0.05, 0.15, edge_dist));
glow *= smoothstep(3.0, 1.5, v_depth) * 0.3;
color.rgb += vec3(0.4, 0.6, 1.0) * glow;
```

- [ ] **Step 3: Add subtle X-axis tilt for receding cards**

In stack.zig, add a slight rotateX to cards that are deeper in the stack:

```zig
const tilt = eff * 0.02; // radians, increases with depth
const model = Mat4.translate(0, y, -depth)
    .mul(Mat4.rotateX(tilt))
    .mul(Mat4.scaleUniform(s));
```

- [ ] **Step 4: Add rounded corners in fragment shader**

```glsl
// Rounded corners via SDF
vec2 p = abs(v_texcoord - 0.5) * 2.0; // -1 to 1
float corner_radius = 0.08;
vec2 q = p - vec2(1.0 - corner_radius);
float d = length(max(q, 0.0)) - corner_radius;
if (d > 0.0) discard;
```

- [ ] **Step 5: Add background gradient**

In host.js, before drawing cards, render a fullscreen quad with a subtle radial gradient that shifts with scroll position.

- [ ] **Step 6: Add DPI awareness and responsive sizing**

Canvas resolution matches `devicePixelRatio`. Card sizes adapt to viewport (smaller cards on mobile, larger on desktop). Recalculate on resize.

- [ ] **Step 7: Verify 60fps with all effects**

Chrome DevTools Performance tab. Every frame < 16.67ms. If GPU bottleneck detected, reduce blur samples.

- [ ] **Step 8: Commit Gate 4**

```bash
git add web/shaders/ web/host.js src/stack.zig
git commit -m "gate 4: visual polish - depth blur, glow, tilt, rounded corners"
```

---

## Task 10: Tap-to-Expand Animation

**Files:**
- Create: `src/ease.zig`
- Modify: `src/main.zig` (expand state)
- Modify: `web/host.js` (dimmed overlay)
- Modify: `web/shaders/card.vert` (expand transform)

- [ ] **Step 1: Write src/ease.zig**

```zig
const std = @import("std");

pub fn easeOutCubic(t: f32) f32 {
    const t1 = t - 1.0;
    return t1 * t1 * t1 + 1.0;
}

pub fn easeInOutCubic(t: f32) f32 {
    if (t < 0.5) return 4.0 * t * t * t;
    const t1 = -2.0 * t + 2.0;
    return 1.0 - t1 * t1 * t1 / 2.0;
}

test "easeOutCubic endpoints" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), easeOutCubic(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), easeOutCubic(1.0), 1e-6);
}
```

- [ ] **Step 2: Add expand/collapse state to main.zig**

Track: `is_expanded: bool`, `expand_progress: f32` (0→1 over 300ms). When `gesture.tap_fired` and state is idle, begin expand animation. When expanded and tap fired again, begin collapse.

During expand: interpolate focused card transform from stack position to near-fullscreen (scale ~1.8, z closer to camera).

- [ ] **Step 3: Add dimmed overlay in host.js**

When expanded, draw a semi-transparent black fullscreen quad behind the expanded card but in front of other cards. Use a separate draw call with depth test disabled.

- [ ] **Step 4: Commit**

```bash
git add src/ease.zig src/main.zig web/host.js
git commit -m "gate 4: tap-to-expand animation with dimmed overlay"
```

---

## Verification

### Per-Gate Verification Commands

```bash
# Gate 0: Foundation
zig build && zig build test && wc -c < web/carousel.wasm
# Expected: < 50KB, tests pass, blue quad rotates at 60fps

# Gate 1: Stack
zig build test  # 15+ tests pass (vec, mat4, stack)
# Browser: 12 colored cards in depth stack, instanced draw

# Gate 2: Interaction
zig build test  # Gesture + physics tests pass
# Browser: drag, fling, snap all working. iOS-quality feel.

# Gate 3: Content
# Browser: progressive image loading, 50+ images, no jank

# Gate 4: Polish
# Browser: depth blur, glow, rounded corners, tap-expand
# Chrome DevTools: every frame < 16.67ms
# Lighthouse: performance > 90
```

### Chrome DevTools MCP Verification

At each gate, use Chrome DevTools MCP tools:
- `take_screenshot` — visual regression check
- `evaluate_script` — read `performance.now()` frame timings
- `performance_start_trace` / `performance_stop_trace` — verify 60fps
- `lighthouse_audit` — Gate 4 must score > 90
