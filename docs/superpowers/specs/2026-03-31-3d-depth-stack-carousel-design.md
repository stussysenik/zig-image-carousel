# 3D Depth Stack Image Carousel — Design Spec

## Context

Build a high-performance 3D image carousel for photography portfolios, written in Zig compiled to WebAssembly (wasm32-freestanding), rendered via raw WebGL2 in the browser. The goal is maximum performance through deep understanding of physical constraints: WASM SIMD, GPU pipeline stages, memory bandwidth, and frame budgets. No frameworks, no abstractions — bare metal web rendering.

**Why Zig/WASM**: Deterministic performance (no GC), SIMD via `@Vector`, zero-cost abstractions, tiny binary (<50KB). JS stays a thin GPU driver (~200 LOC).

**Why Depth Stack**: Cards layer in Z-depth with vertical scrolling. Front card is full size, cards behind scale down and recede. Swipe up pushes current card back, reveals next. Touch-native, maps directly to finger movement.

**Card Aspect Ratio**: 3:2 (matches most camera sensors). Cards are landscape-oriented. The quad geometry is sized to this ratio. Images are center-cropped to fit.

**TAP Action**: Tapping the focused (front) card triggers a scale-up animation to near-fullscreen with a dimmed background. Tap again or swipe to dismiss. Tapping a non-focused card snaps the stack to bring that card to front.

## Architecture: Maximum Zig

```
┌─── LAYER 5: BROWSER HOST (JS ~200 LOC) ────────────────────┐
│ canvas + WebGL2 context creation                             │
│ touch/mouse → write events to WASM shared memory             │
│ image decode → createImageBitmap → texImage2D                │
│ GPU driver → read Zig render state → issue WebGL2 calls      │
│ RAF loop → call zig.frame(deltaTime) → draw                  │
└──────────── ↕ extern fn / shared memory ────────────────────┘

┌─── LAYER 4: ZIG ORCHESTRATOR (main.zig) ────────────────────┐
│ export fn init(), frame(dt), resize(w, h)                    │
│ Reads input buffer → updates gesture FSM → physics → cull    │
└─────────────────────────────────────────────────────────────┘

┌─── LAYER 3: ZIG PHYSICS ENGINE ─────────────────────────────┐
│ gesture.zig — touch FSM (idle→pressed→dragging→fling→snap)   │
│ physics.zig — friction deceleration + critically damped spring│
│ stack.zig   — depth stack layout (position/scale/opacity)     │
└─────────────────────────────────────────────────────────────┘

┌─── LAYER 2: ZIG MATH + SIMD ───────────────────────────────┐
│ mat4.zig — 4×4 matrix: perspective, lookAt, transform        │
│ vec.zig  — Vec3/Vec4 via @Vector(4, f32) → WASM 128-bit SIMD│
│ ease.zig — easing functions                                   │
└─────────────────────────────────────────────────────────────┘

┌─── LAYER 1: MEMORY LAYOUT ──────────────────────────────────┐
│ Card transforms[64] — mat4 pre-allocated                     │
│ Input ring buffer — 64 slots × 16 bytes = 1KB               │
│ Render state — visible card list + texture bindings           │
│ Zero allocations in hot path                                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Per Frame (~16.67ms budget)

1. **JS** (0.5ms): RAF fires → write touch events to shared memory → call `zig.frame(dt)`
2. **Zig** (0.5ms): Read input ring buffer → update gesture FSM
3. **Zig** (0.5ms): Run physics (spring/friction) → new scroll position
4. **Zig** (0.5ms): Calculate 12 card transforms (SIMD mat4 multiply)
5. **Zig** (0.2ms): Frustum cull → write visible card list to shared memory
6. **JS** (0.5ms): Read render state → bind textures → set uniforms → drawArraysInstanced
7. **GPU** (5ms): Vertex shader → Fragment shader → present

**Total: ~8ms | Margin: ~8ms at 60fps**

## Depth Stack Rendering

### Card Layout Formulas

For card at index `i` (0 = focused front card):

```
z_position    = base_z - i * depth_step    (depth_step increases exponentially)
scale         = 1.0 - i * 0.08
opacity       = 1.0 - i * 0.075
y_offset      = i * card_gap               (vertical shift upward)
```

### Rendering Pipeline

- **Geometry**: Single quad (4 vertices, 6 indices). Reused for ALL cards via instancing.
- **Instancing**: 12 mat4 transforms in instance buffer. **1 drawArraysInstanced call per frame.**
- **Textures**: `sampler2DArray` — 1 binding, GPU indexes by card layer.
- **Shaders**: Vertex applies per-instance mat4. Fragment samples texture array, applies opacity + depth effects.
- **State changes per frame**: **0** (static pipeline, only instance buffer updates).

### Progressive Image Loading (4 tiers)

| Tier | Resolution | Size | When Loaded | Target Cards |
|------|-----------|------|-------------|--------------|
| 0 | Dominant color (extracted in JS via 1×1 canvas drawImage) | 0 bytes | Instant | All |
| 1 | 64×64 thumbnail | ~2KB | On init | All visible |
| 2 | 512×512 medium | ~80KB | When near viewport | Cards 1-5 |
| 3 | Full resolution | ~500KB | On demand | Focused card (0) |

Far cards (6-11) stay at Tier 1 — they're tiny on screen.

## Physics & Gesture System

### Gesture State Machine

```
IDLE → (touch_start) → PRESSED → (move > 8px) → DRAGGING
DRAGGING → (touch_end, v > threshold) → FLINGING → (v < snap_threshold) → SNAPPING → (settled) → IDLE
DRAGGING → (touch_end, v ≤ threshold) → SNAPPING (skip fling)
PRESSED → (touch_end, no drag) → TAP (select/expand card)
FLINGING/SNAPPING → (touch_start) → DRAGGING (interrupt, catch mid-motion)
```

### Physics Parameters

**Drag**:
- Sensitivity: 1.0 (1px finger = 1px card movement)
- Velocity window: last 5 samples (smoothed)
- Dead zone: 8px

**Fling**:
- Friction: 0.94 per frame (exponential decay)
- Min velocity: 50px/s (trigger threshold)
- Max velocity: 8000px/s (clamp)
- Settling time: ~300ms

**Snap (Critically Damped Spring)**:
- Stiffness: 300
- Damping ratio: 1.0 (fastest settle, no bounce)
- Settle epsilon: 0.5px
- Settling time: ~200ms

**Total gesture→settled: ~500ms**

### Input Pipeline

Shared memory ring buffer (lock-free):

```zig
const InputEvent = extern struct {  // 16 bytes, cache-line friendly
    event_type: u8,   // 0=none, 1=start, 2=move, 3=end
    _pad: [3]u8,
    x: f32,           // screen-space X
    y: f32,           // screen-space Y
    timestamp: f32,   // high-res time (ms)
};
// 64 slots × 16 bytes = 1024 bytes
// JS writes at head, Zig reads at tail
```

## File Structure

```
zig-image-carousel/
├── build.zig              # wasm32-freestanding target, ReleaseSmall
├── build.zig.zon          # package manifest
├── src/
│   ├── main.zig           # export fn init(), frame(dt), resize()
│   ├── stack.zig          # depth stack layout + card transforms
│   ├── gesture.zig        # touch FSM + velocity sampling
│   ├── physics.zig        # spring + friction + momentum
│   ├── mat4.zig           # 4×4 matrix ops (@Vector SIMD)
│   ├── vec.zig            # Vec3/Vec4 via @Vector(4, f32)
│   ├── ease.zig           # easing functions
│   └── memory.zig         # fixed allocator, ring buffer, shared layout
├── web/
│   ├── index.html         # canvas + minimal markup
│   ├── host.js            # WebGL2 driver + event marshaling (~200 LOC)
│   ├── shaders/
│   │   ├── card.vert      # instanced vertex shader
│   │   └── card.frag      # texture sampling + depth effects
│   └── images/            # sample photos for testing
└── tests/
    ├── test_mat4.zig      # matrix math unit tests
    ├── test_physics.zig   # spring/friction convergence tests
    └── test_gesture.zig   # FSM state transition tests
```

## Build Gates

### Gate 0: Foundation — "Does Zig talk to GPU?"

- **L1**: Scaffold — build.zig (wasm32-freestanding), index.html, host.js, main.zig
- **L2**: WebGL2 Init — JS creates context, Zig exports init() + frame(dt)
- **L3**: Single Quad — 1 colored quad rendered via Zig-computed transform
- **Exit**: Colored quad appears and rotates. WASM binary <50KB. 60fps.

### Gate 1: Stack — "Does the depth stack look right?"

- **L4**: SIMD Math — mat4.zig, vec.zig with @Vector(4, f32). Perspective + lookAt.
- **L5**: Instanced Cards — 12 quads via drawArraysInstanced. Depth stack layout.
- **L6**: Textures — Load placeholder images into texture array. Map to cards.
- **Exit**: 12 textured cards in depth stack. Static, no interaction. 60fps. Visually correct perspective.

### Gate 2: Interaction — "Does it feel good in the hand?"

- **L7**: Input Pipeline — Ring buffer, JS→WASM touch marshaling, mouse adapter.
- **L8**: Gesture FSM — All state transitions implemented and tested.
- **L9**: Physics — Friction deceleration + critically damped spring snap.
- **Exit**: Drag with finger/mouse. Smooth 60fps. Fling decelerates. Snaps to card. Interruptible. iOS-quality feel.

### Gate 3: Content — "Does it handle real photos gracefully?"

- **L10**: Progressive Loading — 4-tier system. LOD by distance.
- **L11**: Texture Streaming — Async upload, idle-frame scheduling, LRU eviction.
- **L12**: Visibility Culling — Only process visible cards.
- **Exit**: 50+ real photos. Rapid scroll. No jank. Progressive reveal. Memory <100MB.

### Gate 4: Polish — "Would a Facebook designer ship this?"

- **L13**: Visual Effects — Depth-of-field blur, subtle shadows, card edge glow.
- **L14**: Transitions — Card enter/exit animations. Smooth focus transitions.
- **L15**: Responsive — Resize handling, DPI awareness, mobile/desktop adaptation.
- **Exit**: Indistinguishable from native iOS. Lighthouse >90. Screenshot comparison passes.

## Testing Strategy

### Tier 1: Zig Unit Tests (`zig build test`)

- mat4: identity, multiply, perspective, lookAt — epsilon 1e-5 comparison
- vec: normalize, dot, cross, length — SIMD correctness
- physics: spring converges within N frames, friction reaches zero, no NaN
- gesture: every FSM edge, velocity sampling accuracy, ring buffer wrap
- stack: card positions at scroll=0/0.5/max, Z-ordering, cull thresholds

### Tier 2: Visual Verification (Chrome DevTools)

- Frame timing: every frame <16.67ms, no spikes >20ms
- GPU: 1 draw call/frame, no redundant state changes
- Touch: first input delay <50ms, drag latency <1 frame
- Memory: heap stays flat during scroll, WASM memory <4MB
- Lighthouse: performance >90

### Tier 3: Integration Tests (Chrome DevTools MCP)

- Render: navigate → screenshot → verify cards visible
- Interaction: simulate touch drag → verify scroll position changed
- Performance: trace → rapid scroll 20 cards → assert p95 frame <16.67ms
- Loading: throttled network → verify progressive loading stages

## Key Constraints & Physical Limits

- **WASM SIMD**: 128-bit only (4×f32). All matrix ops fit in single SIMD pass.
- **WASM memory**: Linear, grows in 64KB pages. Pre-allocate everything. No shrinking.
- **JS↔WASM boundary**: Only i32/i64/f32/f64. Complex data via shared memory pointers.
- **GPU frame budget**: 16.67ms total. Zig+JS must finish in <8ms to leave GPU headroom.
- **Texture upload**: Async via createImageBitmap + texSubImage2D. Never block main thread.
- **Touch sampling**: 120Hz on modern devices → 60Hz frames. Ring buffer absorbs burst.
