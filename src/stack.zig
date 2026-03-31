///! Depth Stack Layout Engine -- Gate 1
///!
///! Computes transforms, opacities, and texture indices for up to 12 cards
///! arranged in a depth-receding stack formation. The front card sits near
///! the camera; successive cards shift up in Y and back in Z with decreasing
///! scale and opacity, producing a cascading "deck" look.
///!
///! Layout formula for card at effective index `i`:
///!   z     = -depth_step_base * (pow(depth_step_exp, i) - 1) / (depth_step_exp - 1)
///!   y     = i * y_gap
///!   scale = max(0.3, 1.0 - i * scale_decay)
///!   alpha = max(0.1, 1.0 - i * opacity_decay)
///!   model = translate(0, y, z) * scaleUniform(scale)
///!
///! The scroll_offset shifts all effective indices, enabling smooth scrolling
///! through the stack in Gate 2.

const std = @import("std");
const math = std.math;
const m4 = @import("mat4.zig");
const Mat4 = m4.Mat4;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of cards in the stack.
pub const MAX_CARDS: u32 = 12;

/// Number of extra cards beyond the visible range to keep alive during
/// layout. This viewport margin prevents popping artifacts when cards
/// are about to scroll into view -- they already have valid transforms
/// and opacities computed, so the transition is seamless.
pub const VIEWPORT_MARGIN: f32 = 2.0;

// ---------------------------------------------------------------------------
// Layout parameters
// ---------------------------------------------------------------------------

/// Controls the base distance between cards along Z.
const depth_step_base: f32 = 0.4;

/// Exponential growth factor for Z spacing -- cards further back are
/// spaced more widely, creating a natural perspective depth effect.
const depth_step_exp: f32 = 1.15;

/// How much to reduce scale per card index (clamped at 0.3 minimum).
const scale_decay: f32 = 0.08;

/// How much to reduce opacity per card index (clamped at 0.1 minimum).
const opacity_decay: f32 = 0.075;

/// Vertical gap between successive cards -- slight upward shift so
/// each card peeks above the one in front.
const y_gap: f32 = 0.12;

// ---------------------------------------------------------------------------
// CardState
// ---------------------------------------------------------------------------

/// Per-card computed state after layout. Contains everything the renderer
/// needs to draw one card instance.
pub const CardState = struct {
    /// The 4x4 model-to-world transform (translate + uniform scale).
    transform: Mat4,
    /// Alpha opacity in [0.1, 1.0] -- decreases for cards further back.
    opacity: f32,
    /// Index into the texture array (0-based). Maps directly to the card's
    /// original position in the deck, NOT its visual position.
    texture_index: u32,
    /// Whether this card should be rendered. Cards scrolled off-screen
    /// (effective index < -VIEWPORT_MARGIN or >= MAX_CARDS + VIEWPORT_MARGIN)
    /// are culled. The margin provides a buffer for smooth scroll transitions.
    visible: bool,
};

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

/// The depth stack manages up to MAX_CARDS cards and computes their layout
/// each frame based on the current scroll offset.
pub const Stack = struct {
    /// Per-card state array. Only indices 0..num_cards-1 are meaningful.
    cards: [MAX_CARDS]CardState,
    /// How many cards are in the deck (may be less than MAX_CARDS).
    num_cards: u32,
    /// Fractional scroll offset -- 0.0 means card 0 is at the front.
    /// Increasing scroll_offset moves the stack so later cards come forward.
    scroll_offset: f32,

    /// Create a stack with `n` cards, all at default state.
    pub fn init(n: u32) Stack {
        var self = Stack{
            .cards = undefined,
            .num_cards = if (n > MAX_CARDS) MAX_CARDS else n,
            .scroll_offset = 0,
        };
        // Initialise every card slot
        for (0..MAX_CARDS) |idx| {
            self.cards[idx] = .{
                .transform = Mat4.identity,
                .opacity = 1.0,
                .texture_index = @intCast(idx),
                .visible = idx < self.num_cards,
            };
        }
        return self;
    }

    /// Recompute transforms, opacities, and visibility for all cards
    /// based on the current scroll_offset and layout parameters.
    ///
    /// Each card's "effective index" is its deck index minus scroll_offset.
    /// Cards with effective index in [-VIEWPORT_MARGIN, MAX_CARDS + VIEWPORT_MARGIN)
    /// are visible; the rest are culled. The margin keeps nearby off-screen
    /// cards ready so scrolling transitions are seamless.
    pub fn computeLayout(self: *Stack) void {
        for (0..self.num_cards) |idx| {
            const i_f32: f32 = @floatFromInt(idx);
            const effective_i = i_f32 - self.scroll_offset;

            // --- Z depth (exponential spacing) ---
            // depth = base * (exp^eff_i - 1) / (exp - 1)
            // This gives linear-ish spacing near the front that increases
            // exponentially deeper into the stack.
            const depth = depth_step_base *
                (math.pow(f32, depth_step_exp, effective_i) - 1.0) /
                (depth_step_exp - 1.0);
            const z = -depth;

            // --- Y offset (slight upward shift per card) ---
            const y = effective_i * y_gap;

            // --- Scale (shrink with distance) ---
            const s = @max(0.3, 1.0 - effective_i * scale_decay);

            // --- Opacity (fade with distance) ---
            const alpha = @max(0.1, 1.0 - effective_i * opacity_decay);

            // --- Visibility (viewport culling with margin) ---
            // Cards outside [-VIEWPORT_MARGIN, MAX_CARDS + VIEWPORT_MARGIN) are
            // culled. The margin keeps nearby off-screen cards alive so they
            // have valid transforms when scrolling brings them into view.
            const upper_bound = @as(f32, @floatFromInt(MAX_CARDS)) + VIEWPORT_MARGIN;
            const visible = effective_i >= -VIEWPORT_MARGIN and effective_i < upper_bound;

            // --- Subtle X-axis tilt for receding cards ---
            // Cards further back tilt slightly, creating a "fanning out"
            // perspective effect that reinforces the depth stack illusion.
            const tilt = effective_i * 0.015;

            // --- Model matrix: translate, tilt, then scale ---
            const model = Mat4.translate(0, y, z)
                .mul(Mat4.rotateX(tilt))
                .mul(Mat4.scaleUniform(s));

            self.cards[idx] = .{
                .transform = model,
                .opacity = alpha,
                .texture_index = @intCast(idx),
                .visible = visible,
            };
        }
    }

    /// Write visible cards' model matrices into a flat f32 buffer.
    /// Each matrix occupies 16 floats (column-major). Returns the number
    /// of visible cards written, which the renderer uses as the instance count.
    pub fn writeTransforms(self: *const Stack, buf: []f32) u32 {
        var count: u32 = 0;
        for (0..self.num_cards) |idx| {
            if (self.cards[idx].visible) {
                const offset = count * 16;
                if (offset + 16 > buf.len) break;
                self.cards[idx].transform.writeTo(buf, offset);
                count += 1;
            }
        }
        return count;
    }

    /// Write visible cards' opacities into a flat f32 buffer.
    /// One float per visible card, in the same order as writeTransforms.
    pub fn writeOpacities(self: *const Stack, buf: []f32) void {
        var count: u32 = 0;
        for (0..self.num_cards) |idx| {
            if (self.cards[idx].visible) {
                if (count >= buf.len) break;
                buf[count] = self.cards[idx].opacity;
                count += 1;
            }
        }
    }

    /// Write visible cards' texture indices as f32 into a flat buffer.
    /// Stored as float because WebGL2 vertex attributes are floats.
    pub fn writeTexIndices(self: *const Stack, buf: []f32) void {
        var count: u32 = 0;
        for (0..self.num_cards) |idx| {
            if (self.cards[idx].visible) {
                if (count >= buf.len) break;
                buf[count] = @floatFromInt(self.cards[idx].texture_index);
                count += 1;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "card 0 is near origin at scroll=0" {
    var stack = Stack.init(MAX_CARDS);
    stack.computeLayout();

    const c0 = stack.cards[0];
    const arr = c0.transform.toArray();

    // Translation is in column 3 (indices 12, 13, 14)
    // At effective_i=0: y=0, z=-base*(1-1)/(exp-1) = 0, scale=1.0
    try std.testing.expectApproxEqAbs(@as(f32, 0), arr[12], 1e-5); // x=0
    try std.testing.expectApproxEqAbs(@as(f32, 0), arr[13], 1e-5); // y=0
    try std.testing.expectApproxEqAbs(@as(f32, 0), arr[14], 1e-5); // z=0

    // Scale should be 1.0 (diagonal)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), arr[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), arr[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), arr[10], 1e-5);

    // Opacity should be 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c0.opacity, 1e-5);
}

test "cards recede in Z" {
    var stack = Stack.init(MAX_CARDS);
    stack.computeLayout();

    // Each successive card should have a more negative Z
    var prev_z: f32 = stack.cards[0].transform.toArray()[14];
    for (1..MAX_CARDS) |idx| {
        const z = stack.cards[idx].transform.toArray()[14];
        try std.testing.expect(z < prev_z);
        prev_z = z;
    }
}

test "opacity decreases with index" {
    var stack = Stack.init(MAX_CARDS);
    stack.computeLayout();

    var prev_opacity: f32 = stack.cards[0].opacity;
    for (1..MAX_CARDS) |idx| {
        const opacity = stack.cards[idx].opacity;
        try std.testing.expect(opacity <= prev_opacity);
        prev_opacity = opacity;
    }
}

test "all 12 cards visible at scroll=0" {
    var stack = Stack.init(MAX_CARDS);
    stack.computeLayout();

    for (0..MAX_CARDS) |idx| {
        try std.testing.expect(stack.cards[idx].visible);
    }
}

test "writeTransforms returns correct count" {
    var stack = Stack.init(MAX_CARDS);
    stack.computeLayout();

    var buf: [MAX_CARDS * 16]f32 = undefined;
    const count = stack.writeTransforms(&buf);
    try std.testing.expectEqual(@as(u32, MAX_CARDS), count);
}

test "writeOpacities matches card opacities" {
    var stack = Stack.init(MAX_CARDS);
    stack.computeLayout();

    var buf: [MAX_CARDS]f32 = undefined;
    stack.writeOpacities(&buf);

    // All visible at scroll=0, so buf[i] should match cards[i].opacity
    for (0..MAX_CARDS) |idx| {
        try std.testing.expectApproxEqAbs(stack.cards[idx].opacity, buf[idx], 1e-6);
    }
}

test "writeTexIndices outputs float indices" {
    var stack = Stack.init(MAX_CARDS);
    stack.computeLayout();

    var buf: [MAX_CARDS]f32 = undefined;
    stack.writeTexIndices(&buf);

    for (0..MAX_CARDS) |idx| {
        const expected: f32 = @floatFromInt(idx);
        try std.testing.expectApproxEqAbs(expected, buf[idx], 1e-6);
    }
}

test "scale decreases with index" {
    var stack = Stack.init(MAX_CARDS);
    stack.computeLayout();

    // Card 0 scale=1.0, card 1 scale=0.92, etc.
    // Check via the diagonal of the transform matrix
    var prev_scale: f32 = stack.cards[0].transform.toArray()[0];
    for (1..MAX_CARDS) |idx| {
        const s = stack.cards[idx].transform.toArray()[0];
        try std.testing.expect(s <= prev_scale);
        prev_scale = s;
    }
}

test "scroll offset shifts effective indices" {
    var stack = Stack.init(MAX_CARDS);
    stack.scroll_offset = 2.0;
    stack.computeLayout();

    // Card 0 is at effective_i = -2, exactly at -VIEWPORT_MARGIN boundary,
    // so still visible (the margin keeps nearby off-screen cards alive).
    try std.testing.expect(stack.cards[0].visible);
    // Card 1 is at effective_i = -1, within margin, visible
    try std.testing.expect(stack.cards[1].visible);
    // Card 2 is at effective_i = 0, visible and near origin
    try std.testing.expect(stack.cards[2].visible);
    const arr = stack.cards[2].transform.toArray();
    try std.testing.expectApproxEqAbs(@as(f32, 0), arr[14], 1e-5); // z ~ 0
}

test "viewport culling hides cards beyond margin" {
    // With 12 cards and scroll_offset=5, effective indices are:
    //   card 0: eff=-5 (culled, < -2)
    //   card 1: eff=-4 (culled, < -2)
    //   card 2: eff=-3 (culled, < -2)
    //   card 3: eff=-2 (visible, at margin boundary)
    //   card 4: eff=-1 (visible)
    //   card 5: eff= 0 (visible, front)
    //   ...
    //   card 11: eff= 6 (visible, < 14)
    // So cards 0..2 should be culled, cards 3..11 visible.
    var stack = Stack.init(MAX_CARDS);
    stack.scroll_offset = 5.0;
    stack.computeLayout();

    // Cards 0, 1, 2 are beyond the viewport margin -- culled
    try std.testing.expect(!stack.cards[0].visible);
    try std.testing.expect(!stack.cards[1].visible);
    try std.testing.expect(!stack.cards[2].visible);

    // Card 3 is at effective_i = -2 (exactly at boundary) -- visible
    try std.testing.expect(stack.cards[3].visible);

    // Cards 4..11 are all within range -- visible
    for (4..MAX_CARDS) |idx| {
        try std.testing.expect(stack.cards[idx].visible);
    }

    // writeTransforms should return 9 visible cards (indices 3..11)
    var buf: [MAX_CARDS * 16]f32 = undefined;
    const count = stack.writeTransforms(&buf);
    try std.testing.expectEqual(@as(u32, 9), count);
}

test "visible count matches writeTransforms at various scroll positions" {
    var stack = Stack.init(MAX_CARDS);

    // At scroll=0, all 12 cards should be visible (eff 0..11, all in [-2, 14))
    stack.scroll_offset = 0.0;
    stack.computeLayout();
    var buf: [MAX_CARDS * 16]f32 = undefined;
    try std.testing.expectEqual(@as(u32, 12), stack.writeTransforms(&buf));

    // At scroll=3, effective indices are -3..8. Card 0 (eff=-3) is culled.
    stack.scroll_offset = 3.0;
    stack.computeLayout();
    try std.testing.expectEqual(@as(u32, 11), stack.writeTransforms(&buf));
}
