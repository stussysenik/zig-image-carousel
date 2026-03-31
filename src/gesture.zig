///! Gesture State Machine -- Gate 2
///!
///! A 5-state finite state machine that converts raw touch/mouse input events
///! (from input.zig's ring buffer) into scroll position, velocity, and snap
///! targets. This is the bridge between raw pointer events and the smooth
///! card-scrolling experience.
///!
///! States:
///!   idle     -- no touch, waiting for input
///!   pressed  -- finger is down, waiting to see if this is a tap or drag
///!   dragging -- actively scrolling; updates scroll_position each move
///!   flinging -- finger lifted with velocity; physics will animate (Task 6)
///!   snapping -- settling to nearest card boundary; physics will animate
///!
///! Scroll position is measured in "card units" where 0 = first card,
///! 1 = second card, etc. Pixel deltas are converted via CARD_SPACING
///! (200 pixels per card unit).
///!
///! Velocity sampling uses a circular buffer of the last 5 move deltas
///! to produce a smoothed velocity estimate, filtering out jitter.

const std = @import("std");
const math = std.math;
const input_mod = @import("input.zig");
const InputEvent = input_mod.InputEvent;
const EventType = input_mod.EventType;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Pixels per card unit. A 200px swipe scrolls exactly one card.
pub const CARD_SPACING: f32 = 200.0;

/// Dead zone in pixels. A move must exceed this threshold from the press
/// origin before we transition from pressed to dragging. This prevents
/// accidental drags during taps.
const DEAD_ZONE_PX: f32 = 8.0;

/// Minimum fling velocity in pixels/second. If the finger lifts with
/// velocity below this, we skip the fling state and snap directly.
const FLING_THRESHOLD_PX_PER_SEC: f32 = 50.0;

/// Number of velocity samples to keep in the circular buffer.
const VELOCITY_BUFFER_SIZE: u32 = 5;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// FSM states for the gesture recogniser.
pub const State = enum {
    idle,
    pressed,
    dragging,
    flinging,
    snapping,
};

/// A single velocity sample: how far (dy) the pointer moved in how much
/// time (dt). Both are in raw units (pixels and milliseconds).
const VelocitySample = struct {
    dy: f32,
    dt: f32,
};

// ---------------------------------------------------------------------------
// GestureState
// ---------------------------------------------------------------------------

/// The gesture finite state machine. Processes InputEvents and produces
/// scroll_position (in card units) that drives the depth stack layout.
pub const GestureState = struct {
    /// Current FSM state.
    state: State,

    /// Scroll position in card units (0 = first card, 1 = second, ...).
    /// This is the value consumed by stack.scroll_offset each frame.
    scroll_position: f32,

    /// Scroll velocity in card-units per second. Positive = scrolling
    /// forward (swiping up reveals later cards).
    scroll_velocity: f32,

    /// The nearest integer card position to snap to, clamped to valid range.
    snap_target: f32,

    /// Set to true when a tap is detected (press + release without drag).
    /// The consumer should read and clear this flag each frame.
    tap_fired: bool,

    /// Total number of cards, used to clamp snap targets.
    card_count: u32,

    // -- Internal tracking --

    /// Y-coordinate where the current press started (pixels).
    press_y: f32,

    /// Timestamp of the press start (ms, from performance.now()).
    press_timestamp: f32,

    /// Y-coordinate of the most recent move event (pixels).
    last_y: f32,

    /// Timestamp of the most recent move event (ms).
    last_timestamp: f32,

    /// Circular buffer of recent velocity samples for smoothing.
    velocity_samples: [VELOCITY_BUFFER_SIZE]VelocitySample,

    /// Write index into velocity_samples (wraps around).
    velocity_index: u32,

    /// Number of valid samples in the buffer (up to VELOCITY_BUFFER_SIZE).
    velocity_count: u32,

    /// Create a gesture state for a deck with `card_count` cards.
    pub fn init(card_count: u32) GestureState {
        return .{
            .state = .idle,
            .scroll_position = 0,
            .scroll_velocity = 0,
            .snap_target = 0,
            .tap_fired = false,
            .card_count = card_count,
            .press_y = 0,
            .press_timestamp = 0,
            .last_y = 0,
            .last_timestamp = 0,
            .velocity_samples = [_]VelocitySample{.{ .dy = 0, .dt = 0 }} ** VELOCITY_BUFFER_SIZE,
            .velocity_index = 0,
            .velocity_count = 0,
        };
    }

    // ---------------------------------------------------------------------------
    // Main FSM dispatch
    // ---------------------------------------------------------------------------

    /// Process a single input event through the gesture FSM.
    ///
    /// This is the heart of the gesture recogniser. It reads the event type
    /// and delegates to the appropriate state handler, implementing the
    /// transition table documented in the module header.
    pub fn processEvent(self: *GestureState, raw: InputEvent) void {
        const event_type: EventType = @enumFromInt(raw.event_type);

        switch (self.state) {
            .idle => self.handleIdle(event_type, raw),
            .pressed => self.handlePressed(event_type, raw),
            .dragging => self.handleDragging(event_type, raw),
            .flinging => self.handleFlinging(event_type, raw),
            .snapping => self.handleSnapping(event_type, raw),
        }
    }

    // ---------------------------------------------------------------------------
    // State handlers
    // ---------------------------------------------------------------------------

    /// idle: waiting for a touch/click to start.
    fn handleIdle(self: *GestureState, event_type: EventType, raw: InputEvent) void {
        if (event_type == .start) {
            self.state = .pressed;
            self.press_y = raw.y;
            self.press_timestamp = raw.timestamp;
            self.last_y = raw.y;
            self.last_timestamp = raw.timestamp;
            self.clearVelocityBuffer();
        }
    }

    /// pressed: finger is down but hasn't moved beyond the dead zone yet.
    /// Could become a drag or a tap.
    fn handlePressed(self: *GestureState, event_type: EventType, raw: InputEvent) void {
        switch (event_type) {
            .move => {
                const dy = raw.y - self.press_y;
                if (@abs(dy) > DEAD_ZONE_PX) {
                    // Exceeded dead zone -- transition to dragging.
                    self.state = .dragging;
                    // Record this as the first move for velocity tracking,
                    // using the delta from the press origin.
                    self.addVelocitySample(raw.y - self.last_y, raw.timestamp - self.last_timestamp);
                    self.last_y = raw.y;
                    self.last_timestamp = raw.timestamp;
                    // Update scroll position for this initial move
                    self.applyDragDelta(dy);
                    // Reset press_y so subsequent deltas are relative
                    self.press_y = raw.y;
                }
            },
            .end => {
                // Released without exceeding dead zone -- this is a tap.
                self.state = .idle;
                self.tap_fired = true;
            },
            .start => {
                // Another start while pressed -- reset press origin.
                self.press_y = raw.y;
                self.press_timestamp = raw.timestamp;
                self.last_y = raw.y;
                self.last_timestamp = raw.timestamp;
            },
            .none => {},
        }
    }

    /// dragging: finger is actively moving and we're updating scroll_position.
    fn handleDragging(self: *GestureState, event_type: EventType, raw: InputEvent) void {
        switch (event_type) {
            .move => {
                const dy = raw.y - self.last_y;
                const dt = raw.timestamp - self.last_timestamp;
                self.addVelocitySample(dy, dt);
                self.last_y = raw.y;
                self.last_timestamp = raw.timestamp;
                self.applyDragDelta(dy);
            },
            .end => {
                // Finger lifted -- check velocity to decide fling vs snap.
                const smoothed = self.computeSmoothedVelocity();
                if (@abs(smoothed) > FLING_THRESHOLD_PX_PER_SEC / CARD_SPACING) {
                    // Velocity exceeds threshold -- fling.
                    self.scroll_velocity = smoothed;
                    self.state = .flinging;
                } else {
                    // Low velocity -- snap directly to nearest card.
                    self.scroll_velocity = 0;
                    self.beginSnap();
                }
            },
            .start => {
                // New touch while dragging -- just update tracking.
                self.last_y = raw.y;
                self.last_timestamp = raw.timestamp;
                self.clearVelocityBuffer();
            },
            .none => {},
        }
    }

    /// flinging: finger was lifted with velocity. Physics (Task 6) will
    /// decelerate scroll_velocity and eventually transition to snapping.
    /// For now, a new touch interrupts the fling immediately.
    fn handleFlinging(self: *GestureState, event_type: EventType, raw: InputEvent) void {
        if (event_type == .start) {
            // Interrupt fling -- grab the scroll where it is.
            self.state = .dragging;
            self.scroll_velocity = 0;
            self.press_y = raw.y;
            self.press_timestamp = raw.timestamp;
            self.last_y = raw.y;
            self.last_timestamp = raw.timestamp;
            self.clearVelocityBuffer();
        }
    }

    /// snapping: settling to the nearest card position. Physics (Task 6)
    /// will animate this. A new touch interrupts the snap.
    fn handleSnapping(self: *GestureState, event_type: EventType, raw: InputEvent) void {
        if (event_type == .start) {
            // Interrupt snap -- start dragging from current position.
            self.state = .dragging;
            self.scroll_velocity = 0;
            self.press_y = raw.y;
            self.press_timestamp = raw.timestamp;
            self.last_y = raw.y;
            self.last_timestamp = raw.timestamp;
            self.clearVelocityBuffer();
        }
    }

    // ---------------------------------------------------------------------------
    // Velocity computation
    // ---------------------------------------------------------------------------

    /// Compute smoothed velocity from the circular buffer of recent samples.
    ///
    /// Returns velocity in card-units/second. The sign convention is:
    /// swipe up (negative pixel dy) = positive scroll (reveals later cards).
    ///
    /// Algorithm: sum all dy and dt in the buffer, then:
    ///   velocity = -(total_dy / total_dt) * 1000 / CARD_SPACING
    ///
    /// The *1000 converts from per-millisecond to per-second.
    /// The negation converts screen coordinates (y-down) to scroll direction
    /// (swipe up = positive scroll).
    pub fn computeSmoothedVelocity(self: *const GestureState) f32 {
        if (self.velocity_count == 0) return 0;

        var total_dy: f32 = 0;
        var total_dt: f32 = 0;
        const count = @min(self.velocity_count, VELOCITY_BUFFER_SIZE);

        for (0..count) |i| {
            total_dy += self.velocity_samples[i].dy;
            total_dt += self.velocity_samples[i].dt;
        }

        if (total_dt <= 0) return 0;

        // Negate because screen Y is inverted relative to scroll direction:
        // swiping up (negative dy) should produce positive scroll velocity.
        return -(total_dy / total_dt) * 1000.0 / CARD_SPACING;
    }

    // ---------------------------------------------------------------------------
    // Snap helpers
    // ---------------------------------------------------------------------------

    /// Round scroll_position to the nearest integer card index, clamped
    /// to the valid range [0, card_count - 1].
    pub fn nearestCardPosition(self: *const GestureState) f32 {
        if (self.card_count == 0) return 0;
        const max_pos: f32 = @floatFromInt(self.card_count - 1);
        const rounded = @round(self.scroll_position);
        return @max(0, @min(max_pos, rounded));
    }

    /// Begin snapping: set the snap target to the nearest card and
    /// transition to the snapping state.
    pub fn beginSnap(self: *GestureState) void {
        self.snap_target = self.nearestCardPosition();
        self.state = .snapping;
    }

    /// Immediately settle: jump scroll_position to snap_target, zero
    /// velocity, and return to idle. Used when snap animation completes
    /// (or for instant snapping before physics is wired in).
    pub fn settle(self: *GestureState) void {
        self.scroll_position = self.snap_target;
        self.scroll_velocity = 0;
        self.state = .idle;
    }

    // ---------------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------------

    /// Apply a pixel-space drag delta to scroll_position.
    /// Negated because swiping up (negative pixel delta) should increase
    /// scroll_position (reveal later cards).
    fn applyDragDelta(self: *GestureState, dy_pixels: f32) void {
        self.scroll_position += -dy_pixels / CARD_SPACING;
    }

    /// Add a velocity sample to the circular buffer.
    fn addVelocitySample(self: *GestureState, dy: f32, dt: f32) void {
        self.velocity_samples[self.velocity_index] = .{ .dy = dy, .dt = dt };
        self.velocity_index = (self.velocity_index + 1) % VELOCITY_BUFFER_SIZE;
        if (self.velocity_count < VELOCITY_BUFFER_SIZE) {
            self.velocity_count += 1;
        }
    }

    /// Reset the velocity buffer. Called at the start of each new gesture.
    fn clearVelocityBuffer(self: *GestureState) void {
        self.velocity_index = 0;
        self.velocity_count = 0;
        for (&self.velocity_samples) |*s| {
            s.dy = 0;
            s.dt = 0;
        }
    }
};

// ---------------------------------------------------------------------------
// Helper: create an InputEvent for testing
// ---------------------------------------------------------------------------

fn makeEvent(event_type: EventType, x: f32, y: f32, timestamp: f32) InputEvent {
    return .{
        .event_type = @intFromEnum(event_type),
        ._pad = .{ 0, 0, 0 },
        .x = x,
        .y = y,
        .timestamp = timestamp,
    };
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "idle transitions to pressed on start" {
    var g = GestureState.init(12);
    try std.testing.expectEqual(State.idle, g.state);

    g.processEvent(makeEvent(.start, 100, 200, 0));
    try std.testing.expectEqual(State.pressed, g.state);
}

test "pressed transitions to dragging after dead zone exceeded" {
    var g = GestureState.init(12);
    g.processEvent(makeEvent(.start, 100, 200, 0));
    try std.testing.expectEqual(State.pressed, g.state);

    // Move within dead zone -- should stay pressed
    g.processEvent(makeEvent(.move, 100, 205, 10));
    try std.testing.expectEqual(State.pressed, g.state);

    // Move beyond dead zone (> 8px from press_y=200)
    g.processEvent(makeEvent(.move, 100, 210, 20));
    try std.testing.expectEqual(State.dragging, g.state);
}

test "tap detection: start then end without drag" {
    var g = GestureState.init(12);
    try std.testing.expect(!g.tap_fired);

    g.processEvent(makeEvent(.start, 100, 200, 0));
    g.processEvent(makeEvent(.end, 100, 200, 50));

    try std.testing.expectEqual(State.idle, g.state);
    try std.testing.expect(g.tap_fired);
}

test "interrupt fling with touch resets to dragging with zero velocity" {
    var g = GestureState.init(12);

    // Manually put into flinging state with some velocity
    g.state = .flinging;
    g.scroll_velocity = 5.0;

    // Touch down should interrupt the fling
    g.processEvent(makeEvent(.start, 100, 300, 1000));

    try std.testing.expectEqual(State.dragging, g.state);
    try std.testing.expectApproxEqAbs(@as(f32, 0), g.scroll_velocity, 1e-6);
}

test "interrupt snapping with touch resets to dragging" {
    var g = GestureState.init(12);

    // Manually put into snapping state
    g.state = .snapping;
    g.snap_target = 3.0;

    g.processEvent(makeEvent(.start, 100, 300, 1000));

    try std.testing.expectEqual(State.dragging, g.state);
    try std.testing.expectApproxEqAbs(@as(f32, 0), g.scroll_velocity, 1e-6);
}

test "drag updates scroll_position" {
    var g = GestureState.init(12);
    const initial_pos = g.scroll_position;

    // Start press
    g.processEvent(makeEvent(.start, 100, 200, 0));

    // Move beyond dead zone to start dragging (move up = positive scroll)
    g.processEvent(makeEvent(.move, 100, 190, 10)); // only 10px, within dead zone
    // Still pressed -- dead zone not exceeded from press_y=200
    // Actually 10px > 8px dead zone, so this transitions to dragging
    try std.testing.expectEqual(State.dragging, g.state);

    // Continue dragging upward (decreasing Y = swiping up = positive scroll)
    g.processEvent(makeEvent(.move, 100, 170, 20));

    // Scroll position should have increased (swiping up = positive scroll)
    try std.testing.expect(g.scroll_position > initial_pos);
}

test "nearestCardPosition clamps to valid range" {
    var g = GestureState.init(12);

    // At 0 -> nearest is 0
    g.scroll_position = 0.3;
    try std.testing.expectApproxEqAbs(@as(f32, 0), g.nearestCardPosition(), 1e-6);

    // At 2.7 -> nearest is 3
    g.scroll_position = 2.7;
    try std.testing.expectApproxEqAbs(@as(f32, 3), g.nearestCardPosition(), 1e-6);

    // Negative -> clamped to 0
    g.scroll_position = -1.5;
    try std.testing.expectApproxEqAbs(@as(f32, 0), g.nearestCardPosition(), 1e-6);

    // Beyond max -> clamped to card_count-1
    g.scroll_position = 15.0;
    try std.testing.expectApproxEqAbs(@as(f32, 11), g.nearestCardPosition(), 1e-6);
}

test "settle sets position to snap_target and returns to idle" {
    var g = GestureState.init(12);
    g.state = .snapping;
    g.snap_target = 5.0;
    g.scroll_velocity = 2.0;

    g.settle();

    try std.testing.expectEqual(State.idle, g.state);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), g.scroll_position, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), g.scroll_velocity, 1e-6);
}

test "computeSmoothedVelocity with no samples returns zero" {
    const g = GestureState.init(12);
    try std.testing.expectApproxEqAbs(@as(f32, 0), g.computeSmoothedVelocity(), 1e-6);
}

test "beginSnap sets correct snap target" {
    var g = GestureState.init(12);
    g.scroll_position = 3.4;
    g.state = .dragging;

    g.beginSnap();

    try std.testing.expectEqual(State.snapping, g.state);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), g.snap_target, 1e-6);
}

test "drag end with low velocity transitions to snapping" {
    var g = GestureState.init(12);

    // Start and drag slowly
    g.processEvent(makeEvent(.start, 100, 200, 0));
    g.processEvent(makeEvent(.move, 100, 189, 10)); // exceed dead zone
    try std.testing.expectEqual(State.dragging, g.state);

    // Slow move -- low velocity
    g.processEvent(makeEvent(.move, 100, 188, 500));

    // End -- velocity is very low, should snap
    g.processEvent(makeEvent(.end, 100, 188, 510));
    try std.testing.expectEqual(State.snapping, g.state);
}

test "drag end with high velocity transitions to flinging" {
    var g = GestureState.init(12);

    // Start and drag fast
    g.processEvent(makeEvent(.start, 100, 200, 0));
    g.processEvent(makeEvent(.move, 100, 189, 5)); // exceed dead zone, fast

    try std.testing.expectEqual(State.dragging, g.state);

    // Fast moves -- high velocity (large dy, small dt)
    g.processEvent(makeEvent(.move, 100, 139, 10));
    g.processEvent(makeEvent(.move, 100, 89, 15));
    g.processEvent(makeEvent(.move, 100, 39, 20));

    // End
    g.processEvent(makeEvent(.end, 100, 39, 21));
    try std.testing.expectEqual(State.flinging, g.state);
    try std.testing.expect(g.scroll_velocity > 0); // swiped up = positive velocity
}
