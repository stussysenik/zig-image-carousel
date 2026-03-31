///! Physics Engine -- Gate 2
///!
///! Frame-rate independent friction deceleration and critically damped spring
///! for animating the carousel's scroll position. This module bridges the gap
///! between the gesture FSM (which sets initial velocities and snap targets)
///! and the depth stack (which reads scroll_position each frame).
///!
///! Two update modes based on gesture state:
///!
///!   flinging -- Exponential friction decays scroll_velocity each frame.
///!     The friction factor is frame-rate independent: pow(0.94, dt / reference_dt).
///!     When velocity drops below a threshold, we transition to snapping.
///!
///!   snapping -- A critically damped spring pulls scroll_position toward
///!     the snap_target. Critical damping (damping_ratio = 1.0) ensures the
///!     fastest convergence without oscillation -- exactly what you want for
///!     a card carousel that should feel "magnetically" snapped.
///!
///! Both modes use semi-implicit Euler integration:
///!   1. Update velocity first (from forces/friction)
///!   2. Update position using the NEW velocity
///!   This is more stable than explicit Euler and nearly free.

const std = @import("std");
const math = std.math;
const gesture_mod = @import("gesture.zig");
const GestureState = gesture_mod.GestureState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Friction multiplier applied each reference frame (60 Hz). Values closer
/// to 1.0 mean less friction (longer coast). 0.94 gives a nice "heavy card"
/// feel -- momentum dies in about 1 second.
pub const FRICTION_PER_FRAME: f32 = 0.94;

/// The reference frame duration for friction calculation. Friction is defined
/// as "per 60 Hz frame" and then adjusted for the actual dt using pow().
/// This makes the deceleration curve identical regardless of frame rate.
pub const REFERENCE_DT: f32 = 1.0 / 60.0;

/// When scroll_velocity drops below this threshold (in card-units/second)
/// during a fling, we stop coasting and begin the snap-to-card spring.
/// 50 card-units/s at 200px/card = 10,000 px/s... but velocities are in
/// card-units, so 50 card-units/s is actually quite slow (~0.83 cards/frame).
pub const SNAP_VELOCITY_THRESHOLD: f32 = 50.0;

/// Spring stiffness for the snap animation. Higher = faster snap.
/// 300 gives a snappy feel without being jarring.
pub const SPRING_STIFFNESS: f32 = 300.0;

/// Damping ratio for the snap spring. 1.0 = critically damped (no overshoot).
/// < 1.0 would be underdamped (bouncy), > 1.0 would be overdamped (sluggish).
pub const DAMPING_RATIO: f32 = 1.0;

/// When both position error and velocity are below this epsilon during a
/// spring animation, we consider the snap complete and settle to idle.
pub const SETTLE_EPSILON: f32 = 0.005;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Per-frame physics update. Call once per frame after processing input events
/// and before computing the depth stack layout.
///
/// `gesture` -- mutable pointer to the gesture FSM (reads/writes position,
///              velocity, snap_target, and state).
/// `dt`      -- frame delta time in seconds.
pub fn update(gesture: *GestureState, dt: f32) void {
    switch (gesture.state) {
        .flinging => updateFriction(gesture, dt),
        .snapping => updateSpring(gesture, dt),
        else => {},
    }
}

// ---------------------------------------------------------------------------
// Friction (flinging state)
// ---------------------------------------------------------------------------

/// Exponential friction deceleration. Each reference frame (1/60s), velocity
/// is multiplied by FRICTION_PER_FRAME (0.94). For arbitrary dt we use:
///
///   factor = pow(0.94, dt / reference_dt)
///
/// This makes the deceleration curve identical at 30, 60, 120, or 144 Hz.
/// Position is updated using semi-implicit Euler (velocity first, then position
/// with the new velocity).
fn updateFriction(gesture: *GestureState, dt: f32) void {
    // Frame-rate independent friction: how many "reference frames" fit in dt?
    const frame_ratio = dt / REFERENCE_DT;
    const friction_factor = math.pow(f32, FRICTION_PER_FRAME, frame_ratio);

    // Apply friction to velocity
    gesture.scroll_velocity *= friction_factor;

    // Semi-implicit Euler: update position using the (already damped) velocity
    gesture.scroll_position += gesture.scroll_velocity * dt;

    // When velocity is negligible, stop coasting and begin snapping to
    // the nearest card boundary.
    if (@abs(gesture.scroll_velocity) < SNAP_VELOCITY_THRESHOLD) {
        gesture.beginSnap();
    }
}

// ---------------------------------------------------------------------------
// Critically Damped Spring (snapping state)
// ---------------------------------------------------------------------------

/// Critically damped spring that pulls scroll_position toward snap_target.
///
/// The spring equation:
///   acceleration = -stiffness * (position - target) - damping * velocity
///
/// where damping = 2 * damping_ratio * sqrt(stiffness).
///
/// Critical damping (ratio = 1.0) gives the fastest convergence without
/// any oscillation or overshoot. This is the same spring model used by
/// iOS's UIScrollView snap animation and Android's OverScroller.
///
/// Integration is semi-implicit Euler:
///   1. velocity += acceleration * dt
///   2. position += new_velocity * dt
fn updateSpring(gesture: *GestureState, dt: f32) void {
    const omega = @sqrt(SPRING_STIFFNESS);
    const damping = 2.0 * DAMPING_RATIO * omega;

    const displacement = gesture.scroll_position - gesture.snap_target;
    const acceleration = -SPRING_STIFFNESS * displacement - damping * gesture.scroll_velocity;

    // Semi-implicit Euler: update velocity first, then position with new velocity
    gesture.scroll_velocity += acceleration * dt;
    gesture.scroll_position += gesture.scroll_velocity * dt;

    // When both displacement and velocity are negligible, snap is complete.
    // Settle: jump to target, zero velocity, return to idle.
    if (@abs(gesture.scroll_position - gesture.snap_target) < SETTLE_EPSILON and
        @abs(gesture.scroll_velocity) < SETTLE_EPSILON)
    {
        gesture.settle();
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "friction reduces velocity" {
    // Set up a flinging gesture with high velocity, run one frame,
    // and verify velocity decreased.
    var g = GestureState.init(12);
    g.state = .flinging;
    g.scroll_velocity = 1000.0;

    const dt: f32 = 1.0 / 60.0;
    update(&g, dt);

    // Velocity should be reduced (but state may have transitioned to snapping
    // if it dropped below threshold -- that's fine, we just check the value).
    try std.testing.expect(g.scroll_velocity < 1000.0);
}

test "spring converges to target" {
    // Start snapping from position 1.5 toward target 2.0 with zero initial
    // velocity. After 600 frames at 60 Hz (10 seconds), the spring should
    // have settled: state = idle, position ≈ 2.0.
    var g = GestureState.init(12);
    g.state = .snapping;
    g.scroll_position = 1.5;
    g.snap_target = 2.0;
    g.scroll_velocity = 0.0;

    const dt: f32 = 1.0 / 60.0;
    for (0..600) |_| {
        update(&g, dt);
    }

    try std.testing.expectEqual(gesture_mod.State.idle, g.state);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), g.scroll_position, 0.01);
}

test "no NaN after extreme velocity fling" {
    // Stress test: start with a very high velocity and run 1000 frames.
    // Neither position nor velocity should ever become NaN.
    var g = GestureState.init(12);
    g.state = .flinging;
    g.scroll_velocity = 8000.0;

    const dt: f32 = 1.0 / 60.0;
    for (0..1000) |_| {
        update(&g, dt);
    }

    try std.testing.expect(!math.isNan(g.scroll_position));
    try std.testing.expect(!math.isNan(g.scroll_velocity));
}
