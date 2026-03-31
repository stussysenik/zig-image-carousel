///! Input Ring Buffer -- Gate 2
///!
///! Lock-free single-producer (JS) / single-consumer (Zig) ring buffer for
///! touch and mouse events. JS writes events into WASM linear memory via
///! known pointers; the Zig frame loop drains them each tick.
///!
///! The ring buffer is safe without locks because there is exactly one writer
///! (the JS main thread, which pushes events into `input_head`) and one reader
///! (the Zig `frame()` function, which advances `input_tail`). Neither side
///! ever touches the other's index.
///!
///! Exported pointers (WASM boundary):
///!   getInputRingPtr()  -- base address of the event ring
///!   getInputHeadPtr()  -- address of the write cursor (JS increments)
///!   getInputTailPtr()  -- address of the read cursor  (Zig increments)
///!
///! Each event is 16 bytes (extern struct with explicit padding) so JS can
///! write fields at fixed byte offsets using DataView.

const std = @import("std");

// ---------------------------------------------------------------------------
// Event types -- encoded as a u8 matching the JS enum
// ---------------------------------------------------------------------------

/// Classifies a pointer event by its lifecycle phase.
///   - `none`:  sentinel / empty slot (never produced by JS)
///   - `start`: touchstart or mousedown  -- finger/button went down
///   - `move`:  touchmove  or mousemove  -- pointer is dragging
///   - `end`:   touchend   or mouseup    -- pointer was released
pub const EventType = enum(u8) {
    none = 0,
    start = 1,
    move = 2,
    end = 3,
};

/// A single input event as written by the JS producer.
///
/// Layout (16 bytes, little-endian):
///   offset 0:  u8    event_type
///   offset 1:  [3]u8 padding
///   offset 4:  f32   x  (CSS pixels)
///   offset 8:  f32   y  (CSS pixels)
///   offset 12: f32   timestamp (ms, from performance.now())
///
/// The `extern struct` qualifier guarantees C ABI layout so JS can write
/// fields at deterministic byte offsets.
pub const InputEvent = extern struct {
    event_type: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
    x: f32,
    y: f32,
    timestamp: f32,
};

// Compile-time guarantee: InputEvent must be exactly 16 bytes so the JS
// side can index slots as `ringPtr + (index * 16)`.
comptime {
    if (@sizeOf(InputEvent) != 16) {
        @compileError("InputEvent must be 16 bytes");
    }
}

// ---------------------------------------------------------------------------
// Ring buffer constants
// ---------------------------------------------------------------------------

/// Number of slots in the ring. Must be a power of two so we can use a
/// bitmask instead of modulo for wrapping.
const RING_SIZE: u32 = 64;

/// Bitmask for wrapping indices: `index & RING_MASK` == `index % RING_SIZE`.
const RING_MASK: u32 = RING_SIZE - 1;

// ---------------------------------------------------------------------------
// Shared state -- lives in WASM linear memory
// ---------------------------------------------------------------------------

/// The event ring itself -- JS writes into slots, Zig reads them.
export var input_ring: [RING_SIZE]InputEvent = [_]InputEvent{.{
    .event_type = 0,
    ._pad = .{ 0, 0, 0 },
    .x = 0,
    .y = 0,
    .timestamp = 0,
}} ** RING_SIZE;

/// Write cursor -- only JS increments this (after writing a slot).
export var input_head: u32 = 0;

/// Read cursor -- only Zig increments this (after consuming a slot).
export var input_tail: u32 = 0;

// ---------------------------------------------------------------------------
// Exported pointer accessors (for JS to locate the ring in WASM memory)
// ---------------------------------------------------------------------------

/// Returns the byte address of the ring buffer in WASM linear memory.
/// JS uses this as the base for DataView writes.
export fn getInputRingPtr() usize {
    return @intFromPtr(&input_ring);
}

/// Returns the byte address of `input_head` so JS can read/write the
/// producer cursor via DataView.
export fn getInputHeadPtr() usize {
    return @intFromPtr(&input_head);
}

/// Returns the byte address of `input_tail` so JS can read the consumer
/// cursor (to detect a full ring and drop events gracefully).
export fn getInputTailPtr() usize {
    return @intFromPtr(&input_tail);
}

// ---------------------------------------------------------------------------
// Consumer API (called from Zig frame loop)
// ---------------------------------------------------------------------------

/// Attempt to dequeue one event from the ring buffer.
///
/// Returns `null` when the ring is empty (tail == head). Otherwise returns
/// the oldest unread event and advances `input_tail` by one. The wrapping
/// add (`+%=`) handles u32 overflow identically to JS `>>> 0`.
///
/// Called in a `while (poll()) |ev|` loop from `frame()`.
pub fn poll() ?InputEvent {
    if (input_tail == input_head) return null;
    const event = input_ring[input_tail & RING_MASK];
    input_tail +%= 1;
    return event;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "InputEvent is 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(InputEvent));
}

test "ring starts empty" {
    // Reset state for test isolation
    input_head = 0;
    input_tail = 0;
    try std.testing.expect(poll() == null);
}

test "poll reads events written by simulated producer" {
    // Reset
    input_head = 0;
    input_tail = 0;

    // Simulate JS writing one event
    input_ring[0] = .{
        .event_type = @intFromEnum(EventType.start),
        ._pad = .{ 0, 0, 0 },
        .x = 100.0,
        .y = 200.0,
        .timestamp = 42.0,
    };
    input_head = 1;

    const ev = poll().?;
    try std.testing.expectEqual(@as(u8, 1), ev.event_type);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), ev.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 200.0), ev.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), ev.timestamp, 1e-6);

    // Ring should now be empty again
    try std.testing.expect(poll() == null);
}

test "poll wraps around RING_SIZE boundary" {
    // Set cursors near the wrap point
    input_head = RING_SIZE - 1;
    input_tail = RING_SIZE - 1;

    // Write two events: one at slot 63, one at slot 0 (wrapped)
    input_ring[(RING_SIZE - 1) & RING_MASK] = .{
        .event_type = @intFromEnum(EventType.move),
        ._pad = .{ 0, 0, 0 },
        .x = 1.0,
        .y = 2.0,
        .timestamp = 10.0,
    };
    input_ring[RING_SIZE & RING_MASK] = .{
        .event_type = @intFromEnum(EventType.end),
        ._pad = .{ 0, 0, 0 },
        .x = 3.0,
        .y = 4.0,
        .timestamp = 20.0,
    };
    input_head = RING_SIZE + 1;

    // Read first event (slot 63)
    const ev1 = poll().?;
    try std.testing.expectEqual(@as(u8, @intFromEnum(EventType.move)), ev1.event_type);

    // Read second event (slot 0, wrapped)
    const ev2 = poll().?;
    try std.testing.expectEqual(@as(u8, @intFromEnum(EventType.end)), ev2.event_type);

    // Should be empty
    try std.testing.expect(poll() == null);
}

test "exported pointers are non-zero and distinct" {
    const ring_ptr = getInputRingPtr();
    const head_ptr = getInputHeadPtr();
    const tail_ptr = getInputTailPtr();

    try std.testing.expect(ring_ptr != 0);
    try std.testing.expect(head_ptr != 0);
    try std.testing.expect(tail_ptr != 0);

    // All three should be at different addresses
    try std.testing.expect(ring_ptr != head_ptr);
    try std.testing.expect(ring_ptr != tail_ptr);
    try std.testing.expect(head_ptr != tail_ptr);
}
