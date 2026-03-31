/**
 * host.js -- WebGL2 Instanced Rendering Driver for the Zig Image Carousel
 *
 * This is intentionally a *thin* GPU driver. All transform computation
 * happens in Zig; JS only:
 *   1. Loads the WASM module and wires up imports
 *   2. Creates the WebGL2 context, compiles shaders, sets up instancing
 *   3. Progressive texture loading: 4-tier system (color -> thumbnail ->
 *      medium -> full) with idle-frame scheduling and LRU eviction
 *   4. Each frame: reads matrices/opacities/texIndices from WASM memory,
 *      uploads into instance buffers, issues a single drawArraysInstanced call
 *
 * Architecture: "Maximum Zig" -- JS never computes a matrix.
 *
 * Progressive Loading (Gate 3):
 *   The texture system uses a 4-tier quality pyramid. Each card starts at
 *   Tier 0 (1x1 solid color) and progressively upgrades based on proximity
 *   to the scroll focus:
 *     Tier 0: 1x1 dominant color (instant, from CARD_COLORS)
 *     Tier 1: 64x64 gradient thumbnail
 *     Tier 2: 256x256 gradient with visual detail (shapes/noise)
 *     Tier 3: 512x512 full-resolution (only for focused card)
 *
 *   Loading is idle-frame scheduled: after Zig's frame() + GL draw, if
 *   there's frame budget remaining (<12ms of 16.6ms), one pending texture
 *   upload is performed. This prevents jank from texture uploads.
 *
 *   LRU eviction downgrades far-away cards from Tier 2 back to Tier 1
 *   when the total loaded texture count exceeds a budget (20 medium-res).
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_CARDS = 12;
const TEX_SIZE = 512; // Maximum texture resolution (per layer, Tier 3)

/**
 * Tier resolution map: each tier's texture dimensions.
 * Tier 0 = 1x1, Tier 1 = 64x64, Tier 2 = 256x256, Tier 3 = 512x512
 */
const TIER_SIZES = [1, 64, 256, 512];

/**
 * Maximum number of medium-resolution (Tier 2+) textures allowed before
 * LRU eviction kicks in. Keeps GPU memory bounded.
 */
const LRU_BUDGET = 20;

/**
 * Frame budget in milliseconds. If the frame (Zig + GL draw) completes
 * in less than this, we have headroom to upload one pending texture.
 * 12ms leaves 4ms headroom within a 16.6ms frame (60fps target).
 */
const FRAME_BUDGET_MS = 12;

/**
 * Placeholder card colors -- each card gets a distinct hue so the depth
 * stack is immediately visually distinguishable without real images.
 */
const CARD_COLORS = [
    [66, 133, 244],   // 1  Blue
    [219, 68, 55],    // 2  Red
    [244, 180, 0],    // 3  Yellow
    [15, 157, 88],    // 4  Green
    [255, 112, 67],   // 5  Orange
    [171, 71, 188],   // 6  Purple
    [0, 150, 136],    // 7  Teal
    [255, 87, 34],    // 8  Deep Orange
    [139, 195, 74],   // 9  Light Green
    [3, 169, 244],    // 10 Light Blue
    [255, 193, 7],    // 11 Amber
    [103, 58, 183],   // 12 Deep Purple
];

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

const canvas = document.getElementById("carousel-canvas");
const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
if (!gl) {
    document.body.textContent = "WebGL2 is required but not available.";
    throw new Error("WebGL2 not supported");
}

// ---------------------------------------------------------------------------
// Load WASM
// ---------------------------------------------------------------------------

/** @type {WebAssembly.Instance} */
let wasm;

/** @type {WebAssembly.Memory} */
let wasmMemory;

const importObject = {
    env: {
        /**
         * consoleLog -- called from Zig's log() helper.
         * Reads a UTF-8 string from WASM linear memory.
         */
        consoleLog(ptr, len) {
            const bytes = new Uint8Array(wasmMemory.buffer, ptr, len);
            const text = new TextDecoder().decode(bytes);
            console.log("[zig]", text);
        },
    },
};

// ---------------------------------------------------------------------------
// Shader compilation helpers
// ---------------------------------------------------------------------------

/**
 * Fetch a shader source file and compile it into a WebGL shader object.
 * @param {string} url
 * @param {number} type  gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
 * @returns {Promise<WebGLShader>}
 */
async function loadShader(url, type) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch shader: ${url}`);
    const source = await resp.text();
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(`Shader compile error (${url}):\n${info}`);
    }
    return shader;
}

/**
 * Link a vertex + fragment shader into a program.
 * @param {WebGLShader} vert
 * @param {WebGLShader} frag
 * @returns {WebGLProgram}
 */
function linkProgram(vert, frag) {
    const prog = gl.createProgram();
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(prog);
        gl.deleteProgram(prog);
        throw new Error(`Program link error:\n${info}`);
    }
    return prog;
}

// ---------------------------------------------------------------------------
// Progressive Texture System (Gate 3)
// ---------------------------------------------------------------------------

/**
 * Per-card gradient color pairs for generating beautiful placeholder images.
 * Each entry is [startColor, endColor] where colors are [r, g, b] arrays.
 * These produce visually distinct gradients across all 12 cards.
 */
const GRADIENT_PAIRS = [
    [[66, 133, 244],  [25, 72, 155]],    // 1  Ocean blue
    [[219, 68, 55],   [139, 20, 20]],     // 2  Crimson
    [[244, 180, 0],   [255, 111, 0]],     // 3  Sunset gold
    [[15, 157, 88],   [0, 90, 60]],       // 4  Emerald
    [[255, 112, 67],  [230, 60, 100]],    // 5  Coral rose
    [[171, 71, 188],  [90, 24, 120]],     // 6  Amethyst
    [[0, 150, 136],   [0, 77, 90]],       // 7  Deep teal
    [[255, 87, 34],   [180, 40, 10]],     // 8  Burnt sienna
    [[139, 195, 74],  [50, 120, 20]],     // 9  Forest lime
    [[3, 169, 244],   [1, 87, 155]],      // 10 Sky blue
    [[255, 193, 7],   [200, 120, 0]],     // 11 Warm amber
    [[103, 58, 183],  [40, 10, 100]],     // 12 Royal purple
];

/**
 * Per-card loading state tracking for the progressive texture system.
 *
 * Each card independently tracks which quality tier it's currently displaying,
 * what tier it wants to reach, and whether a load is in progress. This allows
 * the idle-frame scheduler to make per-frame decisions about which card to
 * upgrade next.
 *
 * @typedef {Object} CardTexState
 * @property {number} currentTier  - Currently uploaded tier (0-3)
 * @property {number} targetTier   - Desired tier based on scroll proximity
 * @property {boolean} loading     - True if an async load is in flight
 * @property {number} lastAccess   - Timestamp of last time this card was
 *                                   near focus (for LRU eviction)
 */

/** @type {CardTexState[]} */
const cardTexStates = [];

/**
 * Queue of pending texture uploads: { cardIndex, tier, imageData }.
 * Filled by async generateTierImage(), drained by the idle-frame uploader.
 * @type {Array<{cardIndex: number, tier: number, imageData: ImageBitmap|HTMLCanvasElement}>}
 */
const uploadQueue = [];

/**
 * Allocate the texture array at full resolution. All layers start as Tier 0
 * (1x1 solid color stretched by GL's LINEAR filter to fill the card).
 *
 * The TEXTURE_2D_ARRAY is allocated at TEX_SIZE x TEX_SIZE so that higher
 * tiers can be uploaded in-place via texSubImage3D without reallocation.
 *
 * @returns {WebGLTexture}
 */
function createTextureArray() {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);

    // Allocate full-resolution storage for all layers
    gl.texImage3D(
        gl.TEXTURE_2D_ARRAY,
        0,                   // mip level
        gl.RGBA8,            // internal format
        TEX_SIZE, TEX_SIZE,  // width, height (max tier size)
        MAX_CARDS,           // depth (number of layers)
        0,                   // border
        gl.RGBA,             // format
        gl.UNSIGNED_BYTE,    // type
        null                 // no data yet
    );

    // Upload Tier 0 for every card: 1x1 solid color.
    // GL's LINEAR filter stretches this to fill the entire card face,
    // giving an instant "dominant color" placeholder.
    const offscreen = document.createElement("canvas");
    offscreen.width = 1;
    offscreen.height = 1;
    const ctx = offscreen.getContext("2d");

    for (let i = 0; i < MAX_CARDS; i++) {
        const [r, g, b] = CARD_COLORS[i];
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(0, 0, 1, 1);

        gl.texSubImage3D(
            gl.TEXTURE_2D_ARRAY,
            0,             // mip level
            0, 0, i,       // x, y, layer
            1, 1, 1,       // width, height, depth
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            offscreen
        );

        // Initialise per-card state
        cardTexStates.push({
            currentTier: 0,
            targetTier: 0,
            loading: false,
            lastAccess: performance.now(),
        });
    }

    // Texture filtering: LINEAR gives smooth scaling when low-res textures
    // are displayed on larger quads (especially Tier 0's 1x1 pixel).
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
    return tex;
}

// ---------------------------------------------------------------------------
// Gradient image generation (off-thread via createImageBitmap)
// ---------------------------------------------------------------------------

/**
 * Generate a gradient image for a given card at a given tier resolution.
 *
 * Tier 0: Handled at init time (1x1 solid color) -- never called here.
 * Tier 1: 64x64 simple linear gradient between the card's color pair.
 * Tier 2: 256x256 gradient with decorative shapes and subtle noise.
 * Tier 3: 512x512 full-resolution with refined detail and card number.
 *
 * Uses an OffscreenCanvas (or fallback to regular canvas) to render the
 * gradient, then returns an ImageBitmap for off-thread decode.
 *
 * @param {number} cardIndex  Card index (0-11)
 * @param {number} tier       Target tier (1, 2, or 3)
 * @returns {Promise<ImageBitmap>}
 */
async function generateTierImage(cardIndex, tier) {
    const size = TIER_SIZES[tier];
    const [[r1, g1, b1], [r2, g2, b2]] = GRADIENT_PAIRS[cardIndex];

    const offscreen = document.createElement("canvas");
    offscreen.width = size;
    offscreen.height = size;
    const ctx = offscreen.getContext("2d");

    // Base gradient: diagonal sweep from top-left to bottom-right
    const grad = ctx.createLinearGradient(0, 0, size, size);
    grad.addColorStop(0, `rgb(${r1}, ${g1}, ${b1})`);
    grad.addColorStop(1, `rgb(${r2}, ${g2}, ${b2})`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);

    if (tier >= 2) {
        // Tier 2+: add decorative geometric shapes for visual richness.
        // These soft circles and arcs give each card a unique, appealing look
        // without requiring external images.
        ctx.globalCompositeOperation = "soft-light";

        // Large off-center radial glow
        const radGrad = ctx.createRadialGradient(
            size * 0.3, size * 0.4, 0,
            size * 0.3, size * 0.4, size * 0.7
        );
        radGrad.addColorStop(0, `rgba(255, 255, 255, 0.4)`);
        radGrad.addColorStop(1, `rgba(255, 255, 255, 0)`);
        ctx.fillStyle = radGrad;
        ctx.fillRect(0, 0, size, size);

        // Decorative circles -- seeded by card index for variety
        ctx.globalCompositeOperation = "overlay";
        for (let j = 0; j < 3 + cardIndex % 3; j++) {
            // Deterministic pseudo-random placement based on card + circle index
            const cx = ((cardIndex * 97 + j * 137) % size);
            const cy = ((cardIndex * 53 + j * 89) % size);
            const radius = size * (0.1 + (j * 0.07));
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, 0.15)`;
            ctx.fill();
        }

        // Subtle noise pattern via small random rectangles
        ctx.globalCompositeOperation = "overlay";
        const noiseCount = tier === 3 ? 200 : 80;
        for (let j = 0; j < noiseCount; j++) {
            // Use a simple LCG seeded per card for deterministic "randomness"
            const seed = cardIndex * 1000 + j;
            const px = ((seed * 16807) % 2147483647) % size;
            const py = ((seed * 48271) % 2147483647) % size;
            const alpha = ((seed * 69621) % 100) / 1000; // 0.00 to 0.099
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.fillRect(px, py, 2, 2);
        }

        ctx.globalCompositeOperation = "source-over";
    }

    if (tier === 3) {
        // Tier 3 (full resolution): add card number as a subtle watermark.
        // This provides a visual cue that full-res has loaded.
        ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
        ctx.font = `bold ${Math.round(size * 0.35)}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`${cardIndex + 1}`, size / 2, size / 2);

        // Add a thin border highlight to emphasize focused state
        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = 3;
        ctx.strokeRect(4, 4, size - 8, size - 8);
    }

    // Off-thread decode + resize to full layer dimensions.
    // Even for lower tiers (64x64, 256x256), we resize to TEX_SIZE so the
    // upload covers the entire texture layer via texSubImage3D. The lower-res
    // source image gets upscaled by createImageBitmap (bilinear), which is
    // correct: a Tier 1 thumbnail should fill the whole card, just blurry.
    return createImageBitmap(offscreen, {
        resizeWidth: TEX_SIZE,
        resizeHeight: TEX_SIZE,
        resizeQuality: "medium",
    });
}

// ---------------------------------------------------------------------------
// LOD target computation
// ---------------------------------------------------------------------------

/**
 * Determine the target tier for each card based on scroll proximity.
 *
 * The focused card (nearest to scroll position) gets Tier 3 (full-res).
 * Cards within +/-3 positions of focus get Tier 2 (medium).
 * All other cards stay at Tier 1 (thumbnail).
 *
 * This function also updates lastAccess timestamps for LRU tracking:
 * cards near focus get their timestamps refreshed, while distant cards
 * age out and become eviction candidates.
 *
 * @param {number} scrollPos  Current scroll position from Zig (card units)
 */
function updateTargetTiers(scrollPos) {
    const focusIndex = Math.round(Math.max(0, Math.min(MAX_CARDS - 1, scrollPos)));
    const now = performance.now();

    for (let i = 0; i < MAX_CARDS; i++) {
        const distance = Math.abs(i - focusIndex);
        const state = cardTexStates[i];

        if (distance === 0) {
            // Focused card: full resolution
            state.targetTier = 3;
            state.lastAccess = now;
        } else if (distance <= 3) {
            // Near cards: medium quality
            state.targetTier = 2;
            state.lastAccess = now;
        } else {
            // Far cards: thumbnail only
            state.targetTier = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// LRU eviction
// ---------------------------------------------------------------------------

/**
 * Enforce the memory budget by downgrading least-recently-accessed cards.
 *
 * Counts how many cards are at Tier 2 or above. If the count exceeds
 * LRU_BUDGET, the oldest (by lastAccess) cards are scheduled for
 * downgrade back to Tier 1.
 *
 * Downgrade means setting targetTier = 1, which the loading loop will
 * handle by generating and uploading a Tier 1 image (overwriting the
 * higher-res data in the texture array layer).
 */
function evictLRU() {
    // Collect indices of cards at Tier 2+
    const highRes = [];
    for (let i = 0; i < MAX_CARDS; i++) {
        if (cardTexStates[i].currentTier >= 2) {
            highRes.push(i);
        }
    }

    if (highRes.length <= LRU_BUDGET) return;

    // Sort by lastAccess ascending (oldest first = best eviction candidates)
    highRes.sort((a, b) => cardTexStates[a].lastAccess - cardTexStates[b].lastAccess);

    // Evict oldest until we're within budget
    const evictCount = highRes.length - LRU_BUDGET;
    for (let j = 0; j < evictCount; j++) {
        const idx = highRes[j];
        cardTexStates[idx].targetTier = 1;
    }
}

// ---------------------------------------------------------------------------
// Async loading scheduler
// ---------------------------------------------------------------------------

/**
 * Scan all cards and kick off one async image generation for the card
 * that needs the biggest quality upgrade. Only one load per frame to
 * avoid overwhelming the browser.
 *
 * Priority: cards that need the biggest tier jump get loaded first.
 * Among equal jumps, prefer lower card indices (stable ordering).
 */
function scheduleOneLoad() {
    let bestIndex = -1;
    let bestGap = 0;

    for (let i = 0; i < MAX_CARDS; i++) {
        const s = cardTexStates[i];
        if (s.loading) continue;

        if (s.targetTier > s.currentTier) {
            // Upgrade: load the next tier up (not skip tiers)
            const gap = s.targetTier - s.currentTier;
            if (gap > bestGap) {
                bestGap = gap;
                bestIndex = i;
            }
        } else if (s.targetTier < s.currentTier) {
            // Downgrade: also schedule (e.g., LRU eviction back to Tier 1)
            const gap = s.currentTier - s.targetTier;
            if (gap > bestGap) {
                bestGap = gap;
                bestIndex = i;
            }
        }
    }

    if (bestIndex === -1) return; // nothing to do

    const s = cardTexStates[bestIndex];
    s.loading = true;

    // Determine which tier to generate: step one tier toward target.
    // For upgrades we go up one at a time (1->2->3).
    // For downgrades we jump directly to the target tier.
    const nextTier = s.targetTier > s.currentTier
        ? s.currentTier + 1
        : s.targetTier;

    generateTierImage(bestIndex, nextTier).then((bitmap) => {
        uploadQueue.push({
            cardIndex: bestIndex,
            tier: nextTier,
            imageData: bitmap,
        });
        // Note: s.loading is cleared when the upload actually happens,
        // not when the image is generated. This prevents double-scheduling.
    }).catch((err) => {
        console.warn(`[tex] Failed to generate tier ${nextTier} for card ${bestIndex}:`, err);
        s.loading = false;
    });
}

/**
 * Process one pending upload from the queue. Called during idle time
 * within the frame budget.
 *
 * Uploads the generated image into the correct layer of the
 * TEXTURE_2D_ARRAY via texSubImage3D, then updates the card's state.
 *
 * @param {WebGLTexture} texArray  The texture array to upload into
 * @returns {boolean}  True if an upload was performed
 */
function processOneUpload(texArray) {
    if (uploadQueue.length === 0) return false;

    const { cardIndex, tier, imageData } = uploadQueue.shift();

    // All images are pre-resized to TEX_SIZE in generateTierImage(),
    // so we always upload the full layer dimensions.
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, texArray);
    gl.texSubImage3D(
        gl.TEXTURE_2D_ARRAY,
        0,                       // mip level
        0, 0, cardIndex,         // x, y, layer
        TEX_SIZE, TEX_SIZE, 1,   // width, height, depth
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        imageData
    );
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

    // Clean up the ImageBitmap to free memory immediately
    if (imageData.close) imageData.close();

    // Update card state
    const s = cardTexStates[cardIndex];
    s.currentTier = tier;
    s.loading = false;

    return true;
}

// ---------------------------------------------------------------------------
// Input: touch/mouse event marshaling into WASM ring buffer
// ---------------------------------------------------------------------------

/**
 * Wire up touch and mouse listeners that write InputEvent structs directly
 * into the WASM-side ring buffer. This is the JS "producer" half of the
 * lock-free SPSC queue; the Zig frame loop is the consumer.
 *
 * Event layout per slot (16 bytes, little-endian):
 *   +0  u8    event_type  (1=start, 2=move, 3=end)
 *   +1  [3]u8 padding
 *   +4  f32   x  (CSS pixels)
 *   +8  f32   y  (CSS pixels)
 *   +12 f32   timestamp  (ms, performance.now())
 *
 * @param {WebAssembly.Exports} exports
 * @param {WebAssembly.Memory} memory
 */
function initInput(exports, memory) {
    const ringPtr = exports.getInputRingPtr();
    const headPtr = exports.getInputHeadPtr();
    const tailPtr = exports.getInputTailPtr();

    const RING_SIZE = 64;
    const EVENT_SIZE = 16; // bytes per InputEvent

    /**
     * Push one event into the ring buffer. If the ring is full the event
     * is silently dropped -- this is acceptable because touch events arrive
     * at ~60-120 Hz and the consumer drains every frame.
     */
    function writeEvent(type, x, y) {
        const dv = new DataView(memory.buffer);
        const head = dv.getUint32(headPtr, true);
        const tail = dv.getUint32(tailPtr, true);

        // Full check: (head - tail) >>> 0 treats the subtraction as unsigned
        if (((head - tail) >>> 0) >= RING_SIZE) return; // ring full -- drop

        const slot = (head & (RING_SIZE - 1)) * EVENT_SIZE + ringPtr;
        dv.setUint8(slot, type);
        // bytes 1-3 are padding (leave as-is / zero)
        dv.setFloat32(slot + 4, x, true);
        dv.setFloat32(slot + 8, y, true);
        dv.setFloat32(slot + 12, performance.now(), true);

        // Advance head (wrapping add via >>> 0 to stay u32)
        dv.setUint32(headPtr, (head + 1) >>> 0, true);
    }

    // --- Touch events (mobile) ---
    canvas.addEventListener(
        "touchstart",
        (e) => {
            e.preventDefault();
            const t = e.touches[0];
            writeEvent(1, t.clientX, t.clientY);
        },
        { passive: false }
    );

    canvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            const t = e.touches[0];
            writeEvent(2, t.clientX, t.clientY);
        },
        { passive: false }
    );

    canvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            const t = e.changedTouches[0];
            writeEvent(3, t.clientX, t.clientY);
        },
        { passive: false }
    );

    // --- Mouse events (desktop) ---
    let mouseDown = false;

    canvas.addEventListener("mousedown", (e) => {
        mouseDown = true;
        writeEvent(1, e.clientX, e.clientY);
    });

    canvas.addEventListener("mousemove", (e) => {
        if (mouseDown) {
            writeEvent(2, e.clientX, e.clientY);
        }
    });

    canvas.addEventListener("mouseup", (e) => {
        if (mouseDown) {
            writeEvent(3, e.clientX, e.clientY);
            mouseDown = false;
        }
    });

    canvas.addEventListener("mouseleave", (e) => {
        if (mouseDown) {
            writeEvent(3, e.clientX, e.clientY);
            mouseDown = false;
        }
    });
}

// ---------------------------------------------------------------------------
// Geometry: card quad with positions + texcoords
// ---------------------------------------------------------------------------

/**
 * Creates a VAO with a 3:2 aspect card quad. Each vertex has:
 *   - vec2 a_position (location 0)
 *   - vec2 a_texcoord  (location 1)
 *
 * The quad spans from (-0.75, -0.5) to (0.75, 0.5) in the XY plane,
 * with texcoords from (0,0) to (1,1).
 *
 * Instance attributes (locations 2-7) are set up separately.
 *
 * @returns {{ vao: WebGLVertexArrayObject, vertexCount: number }}
 */
function createCardQuad() {
    // Interleaved: [x, y, u, v] per vertex
    // prettier-ignore
    const vertices = new Float32Array([
        // Triangle 1            // Texcoords
        -0.75, -0.5,   0.0, 0.0,
         0.75, -0.5,   1.0, 0.0,
         0.75,  0.5,   1.0, 1.0,
        // Triangle 2
        -0.75, -0.5,   0.0, 0.0,
         0.75,  0.5,   1.0, 1.0,
        -0.75,  0.5,   0.0, 1.0,
    ]);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const stride = 4 * 4; // 4 floats * 4 bytes each = 16 bytes

    // a_position at location 0 (vec2)
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);

    // a_texcoord at location 1 (vec2)
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 2 * 4);

    // --- Instance attribute buffers ---

    // Instance matrix buffer (locations 2-5): 4 vec4 columns per instance
    // Total: 16 floats = 64 bytes per instance
    const instanceMatrixBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceMatrixBuf);
    gl.bufferData(gl.ARRAY_BUFFER, MAX_CARDS * 16 * 4, gl.DYNAMIC_DRAW);

    const matStride = 16 * 4; // 64 bytes per mat4
    for (let col = 0; col < 4; col++) {
        const loc = 2 + col;
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, matStride, col * 16);
        gl.vertexAttribDivisor(loc, 1); // advance once per instance
    }

    // Instance opacity buffer (location 6): 1 float per instance
    const instanceOpacityBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceOpacityBuf);
    gl.bufferData(gl.ARRAY_BUFFER, MAX_CARDS * 4, gl.DYNAMIC_DRAW);

    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 1, gl.FLOAT, false, 4, 0);
    gl.vertexAttribDivisor(6, 1);

    // Instance tex layer buffer (location 7): 1 float per instance
    const instanceTexLayerBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceTexLayerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, MAX_CARDS * 4, gl.DYNAMIC_DRAW);

    gl.enableVertexAttribArray(7);
    gl.vertexAttribPointer(7, 1, gl.FLOAT, false, 4, 0);
    gl.vertexAttribDivisor(7, 1);

    gl.bindVertexArray(null);

    return {
        vao,
        vertexCount: 6,
        instanceMatrixBuf,
        instanceOpacityBuf,
        instanceTexLayerBuf,
    };
}

// ---------------------------------------------------------------------------
// Main initialisation
// ---------------------------------------------------------------------------

async function main() {
    // 1. Load WASM
    const resp = await fetch("carousel.wasm");
    const wasmBytes = await resp.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(wasmBytes, importObject);
    wasm = instance;
    wasmMemory = wasm.exports.memory;

    // 2. Compile shaders and link program
    const [vertShader, fragShader] = await Promise.all([
        loadShader("shaders/card.vert", gl.VERTEX_SHADER),
        loadShader("shaders/card.frag", gl.FRAGMENT_SHADER),
    ]);
    const program = linkProgram(vertShader, fragShader);

    // 3. Get uniform locations (no more u_model -- it's per-instance now)
    const uView = gl.getUniformLocation(program, "u_view");
    const uProjection = gl.getUniformLocation(program, "u_projection");
    const uTextures = gl.getUniformLocation(program, "u_textures");

    // 4. Create card geometry + instance buffers
    const card = createCardQuad();

    // 5. Create progressive texture array (starts at Tier 0 -- solid colors)
    const texArray = createTextureArray();

    // 6. Get WASM buffer pointers (stable after init -- WASM memory might
    //    grow but these are offsets into the data segment, not heap)
    // We'll read pointers after init since init populates the buffers.

    // 7. Handle resize with devicePixelRatio
    function handleResize() {
        const dpr = window.devicePixelRatio || 1;
        const displayW = Math.round(canvas.clientWidth * dpr);
        const displayH = Math.round(canvas.clientHeight * dpr);
        if (canvas.width !== displayW || canvas.height !== displayH) {
            canvas.width = displayW;
            canvas.height = displayH;
            gl.viewport(0, 0, displayW, displayH);
            wasm.exports.resize(displayW, displayH);
        }
    }

    window.addEventListener("resize", handleResize);
    handleResize();

    // 8. Init the Zig side (populates all buffers)
    wasm.exports.init();

    // 9. Wire up touch/mouse input into the WASM ring buffer
    initInput(wasm.exports, wasmMemory);

    // Cache buffer pointers from WASM exports
    const transformPtr = wasm.exports.getTransformBufferPtr();
    const opacityPtr = wasm.exports.getOpacityBufferPtr();
    const texIndexPtr = wasm.exports.getTexIndexBufferPtr();
    const viewPtr = wasm.exports.getViewMatrixPtr();
    const projPtr = wasm.exports.getProjMatrixPtr();

    // 10. Render loop with progressive texture loading
    let lastTime = 0;

    function render(now) {
        const frameStart = performance.now();

        // Convert ms to seconds; cap dt to avoid spiral of death
        const dt = lastTime === 0 ? 0.016 : Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        // Let Zig update transforms (drains input, runs physics, recomputes layout)
        wasm.exports.frame(dt);

        // Read the card count (number of visible instances)
        const cardCount = wasm.exports.getCardCount();

        // --- Progressive Loading: LOD decisions ---
        // Read scroll position from Zig to determine which cards need
        // higher-quality textures (focused card -> Tier 3, near -> Tier 2).
        const scrollPos = wasm.exports.getScrollPosition();
        updateTargetTiers(scrollPos);
        evictLRU();
        scheduleOneLoad();

        // Clear
        gl.clearColor(0.04, 0.04, 0.06, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);

        // Enable alpha blending for opacity fading
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.useProgram(program);

        // Upload view and projection matrices from WASM memory
        const viewMat = new Float32Array(wasmMemory.buffer, viewPtr, 16);
        const projMat = new Float32Array(wasmMemory.buffer, projPtr, 16);
        gl.uniformMatrix4fv(uView, false, viewMat);
        gl.uniformMatrix4fv(uProjection, false, projMat);

        // Bind the texture array to unit 0
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, texArray);
        gl.uniform1i(uTextures, 0);

        // Upload instance data from WASM memory into GPU buffers
        const transforms = new Float32Array(wasmMemory.buffer, transformPtr, cardCount * 16);
        const opacities = new Float32Array(wasmMemory.buffer, opacityPtr, cardCount);
        const texIndices = new Float32Array(wasmMemory.buffer, texIndexPtr, cardCount);

        gl.bindBuffer(gl.ARRAY_BUFFER, card.instanceMatrixBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, transforms);

        gl.bindBuffer(gl.ARRAY_BUFFER, card.instanceOpacityBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, opacities);

        gl.bindBuffer(gl.ARRAY_BUFFER, card.instanceTexLayerBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, texIndices);

        // Draw all visible card instances in one call
        gl.bindVertexArray(card.vao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, card.vertexCount, cardCount);
        gl.bindVertexArray(null);

        // --- Idle-Frame Texture Upload ---
        // After Zig frame + GL draw, check if there's budget remaining.
        // If the frame took less than FRAME_BUDGET_MS (12ms), upload one
        // pending texture. This prevents texture uploads from causing jank
        // by only doing GPU work when we have headroom.
        const elapsed = performance.now() - frameStart;
        if (elapsed < FRAME_BUDGET_MS) {
            processOneUpload(texArray);
        }

        requestAnimationFrame(render);
    }

    requestAnimationFrame(render);
}

main().catch((err) => {
    console.error("Carousel init failed:", err);
    document.body.textContent = `Error: ${err.message}`;
});
