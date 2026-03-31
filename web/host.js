/**
 * host.js -- WebGL2 Instanced Rendering Driver for the Zig Image Carousel
 *
 * This is intentionally a *thin* GPU driver. All transform computation
 * happens in Zig; JS only:
 *   1. Loads the WASM module and wires up imports
 *   2. Creates the WebGL2 context, compiles shaders, sets up instancing
 *   3. Generates placeholder textures (numbered colored cards) as a 2D array
 *   4. Each frame: reads matrices/opacities/texIndices from WASM memory,
 *      uploads into instance buffers, issues a single drawArraysInstanced call
 *
 * Architecture: "Maximum Zig" -- JS never computes a matrix.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_CARDS = 12;
const TEX_SIZE = 256; // Placeholder texture resolution (per layer)

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
// Placeholder texture generation
// ---------------------------------------------------------------------------

/**
 * Generate a 2D array texture with MAX_CARDS layers. Each layer is a
 * TEX_SIZE x TEX_SIZE colored rectangle with a centered card number.
 * Uses an offscreen canvas to render text, then uploads via texSubImage3D.
 *
 * @returns {WebGLTexture}
 */
function createPlaceholderTextures() {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);

    // Allocate storage for all layers at once
    gl.texImage3D(
        gl.TEXTURE_2D_ARRAY,
        0,                   // mip level
        gl.RGBA8,            // internal format
        TEX_SIZE, TEX_SIZE,  // width, height
        MAX_CARDS,           // depth (number of layers)
        0,                   // border
        gl.RGBA,             // format
        gl.UNSIGNED_BYTE,    // type
        null                 // no data yet
    );

    // Create an offscreen canvas for rendering each card's placeholder
    const offscreen = document.createElement("canvas");
    offscreen.width = TEX_SIZE;
    offscreen.height = TEX_SIZE;
    const ctx = offscreen.getContext("2d");

    for (let i = 0; i < MAX_CARDS; i++) {
        const [r, g, b] = CARD_COLORS[i];

        // Fill background with card color
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(0, 0, TEX_SIZE, TEX_SIZE);

        // Draw card number centered
        ctx.fillStyle = "white";
        ctx.font = "bold 96px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`${i + 1}`, TEX_SIZE / 2, TEX_SIZE / 2);

        // Upload this layer
        gl.texSubImage3D(
            gl.TEXTURE_2D_ARRAY,
            0,                // mip level
            0, 0, i,          // x, y, layer offset
            TEX_SIZE, TEX_SIZE, 1, // width, height, depth
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            offscreen          // TexImageSource -- the canvas element
        );
    }

    // Texture parameters
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

    return tex;
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

    // 5. Generate placeholder texture array
    const texArray = createPlaceholderTextures();

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

    // 10. Render loop
    let lastTime = 0;

    function render(now) {
        // Convert ms to seconds; cap dt to avoid spiral of death
        const dt = lastTime === 0 ? 0.016 : Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        // Let Zig update transforms
        wasm.exports.frame(dt);

        // Read the card count (number of visible instances)
        const cardCount = wasm.exports.getCardCount();

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

        requestAnimationFrame(render);
    }

    requestAnimationFrame(render);
}

main().catch((err) => {
    console.error("Carousel init failed:", err);
    document.body.textContent = `Error: ${err.message}`;
});
