/**
 * host.js -- WebGL2 driver for the Zig Image Carousel
 *
 * This is intentionally a *thin* GPU driver. All transform computation
 * happens in Zig; JS only:
 *   1. Loads the WASM module and wires up imports
 *   2. Creates the WebGL2 context and compiles shaders
 *   3. Each frame: reads matrices from WASM memory, sets uniforms, draws
 *
 * Architecture: "Maximum Zig" -- JS never computes a matrix.
 */

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
// Geometry: 3:2 aspect ratio card quad
// ---------------------------------------------------------------------------

/**
 * Creates a VAO with a 3:2 aspect card (vertices from -0.75,-0.5 to 0.75,0.5).
 * Two triangles forming a rectangle, centered at origin in the XY plane.
 * @returns {{ vao: WebGLVertexArrayObject, vertexCount: number }}
 */
function createCardQuad() {
    // prettier-ignore
    const vertices = new Float32Array([
        // Triangle 1
        -0.75, -0.5, 0.0,
         0.75, -0.5, 0.0,
         0.75,  0.5, 0.0,
        // Triangle 2
        -0.75, -0.5, 0.0,
         0.75,  0.5, 0.0,
        -0.75,  0.5, 0.0,
    ]);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    // a_position at location 0
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);

    return { vao, vertexCount: 6 };
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

    // 3. Get uniform locations
    const uModel = gl.getUniformLocation(program, "u_model");
    const uView = gl.getUniformLocation(program, "u_view");
    const uProjection = gl.getUniformLocation(program, "u_projection");
    const uColor = gl.getUniformLocation(program, "u_color");

    // 4. Create card geometry
    const card = createCardQuad();

    // 5. Handle resize with devicePixelRatio
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

    // 6. Init the Zig side
    wasm.exports.init();

    // 7. Render loop
    let lastTime = 0;

    function render(now) {
        // Convert ms to seconds; cap dt to avoid spiral of death
        const dt = lastTime === 0 ? 0.016 : Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        // Let Zig update transforms
        wasm.exports.frame(dt);

        // Clear
        gl.clearColor(0.04, 0.04, 0.06, 1.0); // near-black, matches CSS
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);

        gl.useProgram(program);

        // Read view and projection matrices from WASM memory
        const viewPtr = wasm.exports.getViewMatrixPtr();
        const projPtr = wasm.exports.getProjMatrixPtr();
        const viewMat = new Float32Array(wasmMemory.buffer, viewPtr, 16);
        const projMat = new Float32Array(wasmMemory.buffer, projPtr, 16);

        gl.uniformMatrix4fv(uView, false, viewMat);
        gl.uniformMatrix4fv(uProjection, false, projMat);

        // Draw each card
        const cardCount = wasm.exports.getCardCount();
        const transformPtr = wasm.exports.getTransformBufferPtr();

        gl.bindVertexArray(card.vao);

        for (let i = 0; i < cardCount; i++) {
            // Each Mat4 is 16 floats = 64 bytes
            const modelMat = new Float32Array(wasmMemory.buffer, transformPtr + i * 64, 16);
            gl.uniformMatrix4fv(uModel, false, modelMat);

            // Gate 0: solid teal-ish color
            gl.uniform4f(uColor, 0.26, 0.72, 0.82, 1.0);

            gl.drawArrays(gl.TRIANGLES, 0, card.vertexCount);
        }

        gl.bindVertexArray(null);

        requestAnimationFrame(render);
    }

    requestAnimationFrame(render);
}

main().catch((err) => {
    console.error("Carousel init failed:", err);
    document.body.textContent = `Error: ${err.message}`;
});
