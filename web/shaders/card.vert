#version 300 es
// Card vertex shader -- Gate 1 (Instanced Rendering)
//
// Per-vertex attributes: position and texcoord (shared across all instances).
// Per-instance attributes: model matrix (4 vec4 columns), opacity, tex layer.
// The model matrix is split across 4 attribute locations because GLSL ES 3.0
// does not support mat4 vertex attributes directly with divisor.

precision highp float;

// Per-vertex (from the card quad VBO)
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;

// Per-instance (from instance buffers, advanced once per card)
layout(location = 2) in vec4 a_model_col0;
layout(location = 3) in vec4 a_model_col1;
layout(location = 4) in vec4 a_model_col2;
layout(location = 5) in vec4 a_model_col3;
layout(location = 6) in float a_opacity;
layout(location = 7) in float a_tex_layer;

// Camera transforms (shared across all instances)
uniform mat4 u_view;
uniform mat4 u_projection;

// Interpolated outputs to fragment shader
out vec2 v_texcoord;
out float v_opacity;
out float v_tex_layer;
out float v_depth;

void main() {
    // Reconstruct the model matrix from its four column vectors
    mat4 model = mat4(a_model_col0, a_model_col1, a_model_col2, a_model_col3);

    // Transform to view space first (for depth calculation)
    vec4 viewPos = u_view * model * vec4(a_position, 0.0, 1.0);

    // Final clip-space position
    gl_Position = u_projection * viewPos;

    // Pass through to fragment shader
    v_texcoord = a_texcoord;
    v_opacity = a_opacity;
    v_tex_layer = a_tex_layer;
    v_depth = -viewPos.z;
}
