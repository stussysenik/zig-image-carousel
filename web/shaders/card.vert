#version 300 es
// Card vertex shader -- Gate 0
//
// Transforms each vertex by model, view, and projection matrices.
// All three matrices are computed in Zig and passed as uniforms from JS.

precision highp float;

// Vertex attributes
layout(location = 0) in vec3 a_position;

// Transform uniforms (column-major 4x4, matching Zig's layout)
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
