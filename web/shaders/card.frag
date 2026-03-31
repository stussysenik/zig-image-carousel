#version 300 es
// Card fragment shader -- Gate 0
//
// Outputs a solid color. In later gates this will sample a texture;
// for now the color is passed as a uniform so we can verify the
// pipeline works end-to-end.

precision highp float;

uniform vec4 u_color;

out vec4 fragColor;

void main() {
    fragColor = u_color;
}
