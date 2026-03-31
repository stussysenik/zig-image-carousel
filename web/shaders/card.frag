#version 300 es
// Card fragment shader -- Gate 1 (Texture Array + Opacity)
//
// Samples from a 2D texture array using the per-instance texture layer
// index, and modulates alpha by the per-instance opacity. This enables
// smooth depth-based fading in the stack.

precision highp float;

in vec2 v_texcoord;
in float v_opacity;
in float v_tex_layer;
in float v_depth;

// 2D array texture: each layer is one card's image (or placeholder)
uniform highp sampler2DArray u_textures;

out vec4 fragColor;

void main() {
    // Sample the texture layer for this card instance
    vec4 color = texture(u_textures, vec3(v_texcoord, v_tex_layer));

    // Apply per-instance opacity fade
    fragColor = vec4(color.rgb, color.a * v_opacity);
}
