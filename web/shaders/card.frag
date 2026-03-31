#version 300 es
// Card fragment shader -- Gate 4 (Visual Polish)
//
// Depth-based blur, rounded corners (SDF discard), focused-card edge glow,
// and soft shadow at card edges. All effects are additive enhancements
// that degrade gracefully (the card still renders if any effect is removed).

precision highp float;

in vec2 v_texcoord;
in float v_opacity;
in float v_tex_layer;
in float v_depth;

// 2D array texture: each layer is one card's image (or placeholder)
uniform highp sampler2DArray u_textures;

out vec4 fragColor;

void main() {
    // --- Rounded corners via SDF discard ---
    // Map texcoords to a -1..1 space centered on the card, then check
    // distance to a rounded rectangle. Pixels outside the radius are
    // discarded, giving smooth curved corners without geometry changes.
    vec2 p = abs(v_texcoord - 0.5) * 2.0;
    float corner_radius = 0.08;
    vec2 q = p - vec2(1.0 - corner_radius);
    float d = length(max(q, 0.0)) - corner_radius;
    if (d > 0.0) discard;

    // --- Depth-based blur ---
    // Cards farther from the camera get a subtle blur by averaging
    // multiple texture samples at small offsets. This simulates a
    // depth-of-field effect, drawing the eye to the front card.
    float blur_amount = smoothstep(2.0, 8.0, v_depth) * 0.003;
    vec4 color = vec4(0.0);
    color += texture(u_textures, vec3(v_texcoord + vec2(-blur_amount, 0.0), v_tex_layer));
    color += texture(u_textures, vec3(v_texcoord + vec2( blur_amount, 0.0), v_tex_layer));
    color += texture(u_textures, vec3(v_texcoord + vec2(0.0, -blur_amount), v_tex_layer));
    color += texture(u_textures, vec3(v_texcoord + vec2(0.0,  blur_amount), v_tex_layer));
    color += texture(u_textures, vec3(v_texcoord, v_tex_layer));
    color /= 5.0;

    // --- Soft shadow on bottom/right edge ---
    // Darkens pixels near the bottom-right corner, creating a subtle
    // drop-shadow illusion that reinforces the 3D card stacking.
    float shadow_x = smoothstep(0.0, 0.05, 1.0 - v_texcoord.x);
    float shadow_y = smoothstep(0.0, 0.05, 1.0 - v_texcoord.y);
    color.rgb *= mix(0.7, 1.0, shadow_x * shadow_y);

    // --- Focused card edge glow ---
    // Cards close to the camera (low v_depth) get a blue-white glow
    // along their edges, providing a "selected" highlight that draws
    // attention to the front card without an explicit selection UI.
    float edge_dist = min(
        min(v_texcoord.x, 1.0 - v_texcoord.x),
        min(v_texcoord.y, 1.0 - v_texcoord.y)
    );
    float glow = smoothstep(0.0, 0.03, edge_dist)
               * (1.0 - smoothstep(0.03, 0.08, edge_dist));
    float focus_factor = smoothstep(3.0, 1.5, v_depth);
    color.rgb += vec3(0.3, 0.5, 1.0) * glow * focus_factor * 0.4;

    // Apply per-instance opacity fade
    fragColor = vec4(color.rgb, color.a * v_opacity);
}
