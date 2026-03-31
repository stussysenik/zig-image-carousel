const std = @import("std");

/// Build configuration for the Zig Image Carousel.
///
/// Produces two artifacts:
///   1. `carousel.wasm` -- wasm32-freestanding binary with SIMD128 for browser use
///   2. Native test binary -- runs unit tests on the host machine
///
/// The WASM binary is also copied into `web/` so a simple HTTP server
/// (e.g. `python3 -m http.server -d web`) can serve the full application.
pub fn build(b: *std.Build) void {
    // ---------------------------------------------------------------
    // WASM target: wasm32-freestanding + SIMD128
    // ---------------------------------------------------------------
    var wasm_query: std.Target.Query = .{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    };
    wasm_query.cpu_features_add = std.Target.wasm.featureSet(&.{.simd128});
    const wasm_target = b.resolveTargetQuery(wasm_query);

    const wasm_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
        .strip = true,
    });

    const wasm = b.addExecutable(.{
        .name = "carousel",
        .root_module = wasm_module,
    });

    // WASM-specific linker settings:
    //   - No entry point (we export functions instead)
    //   - Export all public Zig symbols marked `export`
    //   - Let the host read/write linear memory directly
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    wasm.export_memory = true;
    wasm.initial_memory = 32 * 65536; // 2 MB (linker needs ~1 MB for runtime)
    wasm.max_memory = 64 * 65536; // 4 MB

    b.installArtifact(wasm);

    // Also copy the .wasm into web/ for convenient dev serving
    const install_to_web = b.addInstallFile(
        wasm.getEmittedBin(),
        "../web/carousel.wasm",
    );
    b.getInstallStep().dependOn(&install_to_web.step);

    // ---------------------------------------------------------------
    // Native unit tests
    // ---------------------------------------------------------------
    const native_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = native_target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = test_module,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
