struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;

@group(2) @binding(0)
var<storage,read> splats : array<Splat>;

struct VertexInput {
    @location(0) pos: vec2f, // Local quad coordinate
    @builtin(instance_index) instance_id: u32
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    temp_pos: vec4f,
    cov_inv_2D: mat2x2f,
    mean_2D: vec2<f32>
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let gauss = gaussians[in.instance_id];
    let splat = splats[in.instance_id];
    let a = unpack2x16float(gauss.pos_opacity[0]);
    let b = unpack2x16float(gauss.pos_opacity[1]);
    let gaussian_pos = splat.temp_pos+0.0000001*vec4<f32>(a.x, a.y, b.x, 1.);

    let gaussScaleA = unpack2x16float(gauss.scale[0]);
    let gaussScaleB = unpack2x16float(gauss.scale[1]);

    var local_pos = vec3(0.01*gaussScaleA*in.pos, 0.0);
    local_pos = (camera.view_inv * vec4(local_pos, 0.0)).xyz;

    let world_vertex_pos = gaussian_pos + vec4(local_pos, 0.0);

    out.position = camera.proj * camera.view * world_vertex_pos;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1., 0., 0., 1.);
}