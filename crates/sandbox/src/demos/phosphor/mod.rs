// TODO: Render tonemap pass into its own float buffer
//
// TODO: Apply sobel edge detection filter to smooth edges of tonemap buffer
//
// TODO: Render mipmaps for tonemap buffer
//
// TODO: Render HDR bloom using mipmaps
//
// TODO: Switch to EXR + float format for gradient texture
//
// TODO: 3D Rendering
//       * Need to figure out how to make a phosphor decay model work with a rotateable camera
//       * Want to avoid UE4-style whole screen smearing
//
//       [ ] Convert line rendering to true 3D
//           [ ] Use hemisphere meshes as end caps
//           [ ] Calculate 3D look-at rotation in vertex shader
//
//       * Depth buffer
//         [ ] Sample depth along with color in decay shader
//             * Decay toward far plane
//             * Use offset to account for movement
//               * Example case - mech with glowing visor running at camera
//               * Glow should persist, but will pass inside mech body due to motion
//               * Offset will allow motion to be counteracted, retaining the desired effect
//             * Zero out when intensity reaches 0?
//
//       * Cubemap-style setup seems like the best approach currently
//         * Is a geometry-based solution to this viable?
//         * Geometry layer using polar coordinates
//         * Tesselate lines to counteract linear transform artifacts
//         * Current phosphor rendering relies on rasterization
//           * Could supersample, or render cubemap texels via shaped point sprites as a design workaround
//         * Alternately, devise an equivalent-looking effect using geometry animation
//           * Viable - current effect could be achieved with fine enough geo
//           * Can take advantage of MSAA to avoid framebuffer size issues
//
//       * MechWarrior 2 gradient skybox background
//         * Setting for underlay / overlay behavior
//         * Overlay acts like a vectrex color overlay
//         * Underlay respects depth and doesn't draw behind solid objects

mod assemblage;
mod components;
mod systems;

pub use assemblage::*;
pub use components::*;
pub use systems::*;

use std::time::{Duration, Instant};

use antigen_winit::{
    winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoopWindowTarget},
    },
    AssembleWinit, EventLoopHandler, RedrawUnconditionally, WindowComponent,
};

use antigen_core::{parallel, serial, single, AddIndirectComponent, Construct, ImmutableWorld};

use antigen_wgpu::{
    buffer_size_of,
    wgpu::{
        AddressMode, BufferAddress, BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode,
        Maintain, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, TextureAspect,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    },
    AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent,
};

const HDR_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const MAX_MESH_VERTICES: usize = 100;
const MAX_MESH_INDICES: usize = 100;
const MAX_LINES: usize = 10000;
const BASE_FLASH_LINE: usize = MAX_LINES / 2;

pub fn orthographic_matrix(aspect: f32, zoom: f32) -> [[f32; 4]; 4] {
    println!("Zoomed aspect: {}", zoom * aspect);
    let projection =
        nalgebra_glm::ortho_lh_zo(-zoom * aspect, zoom * aspect, -zoom, zoom, 0.0, 1.0);
    projection.into()
}

pub fn perspective_matrix(aspect: f32, (ofs_x, ofs_y): (f32, f32)) -> [[f32; 4]; 4] {
    let x = ofs_x * std::f32::consts::PI;
    let view = nalgebra_glm::look_at_lh(
        &nalgebra::vector![x.sin() * 300.0, ofs_y * 150.0, -x.cos() * 300.0],
        &nalgebra::vector![0.0, 0.0, 0.0],
        &nalgebra::Vector3::y_axis(),
    );
    let projection = nalgebra_glm::perspective_lh_zo(aspect, (45.0f32).to_radians(), 1.0, 500.0);

    let matrix = projection * view;

    matrix.into()
}

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let time_entity = cmd.push(());
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble time entity
    cmd.add_component(time_entity, StartTimeComponent::construct(Instant::now()));
    cmd.add_component(time_entity, TotalTimeComponent::construct(0.0));
    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        time_entity,
        TotalTimeComponent::construct(0.0),
        buffer_size_of::<[[f32; 4]; 4]>() * 2,
        Some(renderer_entity),
    );
    cmd.add_component(time_entity, TimestampComponent::construct(Instant::now()));
    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        time_entity,
        DeltaTimeComponent::construct(1.0 / 60.0),
        (buffer_size_of::<[[f32; 4]; 4]>() * 2) + buffer_size_of::<f32>(),
        Some(renderer_entity),
    );

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Phosphor");

    // Redraw window unconditionally
    cmd.add_component(window_entity, RedrawUnconditionally);

    // Renderer
    cmd.add_component(renderer_entity, PhosphorRenderer);
    cmd.assemble_wgpu_render_pipeline_with_usage::<PhosphorDecay>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<BeamLine>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<BeamMesh>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<Uniform>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<PhosphorFrontBuffer>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<PhosphorBackBuffer>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<Tonemap>(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);
    cmd.add_component(renderer_entity, BufferFlipFlopComponent::construct(false));
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Window reference for input handling
    cmd.add_indirect_component::<WindowComponent>(renderer_entity, window_entity);

    // Beam buffer
    cmd.assemble_wgpu_texture_with_usage::<BeamBuffer>(
        renderer_entity,
        TextureDescriptor {
            label: Some("Beam Buffer"),
            size: Extent3d {
                width: 640,
                height: 480,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: HDR_TEXTURE_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<BeamBuffer>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label: Some("Beam Buffer View"),
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    // Beam depth buffer
    cmd.assemble_wgpu_texture_with_usage::<BeamDepthBuffer>(
        renderer_entity,
        TextureDescriptor {
            label: Some("Beam Depth Buffer"),
            size: Extent3d {
                width: 640,
                height: 480,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<BeamDepthBuffer>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label: Some("Beam Depth Buffer View"),
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    // Phosphor front buffer
    cmd.assemble_wgpu_texture_with_usage::<PhosphorFrontBuffer>(
        renderer_entity,
        TextureDescriptor {
            label: Some("Phosphor Front Buffer"),
            size: Extent3d {
                width: 640,
                height: 480,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: HDR_TEXTURE_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<PhosphorFrontBuffer>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label: Some("Phosphor Front Buffer View"),
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    // Phosphor back buffer
    cmd.assemble_wgpu_texture_with_usage::<PhosphorBackBuffer>(
        renderer_entity,
        TextureDescriptor {
            label: Some("Phosphor Back Buffer"),
            size: Extent3d {
                width: 640,
                height: 480,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: HDR_TEXTURE_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<PhosphorBackBuffer>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label: Some("Phosphor Back Buffer View"),
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    // Gradients texture
    assemble_png_texture(
        cmd,
        renderer_entity,
        Some("Gradients Texture"),
        include_bytes!("textures/gradients.png"),
    );

    // Phosphor sampler
    cmd.assemble_wgpu_sampler_with_usage::<Linear>(
        renderer_entity,
        SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        },
    );

    // Shaders
    cmd.assemble_wgpu_shader_with_usage::<PhosphorDecay>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/hdr_decay.wgsl"
            ))),
        },
    );

    cmd.assemble_wgpu_shader_with_usage::<BeamLine>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/hdr_line.wgsl"
            ))),
        },
    );
    cmd.assemble_wgpu_shader_with_usage::<BeamMesh>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/hdr_mesh.wgsl"
            ))),
        },
    );

    cmd.assemble_wgpu_shader_with_usage::<Tonemap>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/tonemap.wgsl"
            ))),
        },
    );

    // Uniform buffer
    cmd.assemble_wgpu_buffer_with_usage::<Uniform>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: buffer_size_of::<UniformData>(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        renderer_entity,
        PerspectiveMatrixComponent::construct(perspective_matrix(640.0 / 480.0, (0.0, 0.0))),
        0,
        Some(renderer_entity),
    );

    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        renderer_entity,
        OrthographicMatrixComponent::construct(orthographic_matrix(640.0 / 480.0, 200.0)),
        buffer_size_of::<[[f32; 4]; 4]>(),
        Some(renderer_entity),
    );

    fn circle_strip(subdiv: usize) -> Vec<LineVertexData> {
        let subdiv = subdiv as isize;
        let half = 1 + subdiv;

        // Generate left quarter-circle
        let mut left = (-half..1)
            .map(|i| i as f32 / half as f32)
            .map(|f| {
                let f = f * (std::f32::consts::PI * 0.5);
                (f.sin(), f.cos(), 0.0)
            })
            .collect::<Vec<_>>();

        let mut right = (0..half + 1)
            .map(|i| i as f32 / half as f32)
            .map(|f| {
                let f = f * (std::f32::consts::PI * 0.5);
                (f.sin(), f.cos(), 1.0)
            })
            .collect::<Vec<_>>();

        let first = left.remove(0);
        let last = right.pop().unwrap();

        let inter = left
            .into_iter()
            .chain(right.into_iter())
            .flat_map(|(x, y, s)| [(x, -y, s), (x, y, s)]);

        std::iter::once(first)
            .chain(inter)
            .chain(std::iter::once(last))
            .map(|(x, y, s)| LineVertexData {
                position: [x, y, 0.0, 1.0],
                end: s,
                ..Default::default()
            })
            .collect()
    }

    // Line Vertex buffer
    let vertices = circle_strip(2);

    cmd.assemble_wgpu_buffer_with_usage::<LineVertex>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Line Vertex Buffer"),
            size: buffer_size_of::<LineVertexData>() * vertices.len() as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_data_with_usage::<LineVertex, _>(
        renderer_entity,
        LineVertexDataComponent::construct(vertices),
        0,
        None,
    );

    // Line Instance buffer
    cmd.assemble_wgpu_buffer_with_usage::<LineInstance>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Line Instance Buffer"),
            size: buffer_size_of::<LineInstanceData>() * MAX_LINES as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Oscilloscopes
    assemble_oscilloscope(
        cmd,
        renderer_entity,
        0,
        (0.0, 0.0, 0.0),
        Oscilloscope::new(6.66, 30.0, |_| (0.0, 0.0, 0.0)),
        7.0,
        -6.0,
        -0.0,
        0.0,
    );

    assemble_oscilloscope(
        cmd,
        renderer_entity,
        1,
        (-80.0, 40.0, -25.0),
        Oscilloscope::new(3.33, 30.0, |f| (f.sin(), f.cos(), f.sin())),
        7.0,
        -0.0,
        -240.0,
        0.0,
    );

    assemble_oscilloscope(
        cmd,
        renderer_entity,
        2,
        (0.0, 40.0, 0.0),
        Oscilloscope::new(2.22, 30.0, |f| (f.sin(), (f * 1.2).sin(), (f * 1.4).cos())),
        7.0,
        -7.0,
        0.0,
        1.0,
    );

    assemble_oscilloscope(
        cmd,
        renderer_entity,
        3,
        (80.0, 40.0, 25.0),
        Oscilloscope::new(3.33, 30.0, |f| (f.cos(), (f * 1.2).cos(), (f * 1.4).cos())),
        7.0,
        -12.0,
        12.0,
        2.0,
    );

    // Flash timer
    cmd.add_component(
        renderer_entity,
        TimerComponent::construct(Timer {
            timestamp: Instant::now(),
            duration: Duration::from_secs_f32(2.0),
        }),
    );

    // Gradient 3 Triangle
    assemble_line_strip(
        cmd,
        renderer_entity,
        BASE_FLASH_LINE,
        vec![
            MeshVertexData::new((-50.0, -20.0, 0.0), 11.0, -10.0, 0.0, 3.0),
            MeshVertexData::new((-90.0, -80.0, 0.0), 10.0, -10.0, 0.0, 3.0),
            MeshVertexData::new((-10.0, -80.0, 0.0), 9.0, -10.0, 0.0, 3.0),
            MeshVertexData::new((-50.0, -20.0, 0.0), 8.0, -10.0, 0.0, 3.0),
        ],
    );

    // Gradients 0-2 Triangle
    assemble_line_list(
        cmd,
        renderer_entity,
        BASE_FLASH_LINE + 3,
        vec![
            MeshVertexData::new((50.0, -80.0, 0.0), 11.0, -10.0, 0.0, 2.0),
            MeshVertexData::new((90.0, -20.0, 0.0), 10.0, -10.0, 0.0, 2.0),
            MeshVertexData::new((90.0, -20.0, 0.0), 10.0, -10.0, 0.0, 1.0),
            MeshVertexData::new((10.0, -20.0, 0.0), 9.0, -10.0, 0.0, 1.0),
            MeshVertexData::new((10.0, -20.0, 0.0), 9.0, -10.0, 0.0, 0.0),
            MeshVertexData::new((50.0, -80.0, 0.0), 8.0, -10.0, 0.0, 0.0),
        ],
    );

    // Cube lines
    assemble_line_strip(
        cmd,
        renderer_entity,
        4,
        vec![
            MeshVertexData::new((-25.0, -25.0, -25.0), 7.0, -30.0, 0.0, 0.0),
            MeshVertexData::new((25.0, -25.0, -25.0), 7.0, -30.0, 0.0, 1.0),
            MeshVertexData::new((25.0, -25.0, 25.0), 7.0, -30.0, 0.0, 2.0),
            MeshVertexData::new((-25.0, -25.0, 25.0), 7.0, -30.0, 0.0, 3.0),
            MeshVertexData::new((-25.0, -25.0, -25.0), 7.0, -30.0, 0.0, 0.0),
        ],
    );

    assemble_line_strip(
        cmd,
        renderer_entity,
        8,
        vec![
            MeshVertexData::new((-25.0, 25.0, -25.0), 7.0, -30.0, 0.0, 0.0),
            MeshVertexData::new((25.0, 25.0, -25.0), 7.0, -30.0, 0.0, 1.0),
            MeshVertexData::new((25.0, 25.0, 25.0), 7.0, -30.0, 0.0, 2.0),
            MeshVertexData::new((-25.0, 25.0, 25.0), 7.0, -30.0, 0.0, 3.0),
            MeshVertexData::new((-25.0, 25.0, -25.0), 7.0, -30.0, 0.0, 0.0),
        ],
    );

    assemble_line_list(
        cmd,
        renderer_entity,
        12,
        vec![
            MeshVertexData::new((-25.0, -25.0, -25.0), 7.0, -30.0, 0.0, 0.0),
            MeshVertexData::new((-25.0, 25.0, -25.0), 7.0, -30.0, 0.0, 0.0),
            MeshVertexData::new((25.0, -25.0, -25.0), 7.0, -30.0, 0.0, 1.0),
            MeshVertexData::new((25.0, 25.0, -25.0), 7.0, -30.0, 0.0, 1.0),
            MeshVertexData::new((25.0, -25.0, 25.0), 7.0, -30.0, 0.0, 2.0),
            MeshVertexData::new((25.0, 25.0, 25.0), 7.0, -30.0, 0.0, 2.0),
            MeshVertexData::new((-25.0, -25.0, 25.0), 7.0, -30.0, 0.0, 3.0),
            MeshVertexData::new((-25.0, 25.0, 25.0), 7.0, -30.0, 0.0, 3.0),
        ],
    );

    // Mesh Vertex buffer
    cmd.assemble_wgpu_buffer_with_usage::<MeshVertex>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Mesh Vertex Buffer"),
            size: buffer_size_of::<MeshVertexData>() * MAX_MESH_VERTICES as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Mesh Index buffer
    cmd.assemble_wgpu_buffer_with_usage::<MeshIndex>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Mesh Index Buffer"),
            size: buffer_size_of::<u16>() * MAX_MESH_INDICES as BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    assemble_mesh(
        cmd,
        renderer_entity,
        8,
        6 * 6,
        vec![
            MeshVertexData::new((1.0, 1.0, 1.0), 7.0, -14.0, 0.0, 0.0),
            MeshVertexData::new((-1.0, 1.0, 1.0), 7.0, -14.0, 0.0, 0.0),
            MeshVertexData::new((-1.0, 1.0, -1.0), 7.0, -14.0, 0.0, 0.0),
            MeshVertexData::new((1.0, 1.0, -1.0), 7.0, -14.0, 0.0, 0.0),
            MeshVertexData::new((1.0, -1.0, 1.0), 7.0, -14.0, 0.0, 0.0),
            MeshVertexData::new((-1.0, -1.0, 1.0), 7.0, -14.0, 0.0, 0.0),
            MeshVertexData::new((-1.0, -1.0, -1.0), 7.0, -14.0, 0.0, 0.0),
            MeshVertexData::new((1.0, -1.0, -1.0), 7.0, -14.0, 0.0, 0.0),
        ]
        .into_iter()
        .map(|mut vd| {
            vd.position[0] *= 10.0;
            vd.position[1] *= 2.5;
            vd.position[2] *= 2.5;
            vd.position[2] -= 25.0;
            vd
        })
        .collect(),
        vec![
            // Top
            0, 1, 2, 0, 2, 3, // Bottom
            4, 7, 5, 7, 6, 5, // Front
            3, 2, 6, 3, 6, 7, // Back
            0, 5, 1, 0, 4, 5, // Right
            0, 3, 7, 0, 7, 4, // Left
            1, 5, 6, 1, 6, 2,
        ]
        .into_iter()
        .map(|id| id + 8)
        .collect(),
    );

    assemble_triangle_list(
        cmd,
        renderer_entity,
        0,
        0,
        0,
        vec![
            MeshVertexData::new((50.0, -80.0, 0.0), 2.0, -10.0, 0.0, 7.0),
            MeshVertexData::new((90.0, -20.0, 0.0), 4.0, -10.0, 0.0, 7.0),
            MeshVertexData::new((10.0, -20.0, 0.0), 3.0, -10.0, 0.0, 7.0),
            MeshVertexData::new((-50.0, -20.0, 0.0), 9.0, -10.0, 0.0, 3.0),
            MeshVertexData::new((-90.0, -80.0, 0.0), 11.0, -10.0, 0.0, 3.0),
            MeshVertexData::new((-10.0, -80.0, 0.0), 10.0, -10.0, 0.0, 3.0),
        ],
    );

    /*
    assemble_triangle_fan(
        cmd,
        renderer_entity,
        0,
        0,
        0,
        vec![
            MeshVertexData::new((50.0, -40.0), 1.0, -10.0, 0.0, 7.0),
            MeshVertexData::new((50.0, -80.0), 2.0, -10.0, 0.0, 7.0),
            MeshVertexData::new((10.0, -20.0), 3.0, -10.0, 0.0, 7.0),
            MeshVertexData::new((90.0, -20.0), 4.0, -10.0, 0.0, 7.0),
            MeshVertexData::new((50.0, -80.0), 5.0, -10.0, 0.0, 7.0),
            MeshVertexData::new((50.0, -80.0), 5.0, -10.0, 0.0, 7.0),
        ],
    );
    */

    // Load map file
    let map = include_str!("maps/lunatic_fringe.map");
    let map = map.parse::<shambler::shalrath::repr::Map>().unwrap();
    let map_data = build_map_data(map);

    assemble_lines(cmd, renderer_entity, 16, map_data);
}

fn build_map_data(map: shambler::shalrath::repr::Map) -> Vec<LineInstanceData> {
    // Convert to flat structure
    let geo_map = shambler::GeoMap::from(map);

    // Create geo planes from brush planes
    let face_planes = shambler::face::FacePlanes::new(&geo_map.face_planes);

    // Create per-brush hulls from brush planes
    let brush_hulls = shambler::brush::BrushHulls::new(&geo_map.brush_faces, &face_planes);

    // Generate face vertices
    let face_vertices =
        shambler::face::FaceVertices::new(&geo_map.brush_faces, &face_planes, &brush_hulls);

    // Generate flat face normals
    let face_normals = shambler::face::FaceNormals::flat(&face_vertices, &face_planes);

    // Find duplicate faces
    let face_duplicates =
        shambler::face::FaceDuplicates::new(&geo_map.faces, &face_planes, &face_vertices);

    // Generate centers
    let face_centers = shambler::face::FaceCenters::new(&face_vertices);

    // Generate per-plane CCW face indices
    let face_indices = shambler::face::FaceIndices::new(
        &geo_map.face_planes,
        &face_planes,
        &face_vertices,
        &face_centers,
        shambler::face::FaceWinding::Clockwise,
    );

    // Generate tangents
    let face_bases = shambler::face::FaceBases::new(
        &geo_map.faces,
        &face_planes,
        &geo_map.face_offsets,
        &geo_map.face_angles,
        &geo_map.face_scales,
    );

    // Generate line indices
    let lines = shambler::line::Lines::new(&face_indices);

    // Calculate face-face containment
    let face_face_containment = shambler::face::FaceFaceContainment::new(
        &geo_map.faces,
        &face_planes,
        &face_bases,
        &face_vertices,
        &lines,
    );

    // Calculate brush-face containment
    let brush_face_containment = shambler::brush::BrushFaceContainment::new(
        &geo_map.brushes,
        &geo_map.faces,
        &geo_map.brush_faces,
        &brush_hulls,
        &face_vertices,
    );

    // Generate mesh
    let mut mesh_normals: Vec<shambler::Vector3> = Default::default();
    let mut mesh_vertices: Vec<shambler::Vector3> = Default::default();
    let mut mesh_lines: Vec<(shambler::face::FaceId, shambler::line::LineIndices)> =
        Default::default();

    let scale_factor = 1.0;

    for face_id in &geo_map.faces {
        if face_duplicates.contains(&face_id) {
            continue;
        }

        if face_face_containment.is_contained(&face_id) {
            continue;
        }

        if brush_face_containment.is_contained(&face_id) {
            continue;
        }

        let index_offset = mesh_vertices.len();
        let face_lines = &lines.face_lines[&face_id];

        mesh_vertices.extend(face_vertices.vertices(&face_id).unwrap().iter().map(|v| {
            nalgebra::vector![-v.x * scale_factor, v.z * scale_factor, v.y * scale_factor]
        }));
        mesh_normals.extend(
            face_normals[&face_id]
                .iter()
                .map(|n| nalgebra::vector![-n.x, n.z, n.y]),
        );
        mesh_lines.extend(face_lines.iter().map(|line_id| {
            let shambler::line::LineIndices { v0, v1 } = lines.line_indices[line_id];
            (
                *face_id,
                shambler::line::LineIndices {
                    v0: v0 + index_offset,
                    v1: v1 + index_offset,
                },
            )
        }));
    }

    mesh_lines
        .into_iter()
        .map(|(face, shambler::line::LineIndices { v0, v1 })| {
            let texture_id = geo_map.face_textures[&face];
            let texture_name = &geo_map.textures[&texture_id];

            let gradient = if texture_name.contains("blood") {
                0.0
            } else if texture_name.contains("green") {
                1.0
            } else if texture_name.contains("blue") {
                2.0
            } else {
                7.0
            };

            let intensity = if texture_name.ends_with("3") {
                2.0
            } else if texture_name.ends_with("2") {
                4.0
            } else if texture_name.ends_with("1") {
                6.0
            } else {
                0.0
            };

            let v0 = mesh_vertices[v0];
            let v1 = mesh_vertices[v1];

            let v0 = MeshVertexData {
                position: [v0.x, v0.y, v0.z, 1.0],
                intensity,
                delta_intensity: -160.0,
                delta_delta: 0.0,
                gradient,
            };

            let v1 = MeshVertexData {
                position: [v1.x, v1.y, v1.z, 1.0],
                intensity,
                delta_intensity: -160.0,
                delta_delta: 0.0,
                gradient,
            };

            LineInstanceData { v0, v1 }
        })
        .collect()
}

pub fn winit_event_handler<T>(mut f: impl EventLoopHandler<T>) -> impl EventLoopHandler<T> {
    let mut prepare_schedule = serial![
        antigen_wgpu::create_shader_modules_with_usage_system::<BeamLine>(),
        antigen_wgpu::create_shader_modules_with_usage_system::<BeamMesh>(),
        antigen_wgpu::create_shader_modules_with_usage_system::<PhosphorDecay>(),
        antigen_wgpu::create_shader_modules_with_usage_system::<Tonemap>(),
        antigen_wgpu::create_buffers_system::<Uniform>(),
        antigen_wgpu::create_buffers_system::<LineVertex>(),
        antigen_wgpu::create_buffers_system::<LineInstance>(),
        antigen_wgpu::create_buffers_system::<MeshVertex>(),
        antigen_wgpu::create_buffers_system::<MeshIndex>(),
        antigen_wgpu::create_textures_system::<BeamBuffer>(),
        antigen_wgpu::create_textures_system::<BeamDepthBuffer>(),
        antigen_wgpu::create_textures_system::<PhosphorFrontBuffer>(),
        antigen_wgpu::create_textures_system::<PhosphorBackBuffer>(),
        antigen_wgpu::create_textures_system::<Gradients>(),
        antigen_wgpu::create_texture_views_system::<BeamBuffer>(),
        antigen_wgpu::create_texture_views_system::<BeamDepthBuffer>(),
        antigen_wgpu::create_texture_views_system::<PhosphorFrontBuffer>(),
        antigen_wgpu::create_texture_views_system::<PhosphorBackBuffer>(),
        antigen_wgpu::create_texture_views_system::<Gradients>(),
        antigen_wgpu::create_samplers_with_usage_system::<Linear>(),
        antigen_wgpu::buffer_write_system::<Uniform, TotalTimeComponent, f32>(),
        antigen_wgpu::buffer_write_system::<Uniform, DeltaTimeComponent, f32>(),
        antigen_wgpu::buffer_write_system::<Uniform, PerspectiveMatrixComponent, [[f32; 4]; 4]>(),
        antigen_wgpu::buffer_write_system::<Uniform, OrthographicMatrixComponent, [[f32; 4]; 4]>(),
        antigen_wgpu::buffer_write_system::<LineVertex, LineVertexDataComponent, Vec<LineVertexData>>(
        ),
        antigen_wgpu::buffer_write_system::<
            LineInstance,
            LineInstanceDataComponent,
            Vec<LineInstanceData>,
        >(),
        antigen_wgpu::buffer_write_system::<MeshVertex, MeshVertexDataComponent, Vec<MeshVertexData>>(
        ),
        antigen_wgpu::buffer_write_system::<MeshIndex, MeshIndexDataComponent, Vec<u16>>(),
        antigen_wgpu::texture_write_system::<Gradients, GradientDataComponent, GradientData>(),
        phosphor_prepare_system()
    ];

    let mut render_schedule = serial![
        parallel![
            phosphor_update_total_time_system(),
            phosphor_update_delta_time_system(),
        ],
        phosphor_update_oscilloscopes_system(),
        phosphor_update_timers_system(),
        phosphor_render_system(),
        phosphor_update_timestamp_system(),
        antigen_wgpu::device_poll_system(Maintain::Wait),
    ];

    let mut surface_resize_schedule = single![phosphor_resize_system()];
    let mut cursor_moved_schedule = single![phosphor_cursor_moved_system()];

    move |world: &ImmutableWorld,
          event: Event<'static, T>,
          event_loop_window_target: &EventLoopWindowTarget<T>,
          control_flow: &mut ControlFlow| {
        match &event {
            Event::MainEventsCleared => {
                surface_resize_schedule.execute(world);
                prepare_schedule.execute(world);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(_) => {
                    surface_resize_schedule.execute(world);
                }
                WindowEvent::CursorMoved { .. } => cursor_moved_schedule.execute(world),
                _ => (),
            },
            Event::RedrawEventsCleared => {
                render_schedule.execute(world);
            }
            _ => (),
        }

        f(world, event, event_loop_window_target, control_flow);
    }
}
