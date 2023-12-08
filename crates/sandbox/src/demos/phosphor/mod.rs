// TODO: [✓] Evaluate gradient before phosphor front buffer is written
//           * Will prevent per-pixel animations,
//             but allow for proper color combination in phosphor buffer
//           * Fixes trails all fading to red, as the background is cleared to red
//
// TODO: [✓] Implement MSAA for beam buffer
//           * Will likely involve moving gradient evaluation into the beam vertex shaders
//
// TODO: [✓] Remove gradient code
//
// TODO: 3D Rendering
//       * Need to figure out how to make a phosphor decay model work with a rotateable camera
//       * Want to avoid UE4-style whole screen smearing
//       [✓] Depth offset for beam lines
//           * Ideally should appear as if they're 3D w.r.t. depth vs meshes
//             * No Z-fighting with parent geometry
//           * cos(length(vertex_pos.xy))?
//           * Doesn't achieve desired effect
//             * Render pipeline depth bias not controllable enough
//             * Fragment shader math too complex, needs depth sample
//             * Mesh inset doesn't align with line geometry
//             * Applying in vertex space w/appropriate projection matrix tweaks is viable
//
//       [✓] Compute shader for generating lines from mesh vertex buffer and line index buffer
//
//       [✗] Sort lines back-to-front for proper overlay behavior
//           * Not necessary with proper additive rendering
//
//       [✓] Combine duplicate lines via averaging
//           * Will prevent Z-fighting
//           [✓] Initial implementation
//           [✓] Correct evaluation when va0 == vb1 && vb0 == va1
//
//       [>] Render triangle meshes from map file
//           * Can use to clear a specific area to black w/a given decay rate
//           [✓] Basic implementation
//           [✓] More robust predicate for face pruning
//           [✓] Fix erroneous line indices in map geometry
//               * Lines appear to be using mesh vertices rather than line vertices
//                 * Suggested by the purple color
//               * Not dependent on the presence of other geometry
//               * This would suggest an issue in assemble_map_geometry
//           [✗] Apply interior face filter recursively to prune leftover faces
//               * Doesn't work for closed loops like pillars
//               * Not worth the additional cost to remove caps
//               * May be worth looking at some means to detect and prune caps
//           [✓] Fix index buffer alignment crash with test map
//           [✓] Allow lines to override vertex color
//               * Allows for black geo with colored lines without duplicating verts
//           [ ] Account for portal entities when calculating internal faces
//               * Will need some predicate that can be passed to the InternalFaces constructor
//           [ ] Investigate calculating subsectors from internal faces
//           [ ] Paralellize shambler
//
//       [ ] Figure out how to flush command buffers at runtime
//           * Needed to add, remove components or entities
//           * Want to avoid the Godot issue of stalling the main thread for object allocation
//           * Only the allocating thread should block
//           * This would suggest maintaining one world per thread and
//             shuttling data between them via channel through a centralized 'world manager'
//           * May be wiser to downgrade the RwLock-first approach back to special-case usage
//           * Is there a way to compose systems that doesn't involve customized legion types?
//
//       [ ] Changed<PathComponent> map file reloading
//           * Will allow a system to read ArgsComponent and load a map based on its value
//
//       [ ] Investigate infinite perspective projection + reversed Z
//
// TODO: [ ] Implement HDR bloom
//           * Render mipmaps for final buffer
//           * Render HDR bloom using mipmaps
//
//       [ ] Implement automatic line smearing via compute shader
//           * Double-buffer vertices, use to construct quads
//           * Will need to update backbuffer if lines are added / removed at runtime
//             * Ex. Via frustum culling or portal rendering
//
//       [ ] Is automatic mesh smearing viable?
//
//       [ ] Experiment with scrolling / smearing the phosphor buffer based on camera motion
//           * Should be able to move it in a somewhat perspective-correct fashion
//
//       [ ] Sort meshes front-to-back for optimal z-fail behavior
//
//       [ ] Downsample prototype.wad textures to 1x1px to determine color
//
// TODO: [ ] Implement LUT mapping via 3D texture
//           * Replaces per-fragment gradient animation
//           * Will need to figure out how to generate data
//             * Rendering to 3D texture
//             * Unit LUT is just a color cube with B/RGB/CMY/W vertices
//
//       * MechWarrior 2 gradient skybox background
//         * Setting for underlay / overlay behavior
//         * Overlay acts like a vectrex color overlay
//         * Underlay respects depth and doesn't draw behind solid objects
//

mod assemblage;
mod components;
mod systems;

pub use assemblage::*;
pub use components::*;
use legion::Entity;
pub use systems::*;

use expression::EvalTrait;
use std::{collections::BTreeMap, time::Instant};

use legion::IntoQuery;

use antigen_winit::{
    winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoopWindowTarget},
    },
    AssembleWinit, EventLoopHandler, RedrawUnconditionally, WindowComponent,
};

use antigen_core::{
    parallel, serial, single, AddIndirectComponent, Construct, ImmutableWorld, LazyComponent, Usage,
};

use antigen_wgpu::{
    buffer_size_of,
    wgpu::{
        AddressMode, BufferAddress, BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode,
        Maintain, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, TextureAspect,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    },
    AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent,
};

use antigen_shambler::MapFileComponent;

const HDR_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const MAX_MESH_VERTICES: usize = 10000;
const MAX_MESH_INDICES: usize = 10000;
const MAX_LINE_INDICES: usize = 20000;
const MAX_LINES: usize = MAX_LINE_INDICES / 2;
const CLEAR_COLOR: antigen_wgpu::wgpu::Color = antigen_wgpu::wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: -8.0,
};

pub const BLACK: (f32, f32, f32) = (0.0, 0.0, 0.0);
pub const RED: (f32, f32, f32) = (1.0, 0.0, 0.0);
pub const GREEN: (f32, f32, f32) = (0.0, 1.0, 0.0);
pub const BLUE: (f32, f32, f32) = (0.0, 0.0, 1.0);
pub const WHITE: (f32, f32, f32) = (1.0, 1.0, 1.0);

pub fn orthographic_matrix(aspect: f32, zoom: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let projection = nalgebra_glm::ortho_lh_zo(
        -zoom * aspect,
        zoom * aspect,
        -zoom,
        zoom,
        0.0,
        zoom * (far - near) * 2.0,
    );
    projection.into()
}

pub fn perspective_matrix(
    aspect: f32,
    (ofs_x, ofs_y): (f32, f32),
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let x = ofs_x * std::f32::consts::PI;
    let view = nalgebra_glm::look_at_lh(
        &nalgebra::vector![x.sin() * 300.0, ofs_y * 150.0, -x.cos() * 300.0],
        &nalgebra::vector![0.0, 0.0, 0.0],
        &nalgebra::Vector3::y_axis(),
    );
    let projection = nalgebra_glm::perspective_lh_zo(aspect, (45.0f32).to_radians(), near, far);

    let matrix = projection * view;

    matrix.into()
}

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

    // Generate right quarter-circle
    let mut right = (0..half + 1)
        .map(|i| i as f32 / half as f32)
        .map(|f| {
            let f = f * (std::f32::consts::PI * 0.5);
            (f.sin(), f.cos(), 1.0)
        })
        .collect::<Vec<_>>();

    // Find intermediate vertices and duplicate them with negative Y
    let first = left.remove(0);
    let last = right.pop().unwrap();

    let inter = left
        .into_iter()
        .chain(right.into_iter())
        .flat_map(|(x, y, s)| [(x, -y, s), (x, y, s)]);

    // Stitch the first, intermediate and last vertices back together and convert into line vertex data
    std::iter::once(first)
        .chain(inter)
        .chain(std::iter::once(last))
        .map(|(x, y, s)| LineVertexData {
            position: [x, y, -1.0],
            end: s,
            ..Default::default()
        })
        .collect()
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
    cmd.assemble_wgpu_compute_pipeline_with_usage::<ComputeLineInstances>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<PhosphorDecay>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<BeamLine>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<BeamMesh>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<ComputeLineInstances>(renderer_entity);
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
            sample_count: 4,
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

    // Beam multisample
    cmd.assemble_wgpu_texture_with_usage::<BeamMultisample>(
        renderer_entity,
        TextureDescriptor {
            label: Some("Beam Multisample"),
            size: Extent3d {
                width: 640,
                height: 480,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 4,
            dimension: TextureDimension::D2,
            format: HDR_TEXTURE_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<BeamMultisample>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label: Some("Beam Multisample View"),
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
    cmd.assemble_wgpu_shader_with_usage::<ComputeLineInstances>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/line_instances.wgsl"
            ))),
        },
    );

    cmd.assemble_wgpu_shader_with_usage::<PhosphorDecay>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/phosphor_decay.wgsl"
            ))),
        },
    );

    cmd.assemble_wgpu_shader_with_usage::<BeamLine>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/beam_line.wgsl"
            ))),
        },
    );
    cmd.assemble_wgpu_shader_with_usage::<BeamMesh>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/beam_mesh.wgsl"
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
        PerspectiveMatrixComponent::construct(perspective_matrix(
            640.0 / 480.0,
            (0.0, 0.0),
            1.0,
            500.0,
        )),
        0,
        Some(renderer_entity),
    );

    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        renderer_entity,
        OrthographicMatrixComponent::construct(orthographic_matrix(
            640.0 / 480.0,
            200.0,
            1.0,
            500.0,
        )),
        buffer_size_of::<[[f32; 4]; 4]>(),
        Some(renderer_entity),
    );

    // Mesh Vertex buffer
    cmd.assemble_wgpu_buffer_with_usage::<MeshVertex>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Mesh Vertex Buffer"),
            size: buffer_size_of::<MeshVertexData>() * MAX_MESH_VERTICES as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
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

    // Line Index buffer
    cmd.assemble_wgpu_buffer_with_usage::<LineIndex>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Line Index Buffer"),
            size: buffer_size_of::<u32>() * MAX_LINE_INDICES as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Line Instance buffer
    cmd.assemble_wgpu_buffer_with_usage::<LineInstance>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Line Instance Buffer"),
            size: buffer_size_of::<LineInstanceData>() * MAX_LINES as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Assemble geometry
    let mut vertex_head = 0;
    let mut line_index_head = 0;
    let mut mesh_index_head = 0;

    assemble_test_geometry(
        cmd,
        renderer_entity,
        &mut vertex_head,
        &mut mesh_index_head,
        &mut line_index_head,
    );

    // Load map file
    antigen_shambler::assemble_map_file::<MapFile>(
        cmd,
        renderer_entity,
        std::path::PathBuf::from("crates/sandbox/src/demos/phosphor/maps/index_align_test.map"),
    );

    // Store mesh and line index counts for render system
    let vertex_count = VertexCountComponent::construct(vertex_head);
    let mesh_index_count = MeshIndexCountComponent::construct(mesh_index_head);
    let line_index_count = LineIndexCountComponent::construct(line_index_head);

    cmd.add_component(renderer_entity, vertex_count);
    cmd.add_component(renderer_entity, mesh_index_count);
    cmd.add_component(renderer_entity, line_index_count);
}

fn assemble_test_geometry(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_head: &mut BufferAddress,
    mesh_index_head: &mut BufferAddress,
    line_index_head: &mut BufferAddress,
) {
    // Test triangles
    /*
    assemble_triangle_list(
        cmd,
        renderer_entity,
        &mut vertex_head,
        &mut mesh_index_head,
        0,
        vec![
            MeshVertexData::new((50.0, -80.0, 0.0), BLUE, 1.0, -10.0),
            MeshVertexData::new((90.0, -20.0, 0.0), GREEN, 1.0, -10.0),
            MeshVertexData::new((10.0, -20.0, 0.0), RED, 1.0, -10.0),
            MeshVertexData::new((-50.0, -20.0, 0.0), WHITE, 1.0, -10.0),
            MeshVertexData::new((-90.0, -80.0, 0.0), WHITE, 1.0, -10.0),
            MeshVertexData::new((-10.0, -80.0, 0.0), WHITE, 1.0, -10.0),
        ],
    );
    */

    // Oscilloscopes
    assemble_oscilloscope(
        cmd,
        buffer_target,
        vertex_head,
        line_index_head,
        (-80.0, 40.0, -80.0),
        RED,
        Oscilloscope::new(3.33, 30.0, |f| (f.sin(), f.cos(), f.sin())),
        2.0,
        -1.0,
    );

    assemble_oscilloscope(
        cmd,
        buffer_target,
        vertex_head,
        line_index_head,
        (-80.0, 40.0, 0.0),
        GREEN,
        Oscilloscope::new(2.22, 30.0, |f| (f.sin(), (f * 1.2).sin(), (f * 1.4).cos())),
        2.0,
        -2.0,
    );

    assemble_oscilloscope(
        cmd,
        buffer_target,
        vertex_head,
        line_index_head,
        (-80.0, 40.0, 80.0),
        BLUE,
        Oscilloscope::new(3.33, 30.0, |f| (f.cos(), (f * 1.2).cos(), (f * 1.4).cos())),
        2.0,
        -4.0,
    );

    // Gradient 3 Triangle
    assemble_line_strip(
        cmd,
        buffer_target,
        vertex_head,
        line_index_head,
        vec![
            MeshVertexData::new((-50.0, -20.0, 0.0), RED, RED, 5.0, -20.0),
            MeshVertexData::new((-90.0, -80.0, 0.0), GREEN, GREEN, 4.0, -20.0),
            MeshVertexData::new((-10.0, -80.0, 0.0), BLUE, BLUE, 3.0, -20.0),
            MeshVertexData::new((-50.0, -20.0, 0.0), RED, RED, 2.0, -20.0),
        ],
    );

    // Gradients 0-2 Triangle
    assemble_line_list(
        cmd,
        buffer_target,
        vertex_head,
        line_index_head,
        vec![
            MeshVertexData::new((50.0, -80.0, 0.0), BLUE, BLUE, 7.0, -10.0),
            MeshVertexData::new((90.0, -20.0, 0.0), BLUE, BLUE, 6.0, -10.0),
            MeshVertexData::new((90.0, -20.0, 0.0), GREEN, GREEN, 5.0, -10.0),
            MeshVertexData::new((10.0, -20.0, 0.0), GREEN, GREEN, 4.0, -10.0),
            MeshVertexData::new((10.0, -20.0, 0.0), RED, RED, 3.0, -10.0),
            MeshVertexData::new((50.0, -80.0, 0.0), RED, RED, 2.0, -10.0),
        ],
    );
}

#[legion::system]
#[read_component(Usage<MapFile, MapFileComponent>)]
#[read_component(VertexCountComponent)]
#[read_component(MeshIndexCountComponent)]
#[read_component(LineIndexCountComponent)]
pub fn build_map(
    world: &legion::world::SubWorld,
    cmd: &mut legion::systems::CommandBuffer,
    #[state] done: &mut bool,
) -> Option<()> {
    if *done {
        return None;
    }

    let geo_map = <&Usage<MapFile, MapFileComponent>>::query()
        .iter(world)
        .next()
        .unwrap();

    let geo_map = geo_map.read();
    let geo_map = if let LazyComponent::Ready(geo_map) = &*geo_map {
        geo_map
    } else {
        panic!("Geo map is not ready");
    };

    let (buffer_target, vertex_head) = <(Entity, &VertexCountComponent)>::query()
        .iter(world)
        .next()
        .unwrap();

    let buffer_target = *buffer_target;
    let mut vertex_head = vertex_head.write();
    let vertex_head = &mut *vertex_head;

    let mesh_index_head = <&MeshIndexCountComponent>::query()
        .iter(world)
        .next()
        .unwrap();
    let mut mesh_index_head = mesh_index_head.write();
    let mesh_index_head = &mut *mesh_index_head;

    let line_index_head = <&LineIndexCountComponent>::query()
        .iter(world)
        .next()
        .unwrap();
    let mut line_index_head = line_index_head.write();
    let line_index_head = &mut *line_index_head;

    println!("Building map...");

    // Create geo planes from brush planes
    let face_planes = shambler::face::FacePlanes::new(&geo_map.face_planes);

    // Create per-brush hulls from brush planes
    let brush_hulls = shambler::brush::BrushHulls::new(&geo_map.brush_faces, &face_planes);

    // Generate face vertices
    let face_vertices =
        shambler::face::FaceVertices::new(&geo_map.brush_faces, &face_planes, &brush_hulls);

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

    let face_triangle_indices = shambler::face::FaceTriangleIndices::new(&face_indices);
    let face_line_indices = shambler::line::Lines::new(&face_indices);

    let interior_faces = shambler::face::InteriorFaces::new(
        &geo_map.entity_brushes,
        &geo_map.brush_faces,
        &face_duplicates,
        &face_vertices,
        &face_line_indices,
    );

    // Generate tangents
    let face_bases = shambler::face::FaceBases::new(
        &geo_map.faces,
        &face_planes,
        &geo_map.face_offsets,
        &geo_map.face_angles,
        &geo_map.face_scales,
    );

    // Calculate face-face containment
    let face_face_containment = shambler::face::FaceFaceContainment::new(
        &geo_map.faces,
        &face_planes,
        &face_bases,
        &face_vertices,
        &face_line_indices,
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
    let mut mesh_vertices: Vec<MeshVertexData> = Default::default();
    let mut mesh_indices: Vec<u16> = Default::default();
    let mut line_indices: Vec<u32> = Default::default();

    let scale_factor = 1.0;

    // Gather mesh and line geometry
    let mut face_index_head = *vertex_head as u16;
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

        if !interior_faces.contains(&face_id) {
            continue;
        }

        // Fetch and interpret texture data
        let texture_id = geo_map.face_textures[&face_id];
        let texture_name = &geo_map.textures[&texture_id];

        let color = if texture_name.contains("blood") {
            RED
        } else if texture_name.contains("green") {
            GREEN
        } else if texture_name.contains("blue") {
            BLUE
        } else {
            WHITE
        };

        let intensity = if texture_name.ends_with("3") {
            0.25
        } else if texture_name.ends_with("2") {
            0.375
        } else if texture_name.ends_with("1") {
            0.5
        } else {
            0.125
        };

        let face_vertices = face_vertices.vertices(&face_id).unwrap();
        let vertices = face_vertices
            .iter()
            .map(|v| MeshVertexData {
                position: [v.x * scale_factor, v.z * scale_factor, v.y * scale_factor],
                surface_color: [0.0, 0.0, 0.0],
                line_color: [color.0, color.1, color.2],
                intensity,
                delta_intensity: -8.0,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        mesh_vertices.extend(vertices);

        let face_triangle_indices = face_triangle_indices.get(&face_id).unwrap();
        let triangle_indices = face_triangle_indices
            .iter()
            .map(|i| *i as u16 + face_index_head)
            .collect::<Vec<_>>();
        mesh_indices.extend(triangle_indices);

        let face_lines = &face_line_indices.face_lines[&face_id];
        let face_line_indices = face_lines
            .iter()
            .flat_map(|line_id| {
                let shambler::line::LineIndices { v0, v1 } =
                    face_line_indices.line_indices[line_id];
                [
                    (v0 + face_index_head as usize) as u32,
                    (v1 + face_index_head as usize) as u32,
                ]
            })
            .collect::<Vec<_>>();
        line_indices.extend(face_line_indices);

        face_index_head += face_vertices.len() as u16;
    }

    assemble_mesh(
        cmd,
        buffer_target,
        vertex_head,
        mesh_index_head,
        mesh_vertices,
        mesh_indices,
    );

    assemble_line_indices(cmd, buffer_target, line_index_head, line_indices);

    // Spawn player start entities
    let player_start_entities = geo_map.point_entities.iter().flat_map(|point_entity| {
        let properties = geo_map.entity_properties.get(point_entity)?;
        if let Some(classname) = properties.0.iter().find(|p| p.key == "classname") {
            if classname.value == "info_player_start" {
                Some(properties)
            } else {
                None
            }
        } else {
            None
        }
    });

    for player_start in player_start_entities.into_iter() {
        let origin = player_start.0.iter().find(|p| p.key == "origin").unwrap();
        let mut origin = origin.value.split_whitespace();
        let x = origin.next().unwrap().parse::<f32>().unwrap();
        let y = origin.next().unwrap().parse::<f32>().unwrap();
        let z = origin.next().unwrap().parse::<f32>().unwrap();
        assemble_box_bot(
            cmd,
            buffer_target,
            vertex_head,
            mesh_index_head,
            line_index_head,
            (x, z, y),
        );
    }

    // Spawn oscilloscope entities
    let oscilloscope_entities = geo_map.point_entities.iter().flat_map(|point_entity| {
        let properties = geo_map.entity_properties.get(point_entity)?;
        if let Some(classname) = properties.0.iter().find(|p| p.key == "classname") {
            if classname.value == "oscilloscope" {
                Some(properties)
            } else {
                None
            }
        } else {
            None
        }
    });

    for oscilloscope in oscilloscope_entities.into_iter() {
        let origin = oscilloscope.0.iter().find(|p| p.key == "origin").unwrap();
        let mut origin = origin.value.split_whitespace();
        let x = origin.next().unwrap().parse::<f32>().unwrap();
        let z = origin.next().unwrap().parse::<f32>().unwrap();
        let y = origin.next().unwrap().parse::<f32>().unwrap();
        let origin = (x, y, z);

        let color = oscilloscope.0.iter().find(|p| p.key == "color").unwrap();
        let mut color = color.value.split_whitespace();
        let x = color.next().unwrap().parse::<f32>().unwrap();
        let z = color.next().unwrap().parse::<f32>().unwrap();
        let y = color.next().unwrap().parse::<f32>().unwrap();
        let color = (x, y, z);

        let intensity = oscilloscope
            .0
            .iter()
            .find(|p| p.key == "intensity")
            .unwrap()
            .value
            .parse::<f32>()
            .unwrap();
        let delta_intensity = oscilloscope
            .0
            .iter()
            .find(|p| p.key == "delta_intensity")
            .unwrap()
            .value
            .parse::<f32>()
            .unwrap();
        let speed = oscilloscope
            .0
            .iter()
            .find(|p| p.key == "speed")
            .unwrap()
            .value
            .parse::<f32>()
            .unwrap();
        let magnitude = oscilloscope
            .0
            .iter()
            .find(|p| p.key == "magnitude")
            .unwrap()
            .value
            .parse::<f32>()
            .unwrap();

        let x = &oscilloscope.0.iter().find(|p| p.key == "x").unwrap().value;
        let x = expression::parse_expression(x);

        let y = &oscilloscope.0.iter().find(|p| p.key == "y").unwrap().value;
        let y = expression::parse_expression(y);

        let z = &oscilloscope.0.iter().find(|p| p.key == "z").unwrap().value;
        let z = expression::parse_expression(z);

        assemble_oscilloscope(
            cmd,
            buffer_target,
            vertex_head,
            line_index_head,
            origin,
            color,
            Oscilloscope::new(speed, magnitude, move |f| {
                let vars = [("f", f)].into_iter().collect::<BTreeMap<_, _>>();
                (x.eval(&vars), y.eval(&vars), z.eval(&vars))
            }),
            intensity,
            delta_intensity,
        );
    }

    println!("Map build complete");

    *done = true;

    Some(())
}

pub fn winit_event_handler<T>(mut f: impl EventLoopHandler<T>) -> impl EventLoopHandler<T> {
    let mut prepare_schedule = serial![
        parallel![
            antigen_wgpu::create_shader_modules_with_usage_system::<ComputeLineInstances>(),
            antigen_wgpu::create_shader_modules_with_usage_system::<BeamLine>(),
            antigen_wgpu::create_shader_modules_with_usage_system::<BeamMesh>(),
            antigen_wgpu::create_shader_modules_with_usage_system::<PhosphorDecay>(),
            antigen_wgpu::create_shader_modules_with_usage_system::<Tonemap>(),
            antigen_wgpu::create_buffers_system::<Uniform>(),
            antigen_wgpu::create_buffers_system::<LineVertex>(),
            antigen_wgpu::create_buffers_system::<LineIndex>(),
            antigen_wgpu::create_buffers_system::<LineInstance>(),
            antigen_wgpu::create_buffers_system::<MeshVertex>(),
            antigen_wgpu::create_buffers_system::<MeshIndex>(),
            serial![
                antigen_wgpu::create_textures_system::<BeamBuffer>(),
                antigen_wgpu::create_texture_views_system::<BeamBuffer>(),
            ],
            serial![
                antigen_wgpu::create_textures_system::<BeamDepthBuffer>(),
                antigen_wgpu::create_texture_views_system::<BeamDepthBuffer>(),
            ],
            serial![
                antigen_wgpu::create_textures_system::<BeamMultisample>(),
                antigen_wgpu::create_texture_views_system::<BeamMultisample>(),
            ],
            serial![
                antigen_wgpu::create_textures_system::<PhosphorFrontBuffer>(),
                antigen_wgpu::create_texture_views_system::<PhosphorFrontBuffer>(),
            ],
            serial![
                antigen_wgpu::create_textures_system::<PhosphorBackBuffer>(),
                antigen_wgpu::create_texture_views_system::<PhosphorBackBuffer>(),
            ],
            antigen_wgpu::create_samplers_with_usage_system::<Linear>(),
        ],
        parallel![
            antigen_wgpu::buffer_write_system::<Uniform, TotalTimeComponent, f32>(),
            antigen_wgpu::buffer_write_system::<Uniform, DeltaTimeComponent, f32>(),
            antigen_wgpu::buffer_write_system::<Uniform, PerspectiveMatrixComponent, [[f32; 4]; 4]>(
            ),
            antigen_wgpu::buffer_write_system::<Uniform, OrthographicMatrixComponent, [[f32; 4]; 4]>(
            ),
            antigen_wgpu::buffer_write_system::<
                LineVertex,
                LineVertexDataComponent,
                Vec<LineVertexData>,
            >(),
            antigen_wgpu::buffer_write_system::<LineIndex, LineIndexDataComponent, Vec<u32>>(),
            antigen_wgpu::buffer_write_system::<
                MeshVertex,
                MeshVertexDataComponent,
                Vec<MeshVertexData>,
            >(),
            antigen_wgpu::buffer_write_system::<MeshIndex, MeshIndexDataComponent, Vec<u16>>(),
        ],
        phosphor_prepare_system()
    ];

    let mut render_schedule = serial![
        parallel![
            phosphor_update_total_time_system(),
            phosphor_update_delta_time_system(),
        ],
        phosphor_update_oscilloscopes_system(),
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
                prepare_schedule.execute_and_flush(world);
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
