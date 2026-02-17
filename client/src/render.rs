use shared::{
    math::{Quaternion, Transform3, Vector3},
    utility::{Entity, SparseSet},
};
use std::{error::Error, sync::Arc};
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::{ClearValue, Format},
    image::{Image, ImageCreateInfo, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{event_loop::ActiveEventLoop, window::Window};

#[derive(Debug, Clone)]
pub enum CameraMode {
    Orbit {
        theta: f32,
        azimuth: f32,
        distance: f32,
        target: Entity,
    },
    Fixed {
        target: Entity,
        offset: Transform3,
    },
    Freecam,
}

impl Default for CameraMode {
    fn default() -> Self {
        Self::Freecam
    }
}

pub struct CameraData {
    pub mode: CameraMode,
    pub fov_y: f32,
}

impl Default for CameraData {
    fn default() -> Self {
        Self {
            mode: Default::default(),
            fov_y: 60.0,
        }
    }
}

pub struct CameraInput {
    pub delta_x: f32,
    pub delta_y: f32,
    pub delta_scroll: f32,
}

impl CameraData {
    pub fn orbit(target_id: usize) -> Self {
        Self {
            mode: CameraMode::Orbit {
                theta: 0.0,
                azimuth: 0.0,
                distance: 10.0,
                target: target_id,
            },
            fov_y: 60.0,
        }
    }

    pub fn fixed(target_entity: Entity, offset: Transform3) -> Self {
        Self {
            mode: CameraMode::Fixed {
                target: target_entity,
                offset,
            },
            fov_y: 60.0,
        }
    }

    pub fn handle_input(&mut self, input: CameraInput) {
        if let CameraMode::Orbit {
            theta,
            azimuth,
            distance,
            ..
        } = &mut self.mode
        {
            *theta += input.delta_x;
            *azimuth += input.delta_y;
            *distance += input.delta_scroll;
        }
    }

    pub fn update_tranform(&self, transforms: &mut SparseSet<Transform3>, entity: Entity) {
        match self.mode {
            CameraMode::Orbit {
                theta,
                azimuth,
                distance,
                target,
            } => {
                let rotation = Quaternion::from_euler(0.0, azimuth, theta);
                let offset_position = rotation.rotate_vector(Vector3::X) * distance;
                let target_position = transforms[target].position;

                transforms[entity].position = target_position - offset_position;
                transforms[entity].rotation = rotation;
            }
            CameraMode::Fixed { target, offset } => {
                transforms[entity] = transforms[target] * offset
            }
            CameraMode::Freecam => {}
        }
    }
}

pub mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shader/vertex.glsl"
    }
}

pub mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shader/fragment.glsl"
    }
}

#[derive(
    Debug, Clone, Copy, Default, BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex,
)]
#[repr(C)]
pub struct MeshVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

pub struct Renderable {
    pub vertices: Subbuffer<[MeshVertex]>,
    pub indices: Subbuffer<[u32]>,
    pub index_count: u32,
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushConstants {
    model: [[f32; 4]; 4],
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

struct DeviceSelector;
impl DeviceSelector {
    fn select(
        instance: &Arc<Instance>,
        event_loop: &ActiveEventLoop,
        extensions: &DeviceExtensions,
    ) -> Result<(Arc<PhysicalDevice>, u32), Box<dyn Error>> {
        instance
            .enumerate_physical_devices()?
            .filter(|d| d.supported_extensions().contains(extensions))
            .filter_map(|d| Self::find_queue_family(&d, event_loop).map(|i| (d, i)))
            .min_by_key(|(d, _)| Self::device_priority(d))
            .ok_or_else(|| "No suitable GPU found".into())
    }

    fn find_queue_family(
        device: &Arc<PhysicalDevice>,
        event_loop: &ActiveEventLoop,
    ) -> Option<u32> {
        device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(idx, q)| {
                q.queue_flags.intersects(QueueFlags::GRAPHICS)
                    && device.presentation_support(idx as u32, event_loop).is_ok()
            })
            .map(|i| i as u32)
    }

    fn device_priority(device: &Arc<PhysicalDevice>) -> u32 {
        match device.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        }
    }
}

struct SwapchainManager {
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    image_views: Vec<Arc<ImageView>>,
}

impl SwapchainManager {
    fn new(
        device: Arc<Device>,
        surface: Arc<Surface>,
        window: &Window,
    ) -> Result<Self, Box<dyn Error>> {
        let capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())?;
        let formats = device
            .physical_device()
            .surface_formats(&surface, Default::default())?;

        let image_format = formats
            .iter()
            .find(|(f, _)| *f == Format::R8G8B8A8_UNORM)
            .or_else(|| formats.first())
            .ok_or("No surface formats available")?
            .0;

        let composite_alpha = capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .ok_or("No composite alpha modes supported")?;

        let (swapchain, images) = Swapchain::new(
            device,
            surface,
            SwapchainCreateInfo {
                min_image_count: capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        )?;

        let image_views = Self::make_views(&images)?;
        Ok(Self {
            swapchain,
            images,
            image_views,
        })
    }

    fn recreate(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        let (sc, imgs) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: window.inner_size().into(),
            ..self.swapchain.create_info()
        })?;

        self.swapchain = sc;
        self.images = imgs;
        self.image_views = Self::make_views(&self.images)?;

        Ok(())
    }

    fn make_views(images: &[Arc<Image>]) -> Result<Vec<Arc<ImageView>>, Box<dyn Error>> {
        images
            .iter()
            .map(|i| ImageView::new_default(i.clone()).map_err(Into::into))
            .collect()
    }

    fn format(&self) -> Format {
        self.swapchain.image_format()
    }

    fn extent(&self) -> [u32; 2] {
        let e = self.images[0].extent();
        [e[0], e[1]]
    }
}

struct FrameSync {
    previous_frame: Option<Box<dyn GpuFuture>>,
}

impl FrameSync {
    fn new() -> Self {
        Self {
            previous_frame: None,
        }
    }
    fn wait_for_previous(&mut self) {
        if let Some(mut p) = self.previous_frame.take() {
            p.cleanup_finished();
        }
    }
    fn store(&mut self, f: Box<dyn GpuFuture>) {
        self.previous_frame = Some(f);
    }
    fn handle_error(&mut self, device: Arc<Device>) {
        self.previous_frame = Some(sync::now(device).boxed());
    }
}

fn perspective(fov_y_deg: f32, aspect: f32, z_near: f32, z_far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y_deg.to_radians() / 2.0).tan();
    let range = z_far - z_near;

    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, z_far / range, 1.0],
        [0.0, 0.0, -(z_near * z_far) / range, 0.0],
    ]
}

fn model_matrix(t: &Transform3) -> [[f32; 4]; 4] {
    let r = t.rotation;
    let p = t.position;

    let c0x = 1.0 - 2.0 * (r.y * r.y + r.z * r.z);
    let c0y = 2.0 * (r.x * r.y + r.w * r.z);
    let c0z = 2.0 * (r.x * r.z - r.w * r.y);

    let c1x = 2.0 * (r.x * r.y - r.w * r.z);
    let c1y = 1.0 - 2.0 * (r.x * r.x + r.z * r.z);
    let c1z = 2.0 * (r.y * r.z + r.w * r.x);

    let c2x = 2.0 * (r.x * r.z + r.w * r.y);
    let c2y = 2.0 * (r.y * r.z - r.w * r.x);
    let c2z = 1.0 - 2.0 * (r.x * r.x + r.y * r.y);

    [
        [c0x, c0y, c0z, 0.0],
        [c1x, c1y, c1z, 0.0],
        [c2x, c2y, c2z, 0.0],
        [p.x, p.y, p.z, 1.0],
    ]
}

fn view_matrix(t: &Transform3) -> [[f32; 4]; 4] {
    let r = t.rotation;
    let p = t.position;

    let c0x = 1.0 - 2.0 * (r.y * r.y + r.z * r.z);
    let c0y = 2.0 * (r.x * r.y + r.w * r.z);
    let c0z = 2.0 * (r.x * r.z - r.w * r.y);

    let c1x = 2.0 * (r.x * r.y - r.w * r.z);
    let c1y = 1.0 - 2.0 * (r.x * r.x + r.z * r.z);
    let c1z = 2.0 * (r.y * r.z + r.w * r.x);

    let c2x = 2.0 * (r.x * r.z + r.w * r.y);
    let c2y = 2.0 * (r.y * r.z - r.w * r.x);
    let c2z = 1.0 - 2.0 * (r.x * r.x + r.y * r.y);

    let tx = c1x * p.x + c1y * p.y + c1z * p.z;
    let ty = c2x * p.x + c2y * p.y + c2z * p.z;
    let tz = -(c0x * p.x + c0y * p.y + c0z * p.z);

    [
        [-c1x, -c2x, c0x, 0.0],
        [-c1y, -c2y, c0y, 0.0],
        [-c1z, -c2z, c0z, 0.0],
        [tx, ty, tz, 1.0],
    ]
}

pub struct Renderer {
    should_recreate_swapchain: bool,

    library: Arc<VulkanLibrary>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    window: Arc<Window>,
    surface: Arc<Surface>,

    render_pass: Arc<RenderPass>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
    depth_image_view: Arc<ImageView>,
    camera_buffer: Subbuffer<CameraUniform>,

    swapchain_manager: SwapchainManager,
    frame_sync: FrameSync,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl Renderer {
    pub fn new(event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        let library = VulkanLibrary::new()?;
        let instance = Self::create_instance(&library, event_loop)?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) =
            DeviceSelector::select(&instance, event_loop, &device_extensions)?;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;

        let queue = queues.next().ok_or("Failed to get queue")?;
        let window = Arc::new(event_loop.create_window(Window::default_attributes())?);
        let surface = Surface::from_window(instance.clone(), window.clone())?;

        let swapchain_manager = SwapchainManager::new(device.clone(), surface.clone(), &window)?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let vertex = vertex_shader::load(device.clone())?;
        let fragment = fragment_shader::load(device.clone())?;

        let render_pass = Self::create_render_pass(&device, swapchain_manager.format())?;
        let graphics_pipeline = Self::create_graphics_pipeline(
            &device,
            vertex,
            fragment,
            render_pass.clone(),
            &swapchain_manager,
        )?;

        let depth_image_view =
            Self::create_depth_image(&device, &memory_allocator, swapchain_manager.extent())?;
        let framebuffers = Self::create_framebuffers(
            &swapchain_manager,
            render_pass.clone(),
            depth_image_view.clone(),
        )?;

        let camera_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            CameraUniform {
                view: [[0.0; 4]; 4],
                proj: [[0.0; 4]; 4],
            },
        )?;

        Ok(Self {
            should_recreate_swapchain: false,
            library,
            instance,
            device,
            queue,
            window,
            surface,
            render_pass,
            graphics_pipeline,
            framebuffers,
            depth_image_view,
            camera_buffer,
            swapchain_manager,
            frame_sync: FrameSync::new(),
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
        })
    }

    pub fn create_renderable(
        &self,
        mesh: &shared::geometry::Mesh,
    ) -> Result<Renderable, Box<dyn Error>> {
        let vertices = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            mesh.vertices
                .iter()
                .zip(mesh.uv.iter())
                .zip(mesh.normals.iter())
                .map(|((position, uv), normal)| MeshVertex {
                    position: [position.x, position.y, position.z],
                    normal: [normal.x, normal.y, normal.z],
                    uv: [uv.0, uv.1],
                }),
        )?;

        let indices_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            mesh.indices.iter().map(|i| *i as u32),
        )?;

        Ok(Renderable {
            vertices,
            indices: indices_buffer,
            index_count: mesh.indices.len() as u32,
        })
    }

    pub fn draw_scene(
        &mut self,
        camera: Transform3,
        fov: f32,
        renderables: &SparseSet<Renderable>,
        transforms: &SparseSet<Transform3>,
    ) -> Result<(), Box<dyn Error>> {
        self.frame_sync.wait_for_previous();

        if self.should_recreate_swapchain {
            self.recreate_swapchain()?;
        }

        {
            let extent = self.swapchain_manager.extent();
            let aspect = extent[0] as f32 / extent[1] as f32;
            let mut w = self.camera_buffer.write()?;

            w.view = view_matrix(&camera);
            w.proj = perspective(fov, aspect, 0.1, 1000.0);
        }

        let (image_index, acquire_future) = match self.acquire_image()? {
            Some(r) => r,
            None => return Ok(()),
        };

        let camera_ds = self.create_camera_descriptor_set()?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let extent = self.swapchain_manager.extent();
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some(ClearValue::Float([0.01, 0.01, 0.01, 1.0])),
                        Some(ClearValue::Depth(1.0)),
                    ],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .set_viewport(0, [viewport].into_iter().collect())?
            .bind_pipeline_graphics(self.graphics_pipeline.clone())?
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                self.graphics_pipeline.layout().clone(),
                0,
                camera_ds,
            )?;

        for (id, renderable) in renderables.iter() {
            let push = PushConstants {
                model: model_matrix(&transforms[*id]),
            };
            unsafe {
                builder
                    .push_constants(self.graphics_pipeline.layout().clone(), 0, push)?
                    .bind_vertex_buffers(0, renderable.vertices.clone())?
                    .bind_index_buffer(renderable.indices.clone())?
                    .draw_indexed(renderable.index_count, 1, 0, 0, 0)?;
            }
        }

        builder.end_render_pass(SubpassEndInfo::default())?;

        let command_buffer = builder.build()?;
        self.submit_and_present(acquire_future, command_buffer, image_index)?;
        self.window.request_redraw();

        Ok(())
    }

    pub fn set_recreate_swapchain(&mut self) {
        self.should_recreate_swapchain = true;
    }

    fn create_camera_descriptor_set(&self) -> Result<Arc<DescriptorSet>, Box<dyn Error>> {
        let layout = self
            .graphics_pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or("Missing pipeline layout")?;
        Ok(DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.camera_buffer.clone())],
            [],
        )?)
    }

    fn acquire_image(&mut self) -> Result<Option<(u32, Box<dyn GpuFuture>)>, Box<dyn Error>> {
        match acquire_next_image(self.swapchain_manager.swapchain.clone(), None)
            .map_err(Validated::unwrap)
        {
            Ok((index, suboptimal, future)) => {
                if suboptimal {
                    self.should_recreate_swapchain = true;
                }
                Ok(Some((index, future.boxed())))
            }
            Err(VulkanError::OutOfDate) => {
                self.should_recreate_swapchain = true;
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn submit_and_present(
        &mut self,
        acquire_future: Box<dyn GpuFuture>,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
        image_index: u32,
    ) -> Result<(), Box<dyn Error>> {
        let future = acquire_future
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain_manager.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(f) => {
                self.frame_sync.store(f.boxed());
                Ok(())
            }
            Err(VulkanError::OutOfDate) => {
                self.should_recreate_swapchain = true;
                self.frame_sync.handle_error(self.device.clone());
                Ok(())
            }
            Err(e) => {
                self.frame_sync.handle_error(self.device.clone());
                Err(e.into())
            }
        }
    }

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        self.swapchain_manager.recreate(&self.window)?;
        self.depth_image_view = Self::create_depth_image(
            &self.device,
            &self.memory_allocator,
            self.swapchain_manager.extent(),
        )?;
        self.framebuffers = Self::create_framebuffers(
            &self.swapchain_manager,
            self.render_pass.clone(),
            self.depth_image_view.clone(),
        )?;
        self.should_recreate_swapchain = false;
        Ok(())
    }

    fn create_instance(
        library: &Arc<VulkanLibrary>,
        event_loop: &ActiveEventLoop,
    ) -> Result<Arc<Instance>, Box<dyn Error>> {
        Ok(Instance::new(
            library.clone(),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: Surface::required_extensions(event_loop)?,
                ..Default::default()
            },
        )?)
    }

    fn create_render_pass(
        device: &Arc<Device>,
        color_format: Format,
    ) -> Result<Arc<RenderPass>, Box<dyn Error>> {
        Ok(vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: { format: color_format, samples: 1, load_op: Clear, store_op: Store },
                depth: { format: Format::D32_SFLOAT, samples: 1, load_op: Clear, store_op: DontCare },
            },
            pass: { color: [color], depth_stencil: {depth} },
        )?)
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
        render_pass: Arc<RenderPass>,
        swapchain_manager: &SwapchainManager,
    ) -> Result<Arc<GraphicsPipeline>, Box<dyn Error>> {
        let vs_entry = vs
            .entry_point("main")
            .ok_or("Missing vertex shader entry point")?;
        let fs_entry = fs
            .entry_point("main")
            .ok_or("Missing fragment shader entry point")?;

        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry.clone()),
            PipelineShaderStageCreateInfo::new(fs_entry),
        ];

        let vertex_input_state = MeshVertex::per_vertex().definition(&vs_entry)?;

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())?,
        )?;

        let extent = swapchain_manager.extent();
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };

        Ok(GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    1,
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(
                    Subpass::from(render_pass, 0)
                        .ok_or("Missing subpass")?
                        .into(),
                ),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?)
    }

    fn create_depth_image(
        device: &Arc<Device>,
        allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
    ) -> Result<Arc<ImageView>, Box<dyn Error>> {
        let image = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;
        Ok(ImageView::new_default(image)?)
    }

    fn create_framebuffers(
        swapchain_manager: &SwapchainManager,
        render_pass: Arc<RenderPass>,
        depth_image_view: Arc<ImageView>,
    ) -> Result<Vec<Arc<Framebuffer>>, Box<dyn Error>> {
        swapchain_manager
            .image_views
            .iter()
            .map(|color| {
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![color.clone(), depth_image_view.clone()],
                        ..Default::default()
                    },
                )
                .map_err(Into::into)
            })
            .collect()
    }
}
