use shared::{
    math::{GeometryTree, Quaternion, Transform3, Vector3},
    utility::{Entity, SparseSet},
};
use std::{error::Error, sync::Arc};
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage,
        PrimaryAutoCommandBuffer, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::ClearColorValue,
    image::{Image, ImageCreateInfo, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    padded::Padded,
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
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

                transforms[entity].position =
                    transforms[target].position - rotation.rotate_vector(Vector3::X) * distance;
                transforms[entity].rotation = rotation
            }
            CameraMode::Fixed { target, offset } => {
                transforms[entity] = transforms[target] * offset
            }
            CameraMode::Freecam => {}
        }
    }
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
            .filter(|device| device.supported_extensions().contains(extensions))
            .filter_map(|device| {
                Self::find_queue_family(&device, event_loop).map(|queue_idx| (device, queue_idx))
            })
            .min_by_key(|(device, _)| Self::device_priority(device))
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
            .position(|(idx, queue)| {
                queue.queue_flags.intersects(QueueFlags::GRAPHICS)
                    && device.presentation_support(idx as u32, event_loop).is_ok()
            })
            .map(|idx| idx as u32)
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
        let window_size = window.inner_size();
        let capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())?;

        let available_formats = device
            .physical_device()
            .surface_formats(&surface, Default::default())?;

        let image_format = available_formats
            .iter()
            .find(|(format, _)| *format == vulkano::format::Format::R8G8B8A8_UNORM)
            .or_else(|| available_formats.first())
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
                image_extent: window_size.into(),
                image_usage: ImageUsage::STORAGE
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_DST,
                composite_alpha,
                ..Default::default()
            },
        )?;

        let image_views = Self::create_image_views(&images)?;

        Ok(Self {
            swapchain,
            images,
            image_views,
        })
    }

    fn recreate(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        let window_size = window.inner_size();

        let (new_swapchain, new_images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: window_size.into(),
            ..self.swapchain.create_info()
        })?;

        self.swapchain = new_swapchain;
        self.images = new_images;
        self.image_views = Self::create_image_views(&self.images)?;

        Ok(())
    }

    fn create_image_views(images: &[Arc<Image>]) -> Result<Vec<Arc<ImageView>>, Box<dyn Error>> {
        images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).map_err(Into::into))
            .collect()
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
        if let Some(mut prev) = self.previous_frame.take() {
            prev.cleanup_finished();
        }
    }

    fn store(&mut self, future: Box<dyn GpuFuture>) {
        self.previous_frame = Some(future);
    }

    fn handle_error(&mut self, device: Arc<Device>) {
        self.previous_frame = Some(sync::now(device).boxed());
    }
}

pub mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shader/compute.glsl"
    }
}

pub struct Renderable {
    geometry: Subbuffer<compute_shader::Geometry>,
}

impl Renderable {
    pub fn set_nodes(&mut self, geometry: &GeometryTree) -> Result<(), Box<dyn Error>> {
        if self.geometry.size()
            < geometry.nodes().len() as u64 * size_of::<compute_shader::BSPNode>() as u64
        {
            return Err("Buffer not big enough".into());
        }

        let mut writer = self.geometry.write()?;

        for (i, x) in geometry.nodes().iter().enumerate() {
            let plane = x.plane;

            writer.nodes[i] = compute_shader::BSPNode {
                plane: [
                    plane.normal.x,
                    plane.normal.y,
                    plane.normal.z,
                    plane.distance,
                ],
                positive: x.positive.0,
                negative: x.negative.0,
                padding1: 0,
                padding2: 0,
            }
        }

        Ok(())
    }
}

pub struct Renderer {
    should_recreate_swapchain: bool,
    camera: Subbuffer<compute_shader::ViewData>,
    depth_buffer: Arc<ImageView>,

    library: Arc<VulkanLibrary>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    window: Arc<Window>,
    surface: Arc<Surface>,

    compute_pipeline: Arc<ComputePipeline>,
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

        let compute_shader = compute_shader::load(device.clone())?;
        let compute_pipeline = Self::create_compute_pipeline(&device, compute_shader)?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let camera = Buffer::from_data(
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
            compute_shader::ViewData {
                resolution_x: window.inner_size().width,
                resolution_y: Padded::from(window.inner_size().height),
                camera_position: [0.0, 0.0, 0.0],
                fov: 60.0,
                camera_rotation: [0.0, 0.0, 0.0, 1.0],
            },
        )?;

        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
                format: vulkano::format::Format::R32_SFLOAT,
                extent: [window.inner_size().width, window.inner_size().height, 1],
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let depth_buffer = ImageView::new_default(image)?;

        Ok(Self {
            should_recreate_swapchain: false,
            depth_buffer,
            camera,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            library,
            instance,
            device,
            queue,
            window,
            surface,
            swapchain_manager,
            compute_pipeline,
            frame_sync: FrameSync::new(),
        })
    }

    pub fn create_renderable(&mut self) -> Result<Renderable, Box<dyn Error>> {
        let geometry = Buffer::new_unsized(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            32 * 1024,
        )?;

        Ok(Renderable { geometry })
    }

    fn set_camera(&mut self, transform: Transform3, fov: f32) -> Result<(), Box<dyn Error>> {
        let mut writer = self.camera.write()?;

        let position = transform.position;
        writer.camera_position = [position.x, position.y, position.z];

        let rotation = transform.rotation;
        writer.camera_rotation = [rotation.x, rotation.y, rotation.z, rotation.w];

        writer.fov = fov;

        Ok(())
    }

    fn begin_frame(
        &self,
        image_index: u32,
    ) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, Box<dyn Error>> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder.clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([1e30, 0.0, 0.0, 0.0]),
            ..ClearColorImageInfo::image(self.depth_buffer.image().clone())
        })?;

        builder.clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([0.0, 0.0, 0.0, 0.0]),
            ..ClearColorImageInfo::image(
                self.swapchain_manager.images[image_index as usize].clone(),
            )
        })?;

        Ok(builder)
    }

    fn draw_renderable<'a>(
        &self,
        builder: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderable: &Renderable,
        transform: &Transform3,
        image_index: u32,
    ) -> Result<&'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, Box<dyn Error>> {
        let descriptor_set = self.create_descriptor_set(image_index, renderable)?;

        let position = transform.position;
        let rotation = transform.rotation;

        let push_constant = compute_shader::PushData {
            position: Padded::from([position.x, position.y, position.z]),
            rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
        };

        let extent = self.swapchain_manager.images[0].extent();
        let dispatch = Self::calculate_dispatch(extent, [8, 8, 1]);

        unsafe {
            builder
                .bind_pipeline_compute(self.compute_pipeline.clone())?
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    self.compute_pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )?
                .push_constants(self.compute_pipeline.layout().clone(), 0, push_constant)?
                .dispatch(dispatch)?;
        }

        Ok(builder)
    }

    pub fn draw_scene(
        &mut self,
        camera: Transform3,
        fov: f32,
        renderables: &SparseSet<Renderable>,
        tranforms: &SparseSet<Transform3>,
    ) -> Result<(), Box<dyn Error>> {
        self.frame_sync.wait_for_previous();

        if self.should_recreate_swapchain {
            self.recreate_swapchain()?;
        }

        self.set_camera(camera, fov)?;

        let (image_index, future) = match self.acquire_image()? {
            Some(result) => result,
            None => return Ok(()),
        };

        let mut builder = self.begin_frame(image_index)?;
        for (id, renderable) in renderables.iter() {
            self.draw_renderable(&mut builder, renderable, &tranforms[*id], image_index)?;
        }

        let command_buffer = builder.build()?;

        self.submit_and_present(future, command_buffer, image_index)?;
        self.window.request_redraw();

        Ok(())
    }

    fn acquire_image(&mut self) -> Result<Option<(u32, Box<dyn GpuFuture>)>, Box<dyn Error>> {
        match acquire_next_image(self.swapchain_manager.swapchain.clone(), None)
            .map_err(Validated::unwrap)
        {
            Ok((index, suboptimal, future)) => {
                if suboptimal {
                    self.should_recreate_swapchain = true
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

    fn create_descriptor_set(
        &self,
        image_index: u32,
        renderable: &Renderable,
    ) -> Result<Arc<DescriptorSet>, Box<dyn Error>> {
        let layout = self
            .compute_pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or("Missing pipeline layout")?;

        Ok(DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view(
                    0,
                    self.swapchain_manager.image_views[image_index as usize].clone(),
                ),
                WriteDescriptorSet::image_view(1, self.depth_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.camera.clone()),
                WriteDescriptorSet::buffer(3, renderable.geometry.clone()),
            ],
            [],
        )?)
    }

    fn calculate_dispatch(extent: [u32; 3], workgroup: [u32; 3]) -> [u32; 3] {
        [
            (extent[0] + workgroup[0] - 1) / workgroup[0],
            (extent[1] + workgroup[1] - 1) / workgroup[1],
            1,
        ]
    }

    fn submit_and_present(
        &mut self,
        acquire_future: Box<dyn GpuFuture>,
        command_buffer: Arc<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
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
            Ok(future) => {
                self.frame_sync.store(future.boxed());
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
        self.depth_buffer = ImageView::new_default(Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
                format: vulkano::format::Format::R32_SFLOAT,
                extent: [
                    self.window.inner_size().width,
                    self.window.inner_size().height,
                    1,
                ],
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?)?;

        let mut camera_writer = self.camera.write()?;

        camera_writer.resolution_x = self.window.inner_size().width;
        camera_writer.resolution_y = Padded::from(self.window.inner_size().height);

        self.should_recreate_swapchain = false;

        Ok(())
    }

    pub fn set_recreate_swapchain(&mut self) {
        self.should_recreate_swapchain = true
    }

    fn create_instance(
        library: &Arc<VulkanLibrary>,
        event_loop: &ActiveEventLoop,
    ) -> Result<Arc<Instance>, Box<dyn Error>> {
        let required_extensions = Surface::required_extensions(event_loop)?;

        Ok(Instance::new(
            library.clone(),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?)
    }

    fn create_compute_pipeline(
        device: &Arc<Device>,
        shader: Arc<ShaderModule>,
    ) -> Result<Arc<ComputePipeline>, Box<dyn Error>> {
        let entry_point = shader
            .entry_point("main")
            .ok_or("Shader missing main entry point")?;

        let stage = PipelineShaderStageCreateInfo::new(entry_point);

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())?,
        )?;

        Ok(ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )?)
    }
}
