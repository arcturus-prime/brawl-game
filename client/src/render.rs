// render.rs - Cleaner, more modular version

use std::{error::Error, sync::Arc};
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    image::{Image, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
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
        target: usize,
    },
    Fixed,
}

pub struct CameraData {
    pub mode: CameraMode,
    pub fov_y: f32,
}

pub struct CameraInput {
    pub delta_x: f32,
    pub delta_y: f32,
    pub delta_scroll: f32,
}

impl CameraData {
    pub fn new() -> Self {
        Self {
            mode: CameraMode::Fixed,
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
                image_usage: ImageUsage::STORAGE | ImageUsage::COLOR_ATTACHMENT,
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

pub struct RenderContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    window: Arc<Window>,
    surface: Arc<Surface>,

    swapchain_manager: SwapchainManager,
    compute_pipeline: Arc<ComputePipeline>,
    frame_sync: FrameSync,

    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl RenderContext {
    pub fn render<T: BufferContents>(&mut self, input: T) -> Result<(), Box<dyn Error>> {
        self.frame_sync.wait_for_previous();

        let (image_index, future) = match self.acquire_image()? {
            Some(result) => result,
            None => return Ok(()),
        };

        let descriptor_set = self.create_descriptor_set(input, image_index)?;
        let command_buffer = self.build_command_buffer(descriptor_set)?;

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
                    self.recreate_swapchain()?;
                }
                Ok(Some((index, future.boxed())))
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain()?;
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn create_descriptor_set<T: BufferContents>(
        &self,
        input: T,
        image_index: u32,
    ) -> Result<Arc<DescriptorSet>, Box<dyn Error>> {
        let uniform_buffer = Buffer::from_data(
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
            input,
        )?;

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
                WriteDescriptorSet::buffer(1, uniform_buffer),
            ],
            [],
        )?)
    }

    fn build_command_buffer(
        &self,
        descriptor_set: Arc<DescriptorSet>,
    ) -> Result<Arc<vulkano::command_buffer::PrimaryAutoCommandBuffer>, Box<dyn Error>> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

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
                .dispatch(dispatch)?;
        }

        Ok(builder.build()?)
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
                self.recreate_swapchain()?;
                self.frame_sync.handle_error(self.device.clone());
                Ok(())
            }
            Err(e) => {
                self.frame_sync.handle_error(self.device.clone());
                Err(e.into())
            }
        }
    }

    pub fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        self.swapchain_manager.recreate(&self.window)
    }

    pub fn window_size(&self) -> (u32, u32) {
        let size = self.window.inner_size();

        (size.width, size.height)
    }
}

pub struct Renderer {
    library: Arc<VulkanLibrary>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
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

        Ok(Self {
            library,
            instance,
            memory_allocator: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            )),
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            )),
            device,
            queue,
        })
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

    pub fn create_context(
        &self,
        event_loop: &ActiveEventLoop,
        compute_shader: Arc<ShaderModule>,
    ) -> Result<RenderContext, Box<dyn Error>> {
        let window = Arc::new(event_loop.create_window(Window::default_attributes())?);
        let surface = Surface::from_window(self.instance.clone(), window.clone())?;

        let swapchain_mgr = SwapchainManager::new(self.device.clone(), surface.clone(), &window)?;

        let compute_pipeline = Self::create_compute_pipeline(&self.device, compute_shader)?;

        Ok(RenderContext {
            device: self.device.clone(),
            queue: self.queue.clone(),
            window,
            surface,
            swapchain_manager: swapchain_mgr,
            compute_pipeline,
            frame_sync: FrameSync::new(),
            memory_allocator: self.memory_allocator.clone(),
            descriptor_set_allocator: self.descriptor_set_allocator.clone(),
            command_buffer_allocator: self.command_buffer_allocator.clone(),
        })
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

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }
}
pub mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
            layout(set = 0, binding = 1) uniform InputData {
                float time;
                uint resolution_x;
                uint resolution_y;
                vec3 camera_position;
                vec4 camera_rotation;
            } data_in;

            void main() {
                ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
                vec2 uv = vec2(pixel) / vec2(data_in.resolution_x, data_in.resolution_y);

                // Animated color pattern
                float t = data_in.time;
                vec3 color = 0.5 + 0.5 * cos(t + uv.xyx + vec3(0.0, 2.0, 4.0));

                imageStore(img, pixel, vec4(color, 1.0));
            }
        "
    }
}
