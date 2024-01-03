#![feature(try_find)]
#![allow(non_snake_case)]
#![feature(box_patterns)]
#![feature(ascii_char)]
#![feature(concat_idents)]
#![feature(const_fmt_arguments_new)]

pub const VERTEX_SHADER_BYTES_RAW: &[u8] = include_bytes!("resources/shader.vert.spv");

fn convert_u8_array_to_u32_array(data: &[u8]) -> Vec<u32> {
    let rawBytes = data;

    debug_assert!(rawBytes.len() % 4 == 0, "{}", "Invalid byte length");

    let mut buffer = Vec::with_capacity(rawBytes.len() / mem::size_of::<u32>());
    let buffer_ptr = buffer.as_mut_ptr() as *mut u32;

    unsafe {
        buffer.set_len(buffer.capacity());
        std::ptr::copy_nonoverlapping(rawBytes.as_ptr(), buffer_ptr as *mut u8, rawBytes.len());
    }
    buffer.clone()
}

pub const FRAGMENT_SHADER_BYTES_RAW: &[u8] = include_bytes!("resources/shader.frag.spv");

use std::ffi::{c_char, CString};
use std::process::exit;
use std::ptr::null;
use std::time::Duration;
use std::{mem, slice};

use ash::vk::{CommandBufferBeginInfo, CommandBufferUsageFlags, DeviceCreateInfo, Fence, Handle, ImageLayout, PhysicalDevice, PhysicalDeviceProperties, PipelineLayout, PipelineShaderStageCreateInfoBuilder, Queue, QueueFlags};
use ash::{vk, Device, Entry};
use itertools::{concat, Itertools};
use paste::paste;
use rustfft::num_traits::{clamp, One, Zero};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::libc::signal;
use sdl2::pixels::Color;

macro_rules! declare_c_string_array {
    ($name:ident, $content:expr) => {
        paste! {
            let [<__backing_array_ $name>] = $content
                .iter()
                .map(|it| CString::new(*it).unwrap())
                .collect::<Vec<_>>();
            let $name = [<__backing_array_ $name>]
                .iter()
                .map(|it| it.as_ptr() as *const i8)
                .collect::<Vec<_>>();
        }
    };
}

// Suggested by ChatGPT, let's see how it will work out.
struct Destructor<F: FnOnce()> {
    destructor: Option<F>,
}

impl<F: FnOnce()> Destructor<F> {
    fn new(destructor: F) -> Self {
        Destructor {
            destructor: Some(destructor),
        }
    }
}

impl<F: FnOnce()> Drop for Destructor<F> {
    fn drop(&mut self) {
        if let Some(destructor) = self.destructor.take() {
            destructor();
        }
    }
}

macro_rules! declare_with_custom_destructor {
    ($name:ident, $content:expr, $destructor:expr) => {
        paste! {
            let $name = $content;
            let [<__guard_ $name>] = {
                Destructor::new(|| (($destructor)($name.clone())))
            };
        }
    };
}

macro_rules! declare_c_string {
    ($name:ident, $content:expr) => {
        paste! {
            let [<__backing_string_ $name>] = CString::new($content).unwrap();
            let $name = [<__backing_string_ $name>].as_ptr();
        }
    };
}

pub fn unsafeStrFromCstr<'a>(cStr: *const c_char) -> &'a str {
    unsafe { std::ffi::CStr::from_ptr(cStr).to_str().unwrap() }
}

pub fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("rust-sdl2 demo", 800, 600)
        .position_centered()
        .vulkan()
        // .hidden()
        .build()
        .unwrap();

    // let mut canvas = window.into_canvas().build().unwrap();

    // We want to load the entry because this otherwise will result in a vulkan entry that potentially
    // doesn't contain the validation layer.
    let entry = unsafe { Entry::load() }.unwrap();
    let app_info = vk::ApplicationInfo {
        api_version: vk::api_version_major(vk::API_VERSION_1_3),
        ..Default::default()
    };

    let availableLayers = unsafe { entry.enumerate_instance_layer_properties().unwrap() };

    for layer in availableLayers {
        println!(
            "spec version: {}, layer name: {}, description: {}",
            layer.spec_version,
            unsafeStrFromCstr(layer.layer_name.as_ptr()),
            unsafeStrFromCstr(layer.description.as_ptr())
        );
    }

    let instance = {
        // I dont want to always specify the backing array myself plus this always gives it a name that
        // wont get in the way. This doesnt cause a memory leak.
        declare_c_string_array!(rawLayers, ["VK_LAYER_KHRONOS_validation"]);
        declare_c_string_array!(
            instanceExtensions,
            concat(vec![
                window.vulkan_instance_extensions().unwrap(),
                vec!["VK_KHR_surface"]
            ])
        );

        let createInfo = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&rawLayers)
            .enabled_extension_names(&instanceExtensions)
            .build();
        unsafe { entry.create_instance(&createInfo, None).unwrap() }
    };

    // window.vulkan_create_surface(instance.handle().as_raw() as sdl2::video::VkInstance);

    let surfaceKhr = vk::SurfaceKHR::from_raw(
        window
            .vulkan_create_surface(instance.handle().as_raw() as sdl2::video::VkInstance)
            .unwrap(),
    );

    let surfaceFunctions = ash::extensions::khr::Surface::new(&entry, &instance);

    let physicalDevices = unsafe { instance.enumerate_physical_devices() }.unwrap();

    // Boxing is nice and shiny but in this case not any better than just copying the result values.
    // Simply because we copy already once from Stack to Heap with boxing!
    fn chooseDevice<'a>(
        instance: &ash::Instance,
        devices: &'a Vec<PhysicalDevice>,
    ) -> Option<(&'a PhysicalDevice, Box<PhysicalDeviceProperties>)> {
        let devicesWithInfo: Vec<(&'a PhysicalDevice, Box<PhysicalDeviceProperties>)> = devices
            .iter()
            .map(|device| unsafe {
                (
                    device,
                    Box::new(instance.get_physical_device_properties(*device)),
                )
            })
            .collect();
        let res: Option<(&'a PhysicalDevice, Box<PhysicalDeviceProperties>)> = devicesWithInfo
            .iter()
            .find(|(_, ref info)| (*info).device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
            .or(devicesWithInfo.first())
            .map(|&(device, ref info)| (device, info.clone()));
        res
    }

    for (&pDevice, box pDeviceInfo) in chooseDevice(&instance, &physicalDevices) {
        let gpuName = unsafe {
            std::ffi::CStr::from_ptr(pDeviceInfo.device_name.as_ptr())
                .to_str()
                .unwrap()
        };

        println!("Selected gpu {}", gpuName);

        let queueFamiliesProps =
            unsafe { instance.get_physical_device_queue_family_properties(pDevice) };

        for queueFamilyProps in &queueFamiliesProps {
            println!("{:?}", queueFamilyProps);
        }

        let (queueGraphicsFamilyPropsIndex, queueGraphicsFamilyProps) = queueFamiliesProps
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(QueueFlags::GRAPHICS))
            .map(|(index, props)| (index as u32, props))
            .expect("Queue family with graphics support");

        println!(
            "Selected graphics queue family at index {} with props: {:?}",
            queueGraphicsFamilyPropsIndex, queueGraphicsFamilyProps
        );

        let (queuePresentFamilyPropsIndex, queuePresentFamilyProps) = queueFamiliesProps
            .iter()
            .enumerate()
            .find(|&(index, _)| -> bool {
                unsafe {
                    surfaceFunctions
                        .get_physical_device_surface_support(pDevice, index as u32, surfaceKhr)
                        .unwrap()
                }
            })
            .map(|(index, props)| (index as u32, props))
            .expect("Queue family with graphics support");

        println!(
            "Selected graphics queue family at index {} with props: {:?}",
            queuePresentFamilyPropsIndex, queuePresentFamilyProps
        );

        // Create device
        declare_with_custom_destructor!(
            device,
            unsafe {
                let queueCreateInfo = {
                    let priorities = [1.0];
                    vec![queueGraphicsFamilyPropsIndex, queuePresentFamilyPropsIndex]
                        .iter()
                        .dedup()
                        .map(|index| {
                            vk::DeviceQueueCreateInfo::builder()
                                .queue_family_index(*index as u32)
                                .queue_priorities(&priorities)
                                .build()
                        })
                        .collect::<Vec<_>>()
                };
                declare_c_string_array!(device_extensions, ["VK_KHR_swapchain"]); // We need some framebuffer after all...
                let createInfo = DeviceCreateInfo::builder()
                    .queue_create_infos(&queueCreateInfo)
                    .enabled_extension_names(&device_extensions)
                    .build();
                instance.create_device(pDevice, &createInfo, None).unwrap()
            },
            |device: Device| unsafe {
                println!("destroying device");
                device.destroy_device(None)
            }
        );

        let swapchainFun = ash::extensions::khr::Swapchain::new(&instance, &device);

        println!("VDevice was created");

        fn getQueue(device: &Device, familyIndex: u32) -> Queue {
            unsafe {
                let info = vk::DeviceQueueInfo2::builder()
                    .queue_family_index(familyIndex)
                    .queue_index(0)
                    .build();
                device.get_device_queue2(&info)
            }
        }

        // Get our queues so we can later actually draw and present something.
        let graphicsQueue = getQueue(&device, queueGraphicsFamilyPropsIndex as u32);
        let presentQueue = getQueue(&device, queuePresentFamilyPropsIndex as u32);

        // Painfully create the Swapchain and retain the create info so we actually which properties were chosen.
        let (swapchain, swapchainCreateInfo) = unsafe {
            let capabilities = surfaceFunctions
                .get_physical_device_surface_capabilities(pDevice, surfaceKhr)
                .expect("If this doesnt work at this point then something is really off");
            let surfaceFormat = {
                let availableFormats =
                    surfaceFunctions.get_physical_device_surface_formats(pDevice, surfaceKhr)
                        .expect("We have a main window at this point so we should get our formats for the surface");
                for format in &availableFormats {
                    println!("Surface format: {:?}", format);
                }
                availableFormats
                    .iter()
                    .find_map(|format| match (format.format, format.color_space) {
                        (vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR) => {
                            Some(*format)
                        }
                        _ => None,
                    })
                    .expect("We should have such a format because it's somewhat common")
            };

            let presentMode = {
                let availableModes = surfaceFunctions
                    .get_physical_device_surface_present_modes(pDevice, surfaceKhr)
                    .unwrap();
                for mode in &availableModes {
                    println!("Present mode: {:?}", mode);
                }
                if availableModes
                    .iter()
                    .any(|mode| *mode == vk::PresentModeKHR::MAILBOX)
                {
                    vk::PresentModeKHR::MAILBOX
                } else {
                    vk::PresentModeKHR::FIFO
                }
            };

            let size = {
                let (minExtent, maxExtent) =
                    (capabilities.min_image_extent, capabilities.max_image_extent);
                let (width, height) = window.vulkan_drawable_size();
                let (adjustedWidth, adjustedHeight) = (
                    clamp(width, minExtent.width, maxExtent.width),
                    clamp(height, minExtent.height, maxExtent.height),
                );
                println!("width: {}, height: {}", adjustedWidth, adjustedHeight);
                vk::Extent2D::builder()
                    .width(adjustedWidth)
                    .height(adjustedHeight)
                    .build()
            };

            let swapchainCreateInfo = {
                let (sharingMode, queueFamilyIndices) =
                    if queueGraphicsFamilyPropsIndex == queuePresentFamilyPropsIndex {
                        (vk::SharingMode::EXCLUSIVE, vec![])
                    } else {
                        (
                            vk::SharingMode::CONCURRENT,
                            vec![queueGraphicsFamilyPropsIndex, queuePresentFamilyPropsIndex],
                        )
                    };
                vk::SwapchainCreateInfoKHR::builder()
                    .surface(surfaceKhr)
                    .min_image_count(capabilities.min_image_count)
                    .image_format(surfaceFormat.format)
                    .image_color_space(surfaceFormat.color_space)
                    .image_extent(size)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(sharingMode)
                    .queue_family_indices(&queueFamilyIndices)
                    .pre_transform(capabilities.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(presentMode)
                    .clipped(true)
                    .build()
            };
            (
                swapchainFun
                    .create_swapchain(&swapchainCreateInfo, None)
                    .expect("this should just work!"),
                swapchainCreateInfo.clone(),
            )
        };

        let __swapchain_destructor = Destructor::new(|| unsafe {
            println!("destroying swapchain");
            swapchainFun.destroy_swapchain(swapchain, None);
        });

        println!("created swapchain");

        let swapchainImages = unsafe { swapchainFun.get_swapchain_images(swapchain) }
            .expect("Swapchain was already successfully created before");

        let swapchainImageViews = {
            swapchainImages
                .iter()
                .map(|image| unsafe {
                    let imageViewCreateInfo = vk::ImageViewCreateInfo::builder()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(swapchainCreateInfo.image_format)
                        .components(
                            vk::ComponentMapping::builder()
                                .r(vk::ComponentSwizzle::IDENTITY)
                                .g(vk::ComponentSwizzle::IDENTITY)
                                .b(vk::ComponentSwizzle::IDENTITY)
                                .a(vk::ComponentSwizzle::IDENTITY)
                                .build(),
                        )
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .build();
                    device
                        .create_image_view(&imageViewCreateInfo, None)
                        .expect("Images are fetched from the already created Swapchain using the same device and should allow image view creation")
                })
                .collect::<Vec<_>>()
        };

        let __swapchain_image_views_destructor = Destructor::new(|| unsafe {
            println!("destroying image views");
            for &imageView in &swapchainImageViews {
                device.destroy_image_view(imageView, None);
            }
        });

        let VERTEX_SHADER_BYTES: Vec<u32> = convert_u8_array_to_u32_array(VERTEX_SHADER_BYTES_RAW);
        let FRAGMENT_SHADER_BYTES: Vec<u32> =
            convert_u8_array_to_u32_array(FRAGMENT_SHADER_BYTES_RAW);

        let vertexShaderModule = unsafe {
            let shaderModuleCreateInfo = vk::ShaderModuleCreateInfo::builder()
                .code(&VERTEX_SHADER_BYTES)
                .build();
            device
                .create_shader_module(&shaderModuleCreateInfo, None)
                .unwrap()
        };

        println!("Created vertex shader module");

        let fragmentShaderModule = unsafe {
            let shaderModuleCreateInfo = vk::ShaderModuleCreateInfo::builder()
                .code(&FRAGMENT_SHADER_BYTES)
                .build();
            device
                .create_shader_module(&shaderModuleCreateInfo, None)
                .unwrap()
        };

        println!("Created fragment shader module");


        let __shaderModules_destructor = Destructor::new(|| unsafe {
            println!("destroying shader modules");
            for shaderModule in [vertexShaderModule, fragmentShaderModule] {
                device.destroy_shader_module(shaderModule, None);
            }
        });

        let shaderEntryPointName = CString::new("main").unwrap();
        // let foo = CString::new("ham").unwrap().as_ptr()

        let baseShaderStageCreateInfoBuilder = || -> PipelineShaderStageCreateInfoBuilder {
            vk::PipelineShaderStageCreateInfo::builder().name(&shaderEntryPointName)
        };

        let vertexShaderStageCreateInfo = baseShaderStageCreateInfoBuilder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertexShaderModule)
            .build();

        let fragmentShaderStageCreateInfo = baseShaderStageCreateInfoBuilder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragmentShaderModule)
            .build();

        let shaderStages = [vertexShaderStageCreateInfo, fragmentShaderStageCreateInfo];

        let dynamicStates = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let pipelineDynamicStateCreateInfo = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamicStates)
            .build();

        // We dont have any vertices that serve as input for the vertex shader.
        let pipelineVertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo::default();

        let pipelineInputAssemblyStateCreateInfo =
            vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false)
                .build();

        let viewport = vk::Viewport::builder()
            .x(0.)
            .y(0.)
            .width(swapchainCreateInfo.image_extent.width as f32)
            .height(swapchainCreateInfo.image_extent.height as f32)
            .min_depth(0.)
            .max_depth(1.)
            .build();

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(swapchainCreateInfo.image_extent.clone())
            .build();

        let pipelineViewports = [viewport];
        let pipelineScissors = [scissor];
        let pipelineViewportStateCreateInfo = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&pipelineViewports)
            .scissors(&pipelineScissors);

        let pipelineRasterizationStateCreateInfo =
            vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false)
                .depth_bias_constant_factor(0.0)
                .depth_bias_clamp(0.0)
                .depth_bias_slope_factor(0.0)
                .build();
        let pipelineMulitsampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0) // we dont set the sample mask
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();
        let pipelineColorBlendAttachmentState = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build();

        let pipelineColorBlendAttachmentStates = [pipelineColorBlendAttachmentState];

        let pipelineColorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&pipelineColorBlendAttachmentStates)
            .blend_constants([0., 0., 0., 0.])
            .build();

        let pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo::default();

        let (pipelineLayout, _) = {
            let it =
                unsafe {
                    device
                        .create_pipeline_layout(&pipelineLayoutCreateInfo, None)
                        .unwrap()
                };
            (it, Destructor::new((move |ham: Device, spam: PipelineLayout| {
                move || unsafe {
                    println!("destroying pipeline layout");
                    ham.destroy_pipeline_layout(spam, None);
                }
            })(device.clone(), it.clone())))
        };

        let attachmentDescription = vk::AttachmentDescription::builder()
            .format(swapchainCreateInfo.image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();

        let attachmentReference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let colorAttachmentsRefs = [attachmentReference];

        let subpassDescription = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&colorAttachmentsRefs)
            .build();

        let colorAttachmentDescriptions = [attachmentDescription];
        let subpasses = [subpassDescription];

        let subpassDependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let renderPassCreateInfo = vk::RenderPassCreateInfo::builder()
            .attachments(&colorAttachmentDescriptions)
            .subpasses(&subpasses)
            .dependencies(slice::from_ref(&subpassDependency))
            .build();

        let renderPass = unsafe {
            device
                .create_render_pass(&renderPassCreateInfo, None)
                .unwrap()
        };
        println!("Created render pass");

        let graphicsPipelineCreateInfo = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shaderStages)
            .vertex_input_state(&pipelineVertexInputStateCreateInfo)
            .input_assembly_state(&pipelineInputAssemblyStateCreateInfo)
            .viewport_state(&pipelineViewportStateCreateInfo)
            .rasterization_state(&pipelineRasterizationStateCreateInfo)
            .multisample_state(&pipelineMulitsampleStateCreateInfo)
            // .depth_stencil_state(nullptr)
            .color_blend_state(&pipelineColorBlendStateCreateInfo)
            .dynamic_state(&pipelineDynamicStateCreateInfo)
            .layout(pipelineLayout)
            .render_pass(renderPass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .build();

        let graphicsPipelineCreateInfos = [graphicsPipelineCreateInfo];

        let graphicsPipelines = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &graphicsPipelineCreateInfos,
                    None,
                )
                .unwrap()
        };

        let graphicsPipeline = *graphicsPipelines.first().unwrap();

        println!("Created graphics pipeline");

        let framebuffers = swapchainImageViews
            .iter()
            .enumerate()
            .map(|(index, imageView)| unsafe {
                let attachments = [*imageView];
                let framebufferCreateInfo = vk::FramebufferCreateInfo::builder()
                    .render_pass(renderPass)
                    .attachments(&attachments)
                    .width(swapchainCreateInfo.image_extent.width)
                    .height(swapchainCreateInfo.image_extent.height)
                    .layers(1);
                device
                    .create_framebuffer(&framebufferCreateInfo, None)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        println!("Created framebuffers");

        let commandPoolCreateInfo = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queueGraphicsFamilyPropsIndex)
            .build();

        declare_with_custom_destructor!(
            commandPool,
            unsafe {
                device
                    .create_command_pool(&commandPoolCreateInfo, None)
                    .unwrap()
            },
            |commandPool_| unsafe {
                println!("destroying command pool");
                device.clone().destroy_command_pool(commandPool_, None)
            }
        );
        println!("created command pool");

        // let commandPool =

        let commandBufferAllocateInfo = vk::CommandBufferAllocateInfo::builder()
            .command_pool(commandPool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
            .build();

        let commandBuffers = unsafe {
            device
                .allocate_command_buffers(&commandBufferAllocateInfo)
                .unwrap()
        };

        let commandBuffer = *commandBuffers.first().unwrap();

        let recordCommandBuffer = |imageIndex: usize| {
            println!("Recording command buffer");
            let commandBufferBeginInfo = vk::CommandBufferBeginInfo::builder()
                .flags(CommandBufferUsageFlags::empty())
                .build();

            unsafe {
                device
                    .begin_command_buffer(commandBuffer, &commandBufferBeginInfo)
                    .unwrap();
            }

            let clearColor = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };

            let clearValues = [clearColor];

            let renderPassBeginInfo = vk::RenderPassBeginInfo::builder()
                .render_pass(renderPass)
                .framebuffer(framebuffers[imageIndex])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: swapchainCreateInfo.image_extent,
                })
                .clear_values(&clearValues);
            unsafe {
                device.cmd_begin_render_pass(
                    commandBuffer,
                    &renderPassBeginInfo,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    commandBuffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphicsPipeline,
                );
                device.cmd_set_viewport(commandBuffer, 0, std::slice::from_ref(&viewport));
                device.cmd_set_scissor(commandBuffer, 0, slice::from_ref(&scissor));
                device.cmd_draw(commandBuffer, 3, 1, 0, 0);
                device.cmd_end_render_pass(commandBuffer);
                device.end_command_buffer(commandBuffer).unwrap();
            };
        };

        // let fragShaderStageInfo =;

        // let pipelineInputAssemblyStateCreateInfo =
        //     vk::PipelineInputAssemblyStateCreateInfo::builder()
        //         .flags(vk::PipelineInputAssemblyStateCreateFlags::default())
        //         .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        //         .primitive_restart_enable(true)
        //         .build();

        // canvas.set_draw_color(Color::RGB(0, 255, 255));
        // canvas.clear();
        // canvas.present();
        let mut event_pump = sdl_context.event_pump().unwrap();
        let mut i = 0;
        let mut flip = false;
        let mut currSwapchainImageIndex = 0;
        declare_with_custom_destructor!(
            imageAvailableSemaphore,
            unsafe {
                let info = vk::SemaphoreCreateInfo::default();
                device.create_semaphore(&info, None).unwrap()
            },
            |semaphore| unsafe { device.destroy_semaphore(semaphore, None) }
        );
        declare_with_custom_destructor!(
            renderFinishedSemaphore,
            unsafe {
                let info = vk::SemaphoreCreateInfo::default();
                device.create_semaphore(&info, None).unwrap()
            },
            |semaphore| unsafe { device.destroy_semaphore(semaphore, None) }
        );
        declare_with_custom_destructor!(
            inFlightFence,
            unsafe {
                let info = vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build();
                device.create_fence(&info, None).unwrap()
            },
            |fence| unsafe { device.destroy_fence(fence, None) }
        );
        'running: loop {
            if i % 255 == 0 {
                flip = !flip;
                if !flip {
                    i = 255;
                }
            }
            if flip {
                i = i + 1;
            } else {
                i = i - 1;
            }
            // println!("i: {}", i);
            // canvas.set_draw_color(Color::RGB(i, 64, 255 - i));
            // canvas.clear();
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => break 'running,
                    _ => {}
                }
            }

            unsafe {
                device
                    .wait_for_fences(slice::from_ref(&inFlightFence), true, u64::MAX)
                    .expect("TODO: panic message");

                device
                    .reset_fences(slice::from_ref(&inFlightFence))
                    .expect("TODO: panic message");

                let (imageIndex, _) = swapchainFun
                    .acquire_next_image(swapchain, u64::MAX, imageAvailableSemaphore, Fence::null())
                    .unwrap();

                device
                    .reset_command_buffer(commandBuffer, vk::CommandBufferResetFlags::empty())
                    .unwrap();

                recordCommandBuffer(imageIndex as usize);

                let submitInfo = vk::SubmitInfo::builder()
                    .wait_semaphores(slice::from_ref(&imageAvailableSemaphore))
                    .wait_dst_stage_mask(slice::from_ref(
                        &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    ))
                    .command_buffers(&commandBuffers)
                    .signal_semaphores(slice::from_ref(&renderFinishedSemaphore))
                    .build();
                println!("Submitting commands");
                device
                    .queue_submit(graphicsQueue, slice::from_ref(&submitInfo), inFlightFence)
                    .unwrap();
                let presentInfo = vk::PresentInfoKHR::builder()
                    .wait_semaphores(slice::from_ref(&renderFinishedSemaphore))
                    .swapchains(slice::from_ref(&swapchain))
                    .image_indices(slice::from_ref(&imageIndex))
                    .build();
                swapchainFun
                    .queue_present(presentQueue, &presentInfo)
                    .unwrap();
            }

            // The rest of the game loop goes here...
            // canvas.present();
            ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
        }
    }

    exit(0);
}
