#![feature(try_find)]
#![allow(non_snake_case)]
#![feature(box_patterns)]
#![feature(ascii_char)]
#![feature(concat_idents)]

use std::ffi::{c_char, CString};
use std::process::exit;
use std::time::Duration;

use ash::vk::{
    DeviceCreateInfo, Handle, PhysicalDevice, PhysicalDeviceProperties, Queue, QueueFlags,
};
use ash::{vk, Device, Entry};
use itertools::{concat, Itertools};
use paste::paste;
use rustfft::num_traits::{clamp, One, Zero};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
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
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();

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
                canvas.window().vulkan_instance_extensions().unwrap(),
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

    let surfaceKhr = vk::SurfaceKHR::from_raw(
        canvas
            .window()
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
    };

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
        let device: Device = unsafe {
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
        };

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
                let (width, height) = canvas.window().vulkan_drawable_size();
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

        // let pipelineInputAssemblyStateCreateInfo =
        //     vk::PipelineInputAssemblyStateCreateInfo::builder()
        //         .flags(vk::PipelineInputAssemblyStateCreateFlags::default())
        //         .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        //         .primitive_restart_enable(true)
        //         .build();

        canvas.set_draw_color(Color::RGB(0, 255, 255));
        canvas.clear();
        canvas.present();
        let mut event_pump = sdl_context.event_pump().unwrap();
        let mut i = 0;
        let mut flip = false;
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
            canvas.set_draw_color(Color::RGB(i, 64, 255 - i));
            canvas.clear();
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

            // The rest of the game loop goes here...
            canvas.present();
            ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
        }

        unsafe {
            println!("destroying image views");
            for imageView in swapchainImageViews {
                device.destroy_image_view(imageView, None);
            }
            println!("destroying swapchain");
            swapchainFun.destroy_swapchain(swapchain, None);

            println!("destroying device");
            device.destroy_device(None);
        };
    }

    exit(0);
}
