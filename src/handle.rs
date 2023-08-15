use std::sync::Arc;

use wgpu::InstanceDescriptor;
use wgpu::Limits;

/// # GPUHandle
///
/// A reference counted handle to a GPU device and queue.
#[derive(Debug, Clone)]
pub struct GPUHandle {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl std::ops::Deref for GPUHandle {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl GPUHandle {
    pub async fn new() -> Result<Self, anyhow::Error> {
        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .expect("No GPU found given preference");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rumble"),
                    #[cfg(not(target_arch = "wasm32"))]
                    features: wgpu::Features::default() | wgpu::Features::TIMESTAMP_QUERY,
                    //features: wgpu::Features::default() ,
                    #[cfg(target_arch = "wasm32")]
                    features: wgpu::Features::default(),
                    limits: Limits::default(),
                },
                None,
            )
            .await
            .expect("Could not create adapter for GPU device");

        Ok(Self {
            inner: Arc::new(Inner { device, queue }),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
