use super::GPUHandle;
use itertools::Itertools;
use tabled::settings::{object::Rows, Alignment, Modify, Panel, Style};
use tabled::{Table, Tabled};
use wgpu::QuerySet;

//used for formatting table cells
fn float2(n: &f64) -> String {
    format!("{:.2}", n)
}

#[derive(Tabled)]
struct ElapsedTableEntry {
    #[tabled(rename = "Run")]
    run: usize,
    #[tabled(rename = "Elapsed Time (ns)")]
    elapsed_ns: f64,
    #[tabled(rename = "GFLOPs", display_with = "float2")]
    gflops: f64,
}

pub fn runtime_table(runtimes: Vec<(f64, f64)>) {
    let mut entries = Vec::new();
    let average_gflops: f64 =
        runtimes.iter().map(|(_, flops)| flops).sum::<f64>() / runtimes.len() as f64;
    for (idx, (elapsed_ns, flops)) in runtimes.iter().enumerate() {
        entries.push(ElapsedTableEntry {
            run: idx + 1,
            elapsed_ns: *elapsed_ns,
            gflops: *flops,
        });
    }

    let table = Table::new(&entries)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
        .with(Panel::footer(format!(
            "Average GFLOPs: {}",
            float2(&average_gflops)
        )))
        .to_owned();

    println!("{}", table);
}

pub struct Profiler {
    handle: GPUHandle,
    query_set: QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    query_index: u32,
    timestamp_period: f32,
    elements: Vec<usize>,
}

impl Profiler {
    pub fn new(handle: GPUHandle, count: u32) -> Self {
        let query_set = handle.device().create_query_set(&wgpu::QuerySetDescriptor {
            count: count * 2,
            ty: wgpu::QueryType::Timestamp,
            label: None,
        });
        let timestamp_period = handle.queue().get_timestamp_period();

        let buffer_size = (count as usize * 2 * std::mem::size_of::<u64>()) as u64;
        let resolve_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let destination_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            handle,
            query_set,
            resolve_buffer,
            destination_buffer,
            query_index: 0,
            timestamp_period,
            elements: Vec::new(),
        }
    }

    pub fn create_timestamp_queries(
        &mut self,
        elements: usize,
    ) -> wgpu::ComputePassTimestampWrites {
        self.elements.push(elements);
        let beginning_index = self.query_index;
        self.query_index += 1;

        let end_index = self.query_index;
        self.query_index += 1;

        

        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(beginning_index),
            end_of_pass_write_index: Some(end_index),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.query_set,
            0..self.query_index,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    pub fn read_timestamps(&self) {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.handle.device().poll(wgpu::Maintain::Wait);
        let timestamp_view = self
            .destination_buffer
            .slice(
                ..(std::mem::size_of::<u64>() * self.query_index as usize) as wgpu::BufferAddress,
            )
            .get_mapped_range();

        let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_view);
        let mut table_data = Vec::new();
        for (idx, (begin, end)) in timestamps.iter().tuples().enumerate() {
            let elapsed_ns = (end - begin) as f64 * self.timestamp_period as f64;
            let flops = self.elements[idx] as f64 * 2.0;
            let gflops = flops / elapsed_ns;

            table_data.push((elapsed_ns, gflops));
        }

        runtime_table(table_data);
    }
}
