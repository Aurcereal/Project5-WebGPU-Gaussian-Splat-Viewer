import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import pointcloud_wgsl from '../shaders/point_cloud.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {

}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

const minimal2DVertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 8,
    attributes: [
        {
            format: "float32x2",
            offset: 0,
            shaderLocation: 0
        }
    ]
}

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  console.log("shouldn't be here");
  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({code: renderWGSL});
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian test render pipeline',
    layout: 'auto',
    vertex: {
      module: render_shader,
      buffers: [minimal2DVertexBufferLayout],
      entryPoint: 'vs_main'
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{format: presentation_format }]
    },
    primitive: {
      topology: 'triangle-list'
    }
  })

  const camera_bind_group = device.createBindGroup({
    label: 'point cloud camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'point cloud gaussians',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });

  const splat_buffer = device.createBuffer({
    label: 'splat buffer',
    size: pc.num_points * 48,
    usage: GPUBufferUsage.STORAGE
  });

  const splat_bind_group = device.createBindGroup({
    label: 'splats bind group',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      {binding: 0, resource: {buffer: splat_buffer }}
    ]
  });

  const indirect_buffer = device.createBuffer({
    label: 'indirect buffer',
    size: 16,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(indirect_buffer, 0, new Uint32Array([
    6, pc.num_points, 0, 0 // NumVerts, NumInstances, StartVertex, StartInstance
  ])); // TODO make instances


  const vertex_array = new Float32Array([
    -1, -1,
    1, -1,
    1, 1,
    -1, -1,
    -1, 1,
    1, 1
  ]);
  const vertex_buffer = device.createBuffer({
    label: 'quad vertex buffer',
    size: vertex_array.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX
  });
  device.queue.writeBuffer(vertex_buffer, 0, vertex_array);

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      sorter.sort(encoder);

      const pass = encoder.beginRenderPass({
        label: 'gaussian test render',
        colorAttachments: [
          {
            view: texture_view,
            loadOp: 'clear',
            storeOp: 'store',
          }
        ],
      });

      pass.setPipeline(render_pipeline);
      pass.setBindGroup(0, camera_bind_group);
      pass.setBindGroup(1, gaussian_bind_group);

      pass.setVertexBuffer(0, vertex_buffer);

      pass.drawIndirect(indirect_buffer, 0);
      pass.end();
    },
    camera_buffer,
  };
}
