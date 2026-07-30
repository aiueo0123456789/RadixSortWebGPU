import { HillisSteeleScanGPU } from "./HillisSteeleScanGPU.js";

class PingPong {
  constructor(
    device,
    processA_BindGroupLayout,
    processB_BindGroupLayout,
    bitIndicesBuffer,
    size,
  ) {
    this.ping = device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.pong = device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.prefixSum = device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.invertedBit = device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    this.processA_PingToPongBindGroup = device.createBindGroup({
      layout: processA_BindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.invertedBit,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.ping,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processA_PongToPingBindGroup = device.createBindGroup({
      layout: processA_BindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.invertedBit,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.pong,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processB_PingToPongBindGroup = device.createBindGroup({
      layout: processB_BindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.pong,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.ping,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.prefixSum,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.invertedBit,
          },
        },
        {
          binding: 4,
          resource: {
            buffer: bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processB_PongToPingBindGroup = device.createBindGroup({
      layout: processB_BindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.ping,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.pong,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.prefixSum,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.invertedBit,
          },
        },
        {
          binding: 4,
          resource: {
            buffer: bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
  }
}

export class RadixSortGPU {
  constructor(device) {
    this.device = device;

    /** @type {Map<number, PingPong>} */
    this.pingPongMap = new Map();

    this.hillisSteeleScanGPU = new HillisSteeleScanGPU(device);

    this._init();
  }

  _init() {
    this.maxPasses = 16; // 16ビット比較
    const stride = this.device.limits.minUniformBufferOffsetAlignment;
    const params = new Uint32Array((stride / 4) * this.maxPasses);

    this.bitIndicesBuffer = this.device.createBuffer({
      size: stride * this.maxPasses,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    for (let p = 0; p < this.maxPasses; p++) {
      params[p * (stride / 4)] = p;
    }
    this.device.queue.writeBuffer(this.bitIndicesBuffer, 0, params);

    this.processA_BindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
            hasDynamicOffset: true,
            minBindingSize: 16,
          },
        },
      ],
    });

    this.processB_BindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
            hasDynamicOffset: true,
            minBindingSize: 16,
          },
        },
      ],
    });

    // ビット反転
    this.processA_Pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.processA_BindGroupLayout],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "radixSortA",
          code: /* wgsl */ `
struct Uniforms {
  bitIndex: u32,
}

@group(0) @binding(0) var<storage, read_write> invertedBit: array<u32>;
@group(0) @binding(1) var<storage, read> dataRead: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (arrayLength(&dataRead) <= i) {
    return;
  }
  let b = (dataRead[i] >> uniforms.bitIndex) & 1u;
  invertedBit[i] = 1u - b;
}
          `,
        }),
        entryPoint: "main",
      },
    });

    // 移動
    this.processB_Pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.processB_BindGroupLayout],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "radixSortB",
          code: /* wgsl */ `
struct Uniforms {
  bitIndex: u32,
}

@group(0) @binding(0) var<storage, read_write> dataWrite: array<u32>; // 並び替え後
@group(0) @binding(1) var<storage, read> dataRead: array<u32>; // 並び替え前
@group(0) @binding(2) var<storage, read> prefixSum: array<u32>;   // 累積和
@group(0) @binding(3) var<storage, read> invertedBit: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let i = globalId.x;
  if (arrayLength(&dataRead) <= i) {
    return;
  }

  let value = dataRead[i];
  let b = (value >> uniforms.bitIndex) & 1u;

  let f = prefixSum[i] - invertedBit[i];

  let lastIndex = arrayLength(&invertedBit) - 1u;
  let tf = invertedBit[lastIndex] + prefixSum[lastIndex] - invertedBit[lastIndex]; // totalFalse
  let t = i - f + tf;

  let d = select(f, t, b == 1u);

  dataWrite[d] = value;
}
          `,
        }),
        entryPoint: "main",
      },
    });
  }

  dispatch(encoder, computePass, buffer) {
    const stride = this.device.limits.minUniformBufferOffsetAlignment; // 通常 256
    if (!this.pingPongMap.has(buffer.size)) {
      this.pingPongMap.set(
        buffer.size,
        new PingPong(
          this.device,
          this.processA_BindGroupLayout,
          this.processB_BindGroupLayout,
          this.bitIndicesBuffer,
          buffer.size,
        ),
      );
    }

    const pingPong = this.pingPongMap.get(buffer.size);

    computePass.end();

    encoder.copyBufferToBuffer(buffer, 0, pingPong.ping, 0, buffer.size);

    let radixSortPass = encoder.beginComputePass();
    const length = buffer.size / 4;
    for (let bitIndex = 0; bitIndex < this.maxPasses; ++bitIndex) {
      let isPingToPong = (bitIndex & 1) === 0;
      radixSortPass.setPipeline(this.processA_Pipeline);
      radixSortPass.setBindGroup(
        0,
        isPingToPong
          ? pingPong.processA_PingToPongBindGroup
          : pingPong.processA_PongToPingBindGroup,
        [bitIndex * stride],
      );
      radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));

      radixSortPass = this.hillisSteeleScanGPU.dispatch(
        encoder,
        radixSortPass,
        pingPong.invertedBit,
        pingPong.prefixSum,
      );

      radixSortPass.setPipeline(this.processB_Pipeline);
      radixSortPass.setBindGroup(
        0,
        isPingToPong
          ? pingPong.processB_PingToPongBindGroup
          : pingPong.processB_PongToPingBindGroup,
        [bitIndex * stride],
      );
      radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));
    }
    radixSortPass.end();
    encoder.copyBufferToBuffer(
      (this.maxPasses & 1) === 0 ? pingPong.ping : pingPong.pong,
      0,
      buffer,
      0,
      buffer.size,
    );
    return encoder.beginComputePass();
  }
}
