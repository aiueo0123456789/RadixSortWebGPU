import { HillisSteeleScanGPU } from "./HillisSteeleScanGPU.js";

export class RadixSortGPU {
  constructor(device) {
    this.device = device;

    this.hillisSteeleScanGPU = new HillisSteeleScanGPU(device);

    this._init();
  }

  _init() {
    this.maxBitIndex = 16; // 16ビット比較

    this.maxLenght = 10 ** 6;
    this.maxNum = 2 ** this.maxBitIndex;

    const stride = this.device.limits.minUniformBufferOffsetAlignment;
    const params = new Uint32Array((stride / 4) * this.maxBitIndex);

    this.bitIndicesBuffer = this.device.createBuffer({
      size: stride * this.maxBitIndex,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    for (let p = 0; p < this.maxBitIndex; p++) {
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

    this.makeOffset_BindGroupLayout = this.device.createBindGroupLayout({
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
      ],
    });

    this.paramsBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
          },
        },
      ],
    });

    // ビット反転
    this.processA_Pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [
          this.processA_BindGroupLayout,
          this.paramsBindGroupLayout,
        ],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "radixSortA",
          code: /* wgsl */ `
struct Params {
  elementLenght: u32,
}

struct Uniforms {
  bitIndex: u32,
}

@group(0) @binding(0) var<storage, read_write> invertedBit: array<u32>;
@group(0) @binding(1) var<storage, read> dataRead: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (params.elementLenght <= i) {
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
        bindGroupLayouts: [
          this.processB_BindGroupLayout,
          this.paramsBindGroupLayout,
        ],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "radixSortB",
          code: /* wgsl */ `
struct Params {
  elementLenght: u32,
}

struct Uniforms {
  bitIndex: u32,
}

@group(0) @binding(0) var<storage, read_write> dataWrite: array<u32>; // 並び替え後
@group(0) @binding(1) var<storage, read> dataRead: array<u32>; // 並び替え前
@group(0) @binding(2) var<storage, read> prefixSum: array<u32>;   // 累積和
@group(0) @binding(3) var<storage, read> invertedBit: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let i = globalId.x;
  if (params.elementLenght <= i) {
    return;
  }

  let value = dataRead[i];
  let b = (value >> uniforms.bitIndex) & 1u;

  let f = prefixSum[i] - invertedBit[i];

  let lastIndex = params.elementLenght - 1u;
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

    // offsetを計算
    this.makeOffset_Pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [
          this.makeOffset_BindGroupLayout,
          this.paramsBindGroupLayout,
        ],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "makeOffset",
          code: /* wgsl */ `
struct Params {
  elementLenght: u32,
}

@group(0) @binding(0) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(1) var<storage, read> sortedKeys: array<u32>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let i = globalId.x;
  if (params.elementLenght <= i) {
    return;
  }

  let key = sortedKeys[i];

  // 先頭要素、または前の要素とキーが違う場所が「そのキーの開始位置」
  if (i == 0u || sortedKeys[i - 1u] != key) {
    offsets[key] = i;
  }
}
          `,
        }),
        entryPoint: "main",
      },
    });

    const size = this.maxLenght * 4;
    const offset = this.maxNum * 4;

    this.ping = this.device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.pong = this.device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.prefixSum = this.device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.invertedBit = this.device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.offset = this.device.createBuffer({
      size: offset,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    this.params = this.device.createBuffer({
      size: 4 * 4,
      usage:
        GPUBufferUsage.UNIFORM |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    this.processA_PingToPongBindGroup = this.device.createBindGroup({
      layout: this.processA_BindGroupLayout,
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
            buffer: this.bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processA_PongToPingBindGroup = this.device.createBindGroup({
      layout: this.processA_BindGroupLayout,
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
            buffer: this.bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processB_PingToPongBindGroup = this.device.createBindGroup({
      layout: this.processB_BindGroupLayout,
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
            buffer: this.bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processB_PongToPingBindGroup = this.device.createBindGroup({
      layout: this.processB_BindGroupLayout,
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
            buffer: this.bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.makeOffsetBindGroup = this.device.createBindGroup({
      layout: this.makeOffset_BindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.offset,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.ping,
          },
        },
      ],
    });
    this.paramsBindGroup = this.device.createBindGroup({
      layout: this.paramsBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.params,
          },
        },
      ],
    });
  }

  dispatch(encoder, computePass, src, dst = src) {
    const stride = this.device.limits.minUniformBufferOffsetAlignment; // 通常 256

    computePass.end();

    const isResultIsPing = this.maxBitIndex & 1;
    console.log(this.maxBitIndex & 1);

    encoder.copyBufferToBuffer(
      src,
      0,
      isResultIsPing === 0 ? this.ping : this.pong,
      0,
      src.size,
    );

    let radixSortPass = encoder.beginComputePass();
    const length = src.size / 4;
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([length]));

    for (let bitIndex = 0; bitIndex < this.maxBitIndex; ++bitIndex) {
      let isPingToPong = ((bitIndex + isResultIsPing) & 1) === 0;
      radixSortPass.setPipeline(this.processA_Pipeline);
      radixSortPass.setBindGroup(
        0,
        isPingToPong
          ? this.processA_PingToPongBindGroup
          : this.processA_PongToPingBindGroup,
        [bitIndex * stride],
      );
      radixSortPass.setBindGroup(1, this.paramsBindGroup);
      radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));

      radixSortPass = this.hillisSteeleScanGPU.dispatch(
        encoder,
        radixSortPass,
        this.invertedBit,
        this.prefixSum,
      );

      radixSortPass.setPipeline(this.processB_Pipeline);
      radixSortPass.setBindGroup(
        0,
        isPingToPong
          ? this.processB_PingToPongBindGroup
          : this.processB_PongToPingBindGroup,
        [bitIndex * stride],
      );
      radixSortPass.setBindGroup(1, this.paramsBindGroup);
      radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));
    }

    radixSortPass.setPipeline(this.makeOffset_Pipeline);
    radixSortPass.setBindGroup(0, this.makeOffsetBindGroup);
    radixSortPass.setBindGroup(1, this.paramsBindGroup);
    radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));

    radixSortPass.end();
    encoder.copyBufferToBuffer(this.ping, 0, dst, 0, src.size);
    return encoder.beginComputePass();
  }
}
