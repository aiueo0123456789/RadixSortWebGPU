export class RadixSortGPU {
  constructor(device) {
    this.device = device;

    this._init();
  }

  _init() {
    this.maxBitIndex = 16; // 16ビット比較

    this.maxLength = 10 ** 6;
    this.maxNum = 2 ** this.maxBitIndex;

    const stride = this.device.limits.minUniformBufferOffsetAlignment;

    const bitIndices = new Uint32Array((stride / 4) * this.maxBitIndex);

    this.bitIndicesBuffer = this.device.createBuffer({
      size: stride * this.maxBitIndex,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    for (let p = 0; p < this.maxBitIndex; p++) {
      bitIndices[p * (stride / 4)] = p;
    }
    this.device.queue.writeBuffer(this.bitIndicesBuffer, 0, bitIndices);

    const maxPasses = Math.ceil(Math.log2(this.maxLength));
    const steps = new Uint32Array((stride / 4) * maxPasses);

    this.stepsBuffer = this.device.createBuffer({
      size: stride * maxPasses,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    for (let p = 0; p < maxPasses; p++) {
      steps[p * (stride / 4)] = 2 ** p;
    }
    this.device.queue.writeBuffer(this.stepsBuffer, 0, steps);

    const processA_BindGroupLayout = this.device.createBindGroupLayout({
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

    const processB_BindGroupLayout = this.device.createBindGroupLayout({
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

    const makeOffset_BindGroupLayout = this.device.createBindGroupLayout({
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

    const paramsBindGroupLayout = this.device.createBindGroupLayout({
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

    const scanGroupLayout = this.device.createBindGroupLayout({
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
            hasDynamicOffset: true, // これが必須
            minBindingSize: 16,
          },
        },
      ],
    });

    // ビット反転
    this.processA_Pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [processA_BindGroupLayout, paramsBindGroupLayout],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "radixSortA",
          code: /* wgsl */ `
struct Params {
  elementLength: u32,
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
  if (params.elementLength <= i) {
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
        bindGroupLayouts: [processB_BindGroupLayout, paramsBindGroupLayout],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "radixSortB",
          code: /* wgsl */ `
struct Params {
  elementLength: u32,
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
  if (params.elementLength <= i) {
    return;
  }

  let value = dataRead[i];
  let b = (value >> uniforms.bitIndex) & 1u;

  let f = prefixSum[i] - invertedBit[i];

  let lastIndex = params.elementLength - 1u;
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
        bindGroupLayouts: [makeOffset_BindGroupLayout, paramsBindGroupLayout],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "makeOffset",
          code: /* wgsl */ `
struct Params {
  elementLength: u32,
}

@group(0) @binding(0) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(1) var<storage, read> sortedKeys: array<u32>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let i = globalId.x;
  if (params.elementLength <= i) {
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

    // 累積和の計算
    this.scanPipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [scanGroupLayout, paramsBindGroupLayout],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "scanPipeline",
          code: /* wgsl */ `
struct Params {
  elementLength: u32,
}

struct Uniforms {
  step: u32,
};

@group(0) @binding(0) var<storage, read_write> dst: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<uniform> uni: Uniforms;
@group(1) @binding(0) var<uniform> params: Params;

struct CInput {
  @builtin(global_invocation_id) gid: vec3<u32>
}

@compute @workgroup_size(64)
fn main(input: CInput) {
  let i = input.gid.x;
  if (i >= params.elementLength) {
    return;
  }

  if (i < uni.step) {
    dst[i] = src[i];
  } else {
    dst[i] = src[i] + src[i - uni.step];
  }
}
          `,
        }),
        entryPoint: "main",
      },
    });

    const size = this.maxLength * 4;
    const offsetSize = this.maxNum * 4;

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
    this.invertedBit = this.device.createBuffer({
      size: size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.offset = this.device.createBuffer({
      size: offsetSize,
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
    this.scanPong = this.device.createBuffer({
      size: size,
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
            buffer: this.bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processA_PongToPingBindGroup = this.device.createBindGroup({
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
            buffer: this.bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processB_PingToPongBindGroup = this.device.createBindGroup({
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
            buffer: this.bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.processB_PongToPingBindGroup = this.device.createBindGroup({
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
            buffer: this.bitIndicesBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.makeOffsetBindGroup = this.device.createBindGroup({
      layout: makeOffset_BindGroupLayout,
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
      layout: paramsBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.params,
          },
        },
      ],
    });

    this.scanPingToPongBindGroup = this.device.createBindGroup({
      layout: scanGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.scanPong,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.prefixSum,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.stepsBuffer,
            size: 4 * 4,
          },
        },
      ],
    });
    this.scanPongToPingBindGroup = this.device.createBindGroup({
      layout: scanGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.prefixSum,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.scanPong,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.stepsBuffer,
            size: 4 * 4,
          },
        },
      ],
    });

    this.scanFirstPingBindGroup = this.device.createBindGroup({
      layout: scanGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.prefixSum,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.invertedBit,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.stepsBuffer,
            size: 4 * 4,
          },
        },
      ],
    });
    this.scanFirstPongBindGroup = this.device.createBindGroup({
      layout: scanGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.scanPong,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.invertedBit,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.stepsBuffer,
            size: 4 * 4,
          },
        },
      ],
    });
  }

  dispatch(encoder, computePass, src, dst = src) {
    const stride = this.device.limits.minUniformBufferOffsetAlignment; // 通常 256

    computePass.end();

    const isResultIsPing = this.maxBitIndex & 1;

    encoder.copyBufferToBuffer(
      src,
      0,
      isResultIsPing === 0 ? this.ping : this.pong,
      0,
      src.size,
    );

    const length = src.size / 4;
    const prefixSumLoopNum = Math.ceil(Math.log2(length)); // 累積和の計算に必要なディスパッチ回数
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([length]));

    const radixSortPass = encoder.beginComputePass();

    radixSortPass.setBindGroup(1, this.paramsBindGroup);
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
      radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));

      radixSortPass.setPipeline(this.scanPipeline);

      const isScanResultIsPing = prefixSumLoopNum & 1;
      radixSortPass.setBindGroup(
        0,
        isScanResultIsPing === 0
          ? this.scanFirstPongBindGroup
          : this.scanFirstPingBindGroup,
        [0],
      );
      radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));
      for (let p = 1; p < prefixSumLoopNum; p++) {
        radixSortPass.setBindGroup(
          0,
          ((p + isScanResultIsPing) & 1) === 0
            ? this.scanPingToPongBindGroup
            : this.scanPongToPingBindGroup,
          [p * stride],
        );
        radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));
      }

      radixSortPass.setPipeline(this.processB_Pipeline);
      radixSortPass.setBindGroup(
        0,
        isPingToPong
          ? this.processB_PingToPongBindGroup
          : this.processB_PongToPingBindGroup,
        [bitIndex * stride],
      );
      radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));
    }

    radixSortPass.setPipeline(this.makeOffset_Pipeline);
    radixSortPass.setBindGroup(0, this.makeOffsetBindGroup);
    radixSortPass.dispatchWorkgroups(Math.ceil(length / 64));

    radixSortPass.end();
    encoder.copyBufferToBuffer(this.ping, 0, dst, 0, src.size);
    return encoder.beginComputePass();
  }
}
