/**
 * 累積和を求める
 */
class PingPong {
  constructor(device, bindGroupLayout, stepsBuffer, size) {
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

    this.pingToPongBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
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
            buffer: stepsBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.pongToPingBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
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
            buffer: stepsBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
  }
}

export class HillisSteeleScanGPU {
  constructor(device) {
    /** @type {Map<number, PingPong>} */
    this.pingPongMap = new Map();

    this.device = device;

    this._init();
  }

  _init() {
    this.maxLength = 10 ** 6; // 要素数 10 ** 6 まで処理できる

    const maxPasses = Math.ceil(Math.log2(this.maxLength));
    const stride = this.device.limits.minUniformBufferOffsetAlignment;
    const params = new Uint32Array((stride / 4) * maxPasses);

    this.stepsBuffer = this.device.createBuffer({
      size: stride * maxPasses,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    for (let p = 0; p < maxPasses; p++) {
      params[p * (stride / 4)] = 2 ** p;
    }
    this.device.queue.writeBuffer(this.stepsBuffer, 0, params);

    this.bindGroupLayout = this.device.createBindGroupLayout({
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

    this.scanPipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "scanPipeline",
          code: /* wgsl */ `
struct Uniforms {
  step: u32,
};

@group(0) @binding(0) var<storage, read_write> dst: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<uniform> uni: Uniforms;

struct CInput {
  @builtin(global_invocation_id) gid: vec3<u32>
}

@compute @workgroup_size(64)
fn main(input: CInput) {
  let total = arrayLength(&src);

  let i = input.gid.x;

  if (i >= total) {
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

    this.pingToPongBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
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
            buffer: stepsBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
    this.pongToPingBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
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
            buffer: stepsBuffer,
            offset: 0,
            size: 4 * 4,
          },
        },
      ],
    });
  }

  dispatch(encoder, computePass, src, dst = src) {
    const stride = this.device.limits.minUniformBufferOffsetAlignment; // 通常 256

    computePass.end();

    encoder.copyBufferToBuffer(src, 0, this.ping, 0, src.size);

    const scanPass = encoder.beginComputePass();
    scanPass.setPipeline(this.scanPipeline);
    const length = src.size / 4;
    if (this.maxLength < length) {
      console.error("計算しきれない可能性があります");
    }
    const passes = Math.ceil(Math.log2(length));
    for (let p = 0; p < passes; p++) {
      scanPass.setBindGroup(
        0,
        (p & 1) === 0
          ? this.pingToPongBindGroup
          : this.pongToPingBindGroup,
        [p * stride], // uniのoffset指定
      );
      scanPass.dispatchWorkgroups(Math.ceil(length / 64));
    }
    scanPass.end();
    encoder.copyBufferToBuffer(
      (passes & 1) === 0 ? this.ping : this.pong,
      0,
      dst,
      0,
      src.size,
    );
    return encoder.beginComputePass();
  }
}
