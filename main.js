import { printBufferData } from "./dev.js";
import { RadixSortGPU } from "./RadixSortGPU.js";

const adapter = await navigator.gpu.requestAdapter();

const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBuffersPerShaderStage: 10,
  },
});

let num = 10 ** 3;
const radixSortGPU = new RadixSortGPU(device);
const buffer = device.createBuffer({
  size: num * 4,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

device.queue.writeBuffer(
  buffer,
  0,
  new Uint32Array(
    Array.from({ length: num }).map((x, i) => Math.round(Math.random() * 900)),
  ),
);
// printBufferData(device, buffer, ["u32"], "", true);

const commandEncoder = device.createCommandEncoder();
let computePass = commandEncoder.beginComputePass();
computePass = radixSortGPU.dispatch(commandEncoder, computePass, buffer);
computePass.end();
device.queue.submit([commandEncoder.finish()]);

// printBufferData(device, buffer, ["u32"], "", true);
// printBufferData(device, hillisSteeleScanGPU.pingPongMap.get(buffer.size).invertedBit, ["u32"], "invertedBit", true);
// printBufferData(
//   device,
//   hillisSteeleScanGPU.pingPongMap.get(buffer.size).prefixSum,
//   ["u32"],
//   "prefixSum",
//   true,
// );
// printBufferData(
//   device,
//   hillisSteeleScanGPU.pingPongMap.get(buffer.size).ping,
//   ["u32"],
//   "ping",
//   true,
// );
// printBufferData(
//   device,
//   hillisSteeleScanGPU.pingPongMap.get(buffer.size).pong,
//   ["u32"],
//   "pong",
//   true,
// );
