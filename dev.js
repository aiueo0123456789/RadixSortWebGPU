export async function printBufferData(
  device,
  buffer,
  struct,
  text = "",
  open = false,
) {
  // 一時的な読み取り用バッファを作成 (MAP_READ を含む)
  const readBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // コピーコマンドを発行
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
  const commandBuffer = commandEncoder.finish();
  device.queue.submit([commandBuffer]);

  // 一時バッファの内容をマップして表示
  await readBuffer.mapAsync(GPUMapMode.READ);
  const mappedRange = readBuffer.getMappedRange();
  const rawData = new Uint8Array(mappedRange);

  // 構造体に基づいてデータを解析
  const dataView = new DataView(rawData.buffer);
  const structSize = struct.length * 4; // 各フィールドのサイズが 4 バイト固定 (u32, f32)
  const result = [];

  let offset = 0;
  for (let i = 0; i < buffer.size / structSize; i++) {
    const keep = [];
    for (const field of struct) {
      if (field === "u32") {
        keep.push(dataView.getUint32(offset, true));
      } else if (field === "f32") {
        keep.push(dataView.getFloat32(offset, true));
      } else if (field == "bit") {
      }
      offset += 4; // フィールドのサイズを加算
    }
    result.push(keep);
  }

  readBuffer.unmap();
  if (open) console.log(text, ...result);
  else console.log(text, result);
}
