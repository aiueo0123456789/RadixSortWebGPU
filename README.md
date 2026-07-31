# RadixSortWebGPU

```
~
const radixSortGPU = new RadixSortGPU(device);

const commandEncoder = device.createCommandEncoder();
let computePass = commandEncoder.beginComputePass();
computePass = radixSortGPU.dispatch(commandEncoder, computePass, buffer);
computePass.end();
device.queue.submit([commandEncoder.finish()]);
```

radixSortGPU.dispatchは関数内でパスが一度終了するため新しいパスの参照が返されます

## RadixSortの参考

https://qiita.com/tommyecguitar/items/3c1897bceda4a06beef2

## 累積和の参考

https://zenn.dev/yayo1/articles/1273ec6ac3bc17#%E7%B5%82%E3%82%8F%E3%82%8A%E3%81%AB%E8%AB%B8%E3%80%85
