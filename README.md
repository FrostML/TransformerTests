# TransformerTests

用于 Transformer 基本组网结构矩阵乘性能测试。

### 使用方法

``` bash
python performance.py
```

参数配置如下：

* `--hidden-size`：模型隐层大小。默认为 1024。
* `--ffn-inter-size`：FFN 中间隐层大小。默认为 None，表示使用 `hidden_size * 4` 作为其值。
* `--hidden-act`：FFN 中间激活函数类型也包括 FFN 组网类型。可设置为 `relu`，`gelu`，`silu`，`gated-relu`，`gated-gelu`，`gated-silu`。
* `--head-num`：多头注意力机制中头的数目。
* `--input-dtype`：输入的数据类型。目前支持 `bfloat16`，`float16`，`float32`，`float64`。不支持 `int8`。
* `--max-sequence-length`：生成使用的最长输出长度。
* `--input-sequence-length`：模拟的输入的长度。
* `--fused-qkv`：是否将 qkv 的矩阵乘融合。
* `--tensor-parallel-size`：tensor 并行的大小。不会跑多卡，会将设定的模型超参 size 进行切分后，在单卡上进行测试。
* `--inference-model-dir`：输出 inference 模型的位置，一般可以不用设置。
* `--device`：执行性能测试用的设备，即 inference config 所指定的设备。
* `--use-mkl`：Inference 是否开启 MKL。
* `--threads`：Inference 设置线程数。
* `--batch-size`：性能测试使用的 batch size 大小。

性能数据输出结果为：

``` bash
Performance:
Batch size: 1, input_sequence_length: 128, max_sequence_length: 128.
--------------------
The time of QKV with fused_qkv False is: 	0.13480520186324915.
The time of q*k (without softmax) is: 		0.023315586149692535.
The time of qk_weight * v is: 			0.023320985751019582.
The time of ffn with hidden_act relu is: 	0.20547565590176317.
The time of q*cached_k(sequence_length from 0 to 128) (without softmax) in decoding is: 11.440430050715804.
The time of qk_weight*cached_v(sequence_length from 0 to 128) in decoding is: 		11.993675859024126.
```
