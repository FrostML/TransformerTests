import paddle
import paddle.nn.functional as F

from paddle import inference

import numpy as np
import time
import gc
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-size",
        default=1024,
        type=int,
        help="The hidden_size of a model. "
    )
    parser.add_argument(
        "--ffn-inter-size",
        default=None,
        type=int,
        help="The intermediate size in FFN. "
    )
    parser.add_argument(
        "--hidden-act",
        default="relu",
        type=str,
        help="The activation in FFN. "
    )
    parser.add_argument(
        "--head-num",
        default=16,
        type=int,
        help="The head num in Transformer. "
    )
    parser.add_argument(
        "--input-dtype",
        default="float32",
        type=str,
        help="The data type of inputs. "
    )
    parser.add_argument(
        "--max-sequence-length",
        default=128,
        type=int,
        help="The max output sequence length. "
    )
    parser.add_argument(
        "--input-sequence-length",
        default=128,
        type=int,
        help="The input sequence length. Used for cache in generation. "
    )
    parser.add_argument(
        "--fused-qkv",
        action="store_true",
        help="QKV fused. ",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        default=1,
        type=int,
        help="The parallel size used for tensor parallel. "
    )
    parser.add_argument(
        "--inference-model-dir",
        default="./infer_model/",
        type=str,
        help="The inference model dir. "
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to use. "
    )
    parser.add_argument(
        "--use-mkl",
        action="store_true",
        help="Whether to use mkl to process inference. ",
    )
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Threads used in MKL. "
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Batch size. "
    )

    args = parser.parse_args()
    return args


class QKV(paddle.nn.Layer):
    def __init__(self, hidden_size, qkv_output_size, fused_qkv=False):
        super().__init__()
        self.qkv_linear = paddle.nn.Linear(hidden_size, qkv_output_size)
        self.fused_qkv = fused_qkv

    def forward(self, hidden_state):
        if self.fused_qkv:
            return self.qkv_linear(hidden_state)
        else:
            query = self.qkv_linear(hidden_state)
            key = self.qkv_linear(hidden_state)
            value = self.qkv_linear(hidden_state)
            return query, key, value

    def get_input_spec(self, args):
        return [
                    paddle.static.InputSpec(
                        shape=[args.batch_size, args.input_sequence_length, args.hidden_size],
                        dtype=args.input_dtype,
                    )
               ]

class AttentionScoreQK(paddle.nn.Layer):
    def __init__(self, is_decoder=False, output_sequence_length=None):
        super().__init__()
        self.is_decoder = is_decoder
        self.output_sequence_length = output_sequence_length
        if is_decoder and output_sequence_length is None:
            raise ValueError("Invalid decoder parameters. ")

    def forward(self, query, key):
        if not self.is_decoder:
            return paddle.matmul(query, key, transpose_y=True)
        else:
            key_l1 = key
            for i in range(self.output_sequence_length):
                key = paddle.concat([key, key_l1], axis=-2)
                hidden_state = paddle.matmul(query, key, transpose_y=True)
            return hidden_state
    
    def get_input_spec(self, args):
        return [
                    paddle.static.InputSpec(
                        shape=[args.batch_size, 1 if self.is_decoder else args.input_sequence_length, args.hidden_size],
                        dtype=args.input_dtype,
                    ),
                    paddle.static.InputSpec(
                        shape=[args.batch_size, 1 if self.is_decoder else args.input_sequence_length, args.hidden_size],
                        dtype=args.input_dtype,
                    ),
               ]


class AttentionScoreV(paddle.nn.Layer):
    def __init__(self, is_decoder=False, output_sequence_length=None):
        super().__init__()
        self.is_decoder = is_decoder
        self.output_sequence_length = output_sequence_length
        if is_decoder and output_sequence_length is None:
            raise ValueError("Invalid decoder parameters. ")

    def forward(self, attention_weight, value):
        if not self.is_decoder:
            return paddle.matmul(attention_weight, value)
        else:
            weight_l1 = attention_weight
            value_l1 = value
            for i in range(self.output_sequence_length):
                attention_weight = paddle.concat([attention_weight, weight_l1], axis=-1)
                value = paddle.concat([value, value_l1], axis=-2)
                hidden_state = paddle.matmul(attention_weight, value)
            return hidden_state

    def get_input_spec(self, args):
        return [
                    paddle.static.InputSpec(
                        shape=[args.batch_size, 1 if self.is_decoder else args.input_sequence_length, args.hidden_size],
                        dtype=args.input_dtype,
                    ),
                    paddle.static.InputSpec(
                        shape=[args.batch_size, 1 if self.is_decoder else args.input_sequence_length, args.hidden_size],
                        dtype=args.input_dtype,
                    ),
               ]


class FeedFoward(paddle.nn.Layer):
    def __init__(self, hidden_size, ffn_inter_size, hidden_act):
        super().__init__()
        self.ffn_inter_linear = paddle.nn.Linear(hidden_size, ffn_inter_size)
        self.ffn_out_linear = paddle.nn.Linear(ffn_inter_size, hidden_size)
        self.hidden_act = self.activation_converter(hidden_act)
        self.is_gated = ("gated" in hidden_act)
        if self.is_gated:
            self.ffn_inter_linear1 = paddle.nn.Linear(hidden_size, ffn_inter_size)

    def activation_converter(self, hidden_act):
        if hidden_act in ["relu", "gated-relu"]:
            return F.relu
        elif hidden_act in ["gelu", "gated-gelu"]:
            return F.gelu
        elif hidden_act in ["silu", "gated-silu"]:
            return F.silu
        else:
            raise ValueError("{} is not supported. ".format(hidden_act))

    def forward(self, hidden_state):
        ffn_inter = self.hidden_act(self.ffn_inter_linear(hidden_state))
        if self.is_gated:
            ffn_inter1 = self.ffn_inter_linear1(hidden_act)
            ffn_inter = ffn_inter * ffn_inter1

        return self.ffn_out_linear(ffn_inter)

    def get_input_spec(self, args):
        return [
                    paddle.static.InputSpec(
                        shape=[args.batch_size, args.input_sequence_length, args.hidden_size],
                        dtype=args.input_dtype,
                    )
               ]


def export_model(args, module, module_name, is_decoder=False):
    module = paddle.jit.to_static(
        module,
        input_spec=module.get_input_spec(args),
    )
    paddle.jit.save(module, os.path.join(
        args.inference_model_dir,
        module_name + "_dec" if is_decoder else module_name,
        module_name))


def export(args):
    if args.ffn_inter_size is None:
        args.ffn_inter_size = args.hidden_size * 4

    args.qkv_hidden_size = args.hidden_size
    if args.tensor_parallel_size > 1:
        args.qkv_hidden_size /= args.tensor_parallel_size
        args.ffn_inter_size /= args.tensor_parallel_size

    if args.fused_qkv:
        args.qkv_hidden_size *= 3

    qkv_module = QKV(args.hidden_size, args.qkv_hidden_size, args.fused_qkv)
    attentionscores_qk = AttentionScoreQK()
    attentionscores_v = AttentionScoreV()
    ffn = FeedFoward(args.hidden_size, args.ffn_inter_size, args.hidden_act)

    export_model(args, qkv_module, "qkv_module")
    export_model(args, attentionscores_qk, "attentionscores_qk")
    export_model(args, attentionscores_v, "attentionscores_v")
    export_model(args, ffn, "ffn")

    attentionscores_qk_dec = AttentionScoreQK(is_decoder=True, output_sequence_length=args.max_sequence_length)
    attentionscores_v_dec = AttentionScoreV(is_decoder=True, output_sequence_length=args.max_sequence_length)

    export_model(args, attentionscores_qk_dec, "attentionscores_qk", is_decoder=True)
    export_model(args, attentionscores_v_dec, "attentionscores_v", is_decoder=True)


def get_fake_input(args, module, length_q, length_kv):
    input_dtype = "uint16" if args.input_dtype == "bfloat16" else args.input_dtype
    if "qkv" in module:
        return [np.random.randn(
                    args.batch_size, length_q, args.hidden_size).astype(input_dtype)]
    elif "scores_qk" in module:
        return [np.random.randn(
                    args.batch_size, args.head_num, length_q, args.hidden_size // args.head_num).astype(input_dtype),
                np.random.randn(
                    args.batch_size, args.head_num, length_kv, args.hidden_size // args.head_num).astype(input_dtype),
               ]
    elif "scores_v" in module:
        return [np.random.randn(
                    args.batch_size, args.head_num, length_q, length_kv).astype(input_dtype),
                np.random.randn(
                    args.batch_size, args.head_num, length_kv, args.hidden_size // args.head_num).astype(input_dtype),
               ]
    elif "ffn" in module:
        return [np.random.randn(
                    args.batch_size, length_q, args.hidden_size).astype(input_dtype)]
    else:
        raise ValueError("Not support module {}. ".format(module))


def infer(args, module_name, is_decoder=False):
    if "scores" in module_name:
        os.rename(os.path.join(
                    args.inference_model_dir,
                    module_name + "_dec" if is_decoder else module_name,
                    module_name + ".pdmodel"),
                  os.path.join(
                    args.inference_model_dir,
                    module_name + "_dec" if is_decoder else module_name,
                    "__model__"),
                 )
        config = inference.Config(
            os.path.join(
                args.inference_model_dir,
                module_name + "_dec" if is_decoder else module_name),
        )
    else:
        config = inference.Config(
            os.path.join(
                args.inference_model_dir,
                module_name + "_dec" if is_decoder else module_name,
                module_name + ".pdmodel"),
            os.path.join(
                args.inference_model_dir,
                module_name + "_dec" if is_decoder else module_name,
                module_name + ".pdiparams"),
        )
    if args.device == "gpu":
        config.enable_use_gpu(100, 0)
    elif args.device == "xpu":
        config.enable_xpu()
    elif args.device == "npu":
        config.enable_npu()
    else:
        # CPU
        config.disable_gpu()
        if args.use_mkl:
            config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(args.threads)

    config.switch_use_feed_fetch_ops(False)
    predictor = inference.create_predictor(config)

    input_handles = [predictor.get_input_handle(name) for name in predictor.get_input_names()]
    output_handles = [predictor.get_output_handle(name) for name in predictor.get_output_names()]

    inputs = get_fake_input(args, module_name, 1 if is_decoder else args.input_sequence_length, args.input_sequence_length)

    for input_handle, hidden_state in zip(input_handles, inputs):
        input_handle.copy_from_cpu(hidden_state)

    for i in range(1000):
        if 100 == i:
            paddle.device.cuda.synchronize()
            start = time.perf_counter()
        predictor.run()

    paddle.device.cuda.synchronize()
    duration = (time.perf_counter() - start) / 900 * 1000

    predictor.try_shrink_memory()
    del predictor
    gc.collect()

    return duration


def performance(args):
    qkv_duration = infer(args, "qkv_module")
    ac_qk_duration = infer(args, "attentionscores_qk")
    ac_v_duration = infer(args, "attentionscores_v")
    ffn_duration = infer(args, "ffn")

    ac_qk_dec_duration = infer(args, "attentionscores_qk", True)
    ac_v_dec_duration = infer(args, "attentionscores_v", True)

    print("Performance: ")
    print("Batch size: {}, input_sequence_length: {}, max_sequence_length: {}.".format(args.batch_size, args.input_sequence_length, args.max_sequence_length))

    print("-" * 20)
    print("The time of QKV with fused_qkv {} is: \t{}.".format("True" if args.fused_qkv else "False", qkv_duration))
    print("The time of q*k (without softmax) is: \t\t{}.".format(ac_qk_duration))
    print("The time of qk_weight * v is: \t\t\t{}.".format(ac_v_duration))
    print("The time of ffn with hidden_act {} is: \t{}.".format(args.hidden_act, ffn_duration))

    print("The time of q*cached_k(sequence_length from 0 to {}) (without softmax) in decoding is:\t{}.".format(args.max_sequence_length, ac_qk_dec_duration))
    print("The time of qk_weight*cached_v(sequence_length from 0 to {}) in decoding is: \t\t{}. ".format(args.max_sequence_length, ac_v_dec_duration))


if __name__ == "__main__":
    args = parse_args()

    paddle.set_default_dtype(args.input_dtype)

    export(args)
    performance(args)
