#!/usr/bin/env python3
"""Matched throughput and coherence gate for Gemma-4 OpenVINO artifacts."""

import argparse
import gc
import hashlib
import json
import os
import re
import time

import numpy as np
import openvino as ov
from openvino import Op
import openvino_genai as og


TYPE_RE = re.compile(r"GemmaMoEA4W4Group_K(\d+)_N(\d+)x(\d+)")
CONFIG_NAME = "custom_layers_gemma26_moe_a4w4_v1.xml"
REGISTRY = {}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default=(
            "/home/wondernutts/models/heretics/"
            "gemma-4-26B-A4B-heretic-int4-ov-lut131k"
        ),
    )
    parser.add_argument("--device", default="GPU.2")
    parser.add_argument("--dqgs", type=int, default=0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--gpu-profile-dir", default="")
    parser.add_argument(
        "--pa-flashattn-v2",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control the Intel GPU SDPA FlashAttention-v2 implementation.",
    )
    parser.add_argument("--cb-cache-gb", type=int, default=0)
    parser.add_argument("--max-batched-tokens", type=int, default=0)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument(
        "--moe-grouped-prefill",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control the Intel GPU grouped-GEMM MoE prefill implementation.",
    )
    parser.add_argument("--targets", type=int, nargs="*", default=[512, 2048, 6144])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--coherence-tokens", type=int, default=384)
    parser.add_argument("--coherence-target", type=int, default=0)
    parser.add_argument("--shape-warm", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--skip-coherence", action="store_true")
    parser.add_argument("--no-coherence-warm", action="store_true")
    return parser.parse_args()


def token_ids(encoded):
    return np.asarray(encoded.input_ids.data, dtype=np.int64).reshape(-1)


def encode_ids(tokenizer, text):
    return token_ids(tokenizer.encode(text, add_special_tokens=False))


def native_prompt(content):
    return (
        "<bos><|turn>user\n" + content + "<turn|>\n<|turn>model\n"
        "<|channel>thought\n<channel|>"
    )


def exact_prompt(tokenizer, target, label):
    filler = (
        "The archivist recorded the weather, repaired the northern road, counted "
        "the lanterns, and compared every observation with the previous ledger. "
    )
    payload = ("Performance case %s. " % label) + filler * (target // 8 + 40)
    payload_ids = encode_ids(tokenizer, payload)
    best = None
    for cut in range(max(1, target - 80), target + 20):
        content = tokenizer.decode(payload_ids[:cut], skip_special_tokens=True)
        candidate = native_prompt(content)
        actual = encode_ids(tokenizer, candidate).size
        error = abs(actual - target)
        if best is None or error < best[0]:
            best = (error, candidate, actual)
        if actual == target:
            return candidate
    print(
        "TOKEN_TARGET_WARNING target=%d actual=%d" % (target, best[2]),
        flush=True,
    )
    return best[1]


def generation_config(tokens, force_length=True):
    config = og.GenerationConfig()
    config.max_new_tokens = tokens
    config.do_sample = False
    config.repetition_penalty = 1.2
    config.apply_chat_template = False
    if force_length:
        config.min_new_tokens = tokens
        config.ignore_eos = True
    return config


def make_group_extension(type_name):
    match = TYPE_RE.fullmatch(type_name)
    if not match:
        raise RuntimeError("unsupported Gemma A4W4 extension type: " + type_name)
    kdim = int(match.group(1))
    widths = (int(match.group(2)), int(match.group(3)))

    def init(self, *inputs):
        Op.__init__(self, self)
        if inputs:
            self.set_arguments(list(inputs))
            self.constructor_validate_and_infer_types()

    def validate(self):
        activation = list(self.get_input_partial_shape(0))
        experts, rows = activation[0], activation[1]
        output_type = self.get_input_element_type(0)
        self.set_output_size(2)
        for index, width in enumerate(widths):
            self.set_output_type(index, output_type, ov.PartialShape([experts, rows, width]))

    def clone(self, new_inputs):
        return REGISTRY[type_name](*new_inputs)

    cls = type(
        type_name,
        (Op,),
        {
            "__init__": init,
            "validate_and_infer_types": validate,
            "clone_with_new_inputs": clone,
            "visit_attributes": lambda self, visitor: True,
            "get_type_info": lambda self: ov.DiscreteTypeInfo(type_name, "extension"),
        },
    )
    REGISTRY[type_name] = cls
    return ov.OpExtension(cls)


def custom_properties(model_dir):
    config_path = os.path.join(model_dir, CONFIG_NAME)
    if not os.path.isfile(config_path):
        return {}
    model_xml = os.path.join(model_dir, "openvino_language_model.xml")
    with open(model_xml, "r", encoding="utf-8") as handle:
        type_names = sorted(set(re.findall(
            r'type="(GemmaMoEA4W4Group_K\d+_N\d+x\d+)" version="extension"',
            handle.read(),
        )))
    if not type_names:
        raise RuntimeError("custom config exists but no Gemma A4W4 extension was found")
    return {
        "CONFIG_FILE": config_path,
        "extensions": [make_group_extension(name) for name in type_names],
    }


def result_text(result, tokenizer):
    if hasattr(result, "texts"):
        return result.texts[0]
    return tokenizer.decode(result.tokens[0], skip_special_tokens=True)


def run_case(pipe, tokenizer, name, prompt, tokens, force_length=True):
    started = time.perf_counter()
    result = pipe.generate(
        prompt,
        generation_config=generation_config(tokens, force_length),
    )
    wall = time.perf_counter() - started
    metrics = result.perf_metrics
    input_tokens = int(metrics.get_num_input_tokens())
    output_tokens = int(metrics.get_num_generated_tokens())
    ttft_ms = float(metrics.get_ttft().mean)
    tpot_ms = float(metrics.get_tpot().mean)
    text = result_text(result, tokenizer)
    record = {
        "name": name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": round(ttft_ms, 3),
        "pp_tps": round(input_tokens * 1000.0 / ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3),
        "decode_tps": round(float(metrics.get_throughput().mean), 3),
        "wall_s": round(wall, 3),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    print("RESULT " + json.dumps(record, sort_keys=True), flush=True)
    del result
    gc.collect()
    return record, text


def coherence_content():
    needle_1 = (
        "Ysolda's strongbox key is hidden beneath the third barrel behind the "
        "Bannered Mare."
    )
    needle_2 = (
        "The Vigor of the Nine requires juniper berries, void salts, a sprig "
        "of frost mirriam, and Honningbrew mead."
    )
    needle_3 = (
        "The password to the Thieves Guild cache beneath the Ratway is "
        "shadow-hearth."
    )
    entries = []
    for index in range(1, 121):
        entries.append(
            "Entry %d. On the %dth of Frostfall I crossed the road near "
            "Whiterun, counted my septims, repaired my armor, and marked the "
            "mountain passes by candlelight. The wind carried pine and snow. "
            "Talos guide the morning." % (index, (index % 27) + 1)
        )
        if index == 5:
            entries.append("Private addendum: " + needle_1)
        elif index == 58:
            entries.append("Recipe written in charcoal: " + needle_2)
        elif index == 112:
            entries.append("Delvin's late warning: " + needle_3)
    question = (
        "Answer these from the chronicle, preserving exact details: 1) Where is "
        "Ysolda's key? 2) What are every ingredient in the Vigor of the Nine? "
        "3) What is the cache password? Then write Entry 121 in the same style "
        "and end it with 'Talos guide the morning.'"
    )
    return (
        "The Chronicle of the Dragonborn\n\n"
        + "\n\n".join(entries)
        + "\n\n"
        + question
    )


def coherence_prompt(tokenizer, target=0):
    content = coherence_content()
    if not target:
        return native_prompt(content)

    marker = "\n\nAnswer these"
    head, tail = content.rsplit(marker, 1)
    suffix = marker + tail
    deficit = max(0, target - encode_ids(tokenizer, native_prompt(content)).size)
    filler = (
        "Supplemental ledger. The archivist checked the western watchtower, "
        "counted each torch, repaired the road, and recorded clear weather. "
    )
    filler_ids = encode_ids(tokenizer, filler * (deficit // 8 + 80))
    best = None
    for cut in range(max(1, deficit - 120), deficit + 120):
        padding = tokenizer.decode(filler_ids[:cut], skip_special_tokens=True)
        candidate = native_prompt(head + "\n\n" + padding + suffix)
        actual = encode_ids(tokenizer, candidate).size
        error = abs(actual - target)
        if best is None or error < best[0]:
            best = (error, candidate, actual)
        if actual == target:
            return candidate
    print(
        "COHERENCE_TARGET_WARNING target=%d actual=%d" % (target, best[2]),
        flush=True,
    )
    return best[1]


def main():
    args = arguments()
    props = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": args.dqgs}
    if args.moe_grouped_prefill != "auto":
        os.environ["MOE_USE_GROUPED_GEMM_PREFILL"] = (
            "1" if args.moe_grouped_prefill == "on" else "0"
        )
    props.update(custom_properties(args.dir))
    if args.gpu_profile_dir:
        profile_dir = os.path.realpath(args.gpu_profile_dir)
        os.makedirs(profile_dir, exist_ok=True)
        debug_config_path = os.path.join(profile_dir, "gpu_debug_config.json")
        gpu_debug_config = {"GPU_DUMP_PROFILING_DATA_PATH": profile_dir}
        if args.pa_flashattn_v2 != "auto":
            gpu_debug_config["GPU_COULD_USE_FLASHATTN_V2"] = (
                "true" if args.pa_flashattn_v2 == "on" else "false"
            )
        with open(debug_config_path, "w", encoding="utf-8") as handle:
            json.dump({"GPU": gpu_debug_config}, handle, indent=2)
        os.environ["OV_GPU_DEBUG_CONFIG"] = debug_config_path
        props[ov.properties.enable_profiling] = True
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        props["CACHE_DIR"] = args.cache_dir
    if args.cb_cache_gb:
        scheduler = og.SchedulerConfig()
        scheduler.cache_size = args.cb_cache_gb
        scheduler.enable_prefix_caching = args.prefix_cache
        if args.max_batched_tokens:
            scheduler.max_num_batched_tokens = args.max_batched_tokens
        props["scheduler_config"] = scheduler
    print(
        "CONFIG device=%s dqgs=%d cb_cache_gb=%d prefix_cache=%s "
        "max_batched_tokens=%d grouped_moe=%s single_stream=yes "
        "custom=%s gpu_profile=%s pa_flashattn_v2=%s model=%s"
        % (
            args.device,
            args.dqgs,
            args.cb_cache_gb,
            "on" if args.prefix_cache else "off",
            args.max_batched_tokens,
            args.moe_grouped_prefill,
            bool(props.get("CONFIG_FILE")),
            args.gpu_profile_dir or "off",
            args.pa_flashattn_v2,
            args.dir,
        ),
        flush=True,
    )
    started = time.perf_counter()
    pipe = og.VLMPipeline(args.dir, args.device, **props)
    print("PIPELINE_BUILD_S=%.3f" % (time.perf_counter() - started), flush=True)
    tokenizer = pipe.get_tokenizer()

    warm = exact_prompt(tokenizer, 64, "jit")
    started = time.perf_counter()
    warm_result = pipe.generate(warm, generation_config=generation_config(8))
    del warm_result
    gc.collect()
    print("WARM_JIT_S=%.3f" % (time.perf_counter() - started), flush=True)

    performance = []
    if not args.skip_performance:
        for target in args.targets:
            if args.shape_warm:
                shape_warm = exact_prompt(tokenizer, target, "ctx%d-warm" % target)
                started = time.perf_counter()
                warm_result = pipe.generate(
                    shape_warm, generation_config=generation_config(1, False)
                )
                del warm_result
                gc.collect()
                print(
                    "SHAPE_WARM target=%d seconds=%.3f"
                    % (target, time.perf_counter() - started),
                    flush=True,
                )
            for repeat in range(1, args.runs + 1):
                prompt = exact_prompt(
                    tokenizer, target, "ctx%d-r%d" % (target, repeat)
                )
                record, _ = run_case(
                    pipe,
                    tokenizer,
                    "ctx%d-r%d" % (target, repeat),
                    prompt,
                    1,
                    force_length=False,
                )
                performance.append(record)

        if args.decode_tokens > 0:
            decode_prompt = native_prompt(
                "Write a long, detailed story about a dragon who learns to code."
            )
            decode_record, _ = run_case(
                pipe,
                tokenizer,
                "decode-short",
                decode_prompt,
                args.decode_tokens,
            )
            performance.append(decode_record)

    coherence = None
    checks = None
    if not args.skip_coherence:
        prompt = coherence_prompt(tokenizer, args.coherence_target)
        if not args.no_coherence_warm:
            started = time.perf_counter()
            warm_result = pipe.generate(
                prompt, generation_config=generation_config(1, False)
            )
            del warm_result
            gc.collect()
            print(
                "COHERENCE_SHAPE_WARM_S=%.3f" % (time.perf_counter() - started),
                flush=True,
            )
        coherence, output = run_case(
            pipe,
            tokenizer,
            "coherence",
            prompt,
            args.coherence_tokens,
            force_length=False,
        )
        tail = output.lower()[-4000:]
        checks = {
            "needle_1": "third barrel" in tail and "bannered mare" in tail,
            "needle_2": all(
                term in tail
                for term in ("juniper", "void salts", "frost mirriam", "honningbrew")
            ),
            "needle_3": "shadow-hearth" in tail,
            "style": "entry 121" in tail and "talos guide the morning" in tail,
        }
        print("COHERENCE_CHECKS " + json.dumps(checks, sort_keys=True), flush=True)
        print("COHERENCE_OUTPUT_BEGIN", flush=True)
        print(output, flush=True)
        print("COHERENCE_OUTPUT_END", flush=True)

    print(
        "SUMMARY "
        + json.dumps(
            {"performance": performance, "coherence": coherence, "checks": checks},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
