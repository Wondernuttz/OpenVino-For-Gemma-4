"""Portable, explicit model profiles. No GPU workload is killed or restarted."""
import glob
import os
from pathlib import Path
import sys

PROFILES = {
    'gemma12-text': {
        'OV_MODEL_NAME': 'gemma-4-12b-heretic', 'OV_DQGS': '128',
        'OV_PREFIX_CACHE': '0', 'OV_MAX_BATCHED_TOKENS': '0',
        'OV_KV_CACHE_PRECISION': '', 'OV_CONTEXT_TOKENS': '4096',
        'OV_MIN_OUTPUT_RESERVE': '512', 'OV_MAX_NEW_TOKENS': '1024',
        'MOE_USE_GROUPED_GEMM_PREFILL': '0', 'MOE_GROUPED_BINARY_LOOKUP': '0',
    },
    'gemma26-b70': {
        'OV_MODEL_NAME': 'gemma-4-26b-a4b', 'OV_DQGS': '128',
        'OV_PREFIX_CACHE': '8', 'OV_MAX_BATCHED_TOKENS': '16384',
        'OV_KV_CACHE_PRECISION': 'u4', 'OV_CONTEXT_TOKENS': '24576',
        'OV_MIN_OUTPUT_RESERVE': '4096', 'OV_MAX_NEW_TOKENS': '4096',
        'MOE_USE_GROUPED_GEMM_PREFILL': '1', 'MOE_GROUPED_BINARY_LOOKUP': '1',
    },
}

def configure(env):
    result = dict(env)
    profile = result.get('OV_PROFILE', 'gemma12-text')
    result['OV_PROFILE'] = profile
    if profile not in PROFILES:
        raise ValueError('OV_PROFILE must be gemma12-text or gemma26-b70')
    for name, value in PROFILES[profile].items():
        result.setdefault(name, value)
    for name, value in {'OV_DEVICE': 'GPU', 'OV_PORT': '8000', 'OV_FMT': 'gemma',
                        'OV_MAX_NUM_SEQS': '1', 'OV_THINK': '0',
                        'OV_NATIVE_VISION': '0'}.items():
        result.setdefault(name, value)
    if not result.get('OV_MODEL'):
        raise ValueError('Set OV_MODEL to the local model directory')
    if int(result['OV_CONTEXT_TOKENS']) <= int(result['OV_MIN_OUTPUT_RESERVE']) + 128:
        raise ValueError('Context must exceed output reserve plus the 128-token margin')
    if profile == 'gemma12-text' and (float(result['OV_PREFIX_CACHE']) != 0 or
                                    result['MOE_GROUPED_BINARY_LOOKUP'] != '0'):
        raise ValueError('12B profile requires plain pipeline, not the 26B MoE/cache path')
    if int(result['OV_MAX_NEW_TOKENS']) >= int(result['OV_CONTEXT_TOKENS']) - 128:
        raise ValueError('Output limit leaves no input budget')
    tile = result.get('GEMMA_MIXED_512_TILE', '')
    if tile not in ('', 'wideq'):
        raise ValueError('Only the validated wideq tile is supported; rejected lab tiles are not serving profiles')
    if tile and profile != 'gemma26-b70':
        raise ValueError('The wideq tile is qualified only with the gemma26-b70 profile')
    return result

def main():
    env = configure(os.environ)
    model = Path(env['OV_MODEL'])
    if not (model / 'openvino_language_model.xml').is_file():
        raise SystemExit('Expected a complete Gemma VLM export with openvino_language_model.xml')
    import openvino as ov
    core = ov.Core()
    for device in core.available_devices:
        print(device, core.get_property(device, 'FULL_DEVICE_NAME'), flush=True)
    device = env['OV_DEVICE']
    name = str(core.get_property(device, 'FULL_DEVICE_NAME'))
    if env['OV_PROFILE'] == 'gemma26-b70' and 'B70' not in name:
        raise SystemExit('The 26B/B70 profile is not qualified on this device. Do not use it on a B50.')
    if env['MOE_GROUPED_BINARY_LOOKUP'] == '1':
        plugins = glob.glob(str(Path(ov.__file__).parent / 'libs/libopenvino_intel_gpu_plugin.so'))
        if not plugins or b'MOE_GROUPED_BINARY_LOOKUP' not in Path(plugins[0]).read_bytes():
            raise SystemExit('Custom lookup missing from GPU plugin; use the custom fork image, not upstream wheels')
    if env.get('GEMMA_MIXED_512_TILE') == 'wideq':
        plugins = glob.glob(str(Path(ov.__file__).parent / 'libs/libopenvino_intel_gpu_plugin.so'))
        if not plugins or b'GEMMA_MIXED_512_TILE' not in Path(plugins[0]).read_bytes():
            raise SystemExit('Wide-query selector missing: rebuild custom runtime wheels and the custom-server image')
    print('PROFILE', env['OV_PROFILE'], 'device', device, 'context', env['OV_CONTEXT_TOKENS'], flush=True)
    os.execve(sys.executable, [sys.executable, '-u', str(Path(__file__).with_name('ovserver_moe.py'))], env)

if __name__ == '__main__':
    main()
