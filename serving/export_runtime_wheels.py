"""Package an installed, validated custom runtime without copying credentials/models.

Run using that runtime's Python 3.12, after installing `wheel` in a separate
packaging environment and pass --wheel-python for that environment's interpreter.
Only the files owned by the three named runtime distributions are included.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wheel-python', required=True)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.platform != 'linux':
        raise SystemExit('This Docker package expects Linux/Python 3.12 runtime wheels')
    import openvino
    plugin = Path(openvino.__file__).parent / 'libs/libopenvino_intel_gpu_plugin.so'
    if b'MOE_GROUPED_BINARY_LOOKUP' not in plugin.read_bytes():
        raise SystemExit('Installed runtime does not contain the custom lookup patch')
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob('*.whl')):
        raise SystemExit('Output already has wheels; use an empty output directory')
    receipt = {'gpu_plugin_sha256': hashlib.sha256(plugin.read_bytes()).hexdigest(), 'wheels': {}}
    for name in ('openvino', 'openvino-tokenizers', 'openvino-genai'):
        dist = metadata.distribution(name)
        site = Path(dist.locate_file('')).resolve()
        with tempfile.TemporaryDirectory(prefix='ov-wheel-') as temp:
            stage = Path(temp)
            for entry in dist.files or []:
                relative = Path(str(entry))
                # Console scripts are reconstructed by wheel installation.
                if relative.is_absolute() or '..' in relative.parts or '__pycache__' in relative.parts:
                    continue
                source = Path(dist.locate_file(entry)).resolve()
                if not source.is_relative_to(site):
                    raise SystemExit('Distribution file escapes site-packages: ' + str(entry))
                if source.is_file():
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            if name == 'openvino':
                # Runtime integration may add/replace files without updating pip's RECORD.
                shutil.copytree(Path(openvino.__file__).parent, stage / 'openvino', dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            subprocess.run([args.wheel_python, '-m', 'wheel', 'pack', str(stage),
                            '--dest-dir', str(args.output.resolve())], check=True)
    for path in args.output.glob('*.whl'):
        receipt['wheels'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / 'runtime-manifest.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
