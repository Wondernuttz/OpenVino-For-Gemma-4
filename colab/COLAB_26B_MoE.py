########################################################################
#  26B MoE HERETIC (your bots) -> OPENVINO INT4   *** ROUTER KEPT UNQUANTIZED ***
#
#  THE FIX (root cause of every failed build): the MoE router was being quantized to int4, which
#  breaks the GPU's MoE fusion so the model WON'T LOAD. CELL 3 excludes it (ignored_scope .*router.*),
#  matching Intel's own config, keeping the heretic's OWN router intact. It VERIFIES the router was
#  excluded before it saves, then writes to BOTH HuggingFace (private) and Google Drive.
#
#  Everything lives on the 368 GB scratch disk (/mnt/local-scratch) so nothing fills up.
#  RUNTIME:  A100 + High-RAM, with /mnt/local-scratch present.
#  Run:  CELL 1 (install)  ->  CELL 2 (download)  ->  CELL 3 (convert). Re-running CELL 3 won't re-download.
########################################################################


# ========================= CELL 1 : install (run once) =========================
# git-main optimum-intel + nightly openvino + transformers 5.5.0 -- the toolchain that DID convert
# (group 64). The ONLY thing it got wrong was quantizing the MoE router; CELL 3 now excludes it.
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
!pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
!pip install -qU --pre "openvino" "openvino-tokenizers" "openvino-genai" "nncf"
!pip install -q "transformers==5.5.0" "Pillow" "huggingface_hub"
import transformers, openvino, nncf
print("transformers", transformers.__version__, "| openvino", openvino.__version__, "| nncf", nncf.__version__)


# ========================= CELL 2 : DOWNLOAD the model (only) =========================
import os, shutil
SD = "/mnt/local-scratch"
assert os.path.isdir(SD), "/mnt/local-scratch is missing -- do not run this on /content."
os.environ.update({"HF_HUB_DISABLE_XET": "1", "HF_HOME": SD + "/hf", "HF_HUB_CACHE": SD + "/hf/hub"})
os.makedirs(SD + "/hf", exist_ok=True)
HF_TOKEN    = "YOUR_HF_TOKEN_HERE"
SOURCE_REPO = "llmfan46/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic"
WORK = SD + "/work"                                      # everything on the scratch disk
SRC_DIR = os.path.join(WORK, "src_model")
os.makedirs(WORK, exist_ok=True)
print(">>> downloading to:", SRC_DIR, "(on the big scratch disk)", flush=True)
from huggingface_hub import login, snapshot_download
login(token=HF_TOKEN)
snapshot_download(SOURCE_REPO, local_dir=SRC_DIR, token=HF_TOKEN)
shutil.rmtree(SRC_DIR + "/.cache", ignore_errors=True)
shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)
_, u, f = shutil.disk_usage(WORK)
print(">>> DOWNLOADED. disk on %s: %.0f used / %.0f free GB" % (WORK, u/1e9, f/1e9), flush=True)
print(">>> Now run CELL 3 to convert.")


# ========================= CELL 3 : CONVERT (ROUTER EXCLUDED) -> VERIFY -> SAVE TO HF + DRIVE =========================
# *** THE KEY FIX: ignored_scope ".*router.*" keeps the MoE router UNQUANTIZED (matches Intel's config).
#     Quantizing the router is what made every prior build fail to LOAD on the GPU. optimum-cli has no
#     flag for this, so CELL 3 runs the PYTHON API in a subprocess. group_size 64 = Intel's value.
# Also baked in, all proven during debugging:
#  - VERIFY step: aborts BEFORE upload if ignored_scope didn't take (so you never save a bad file again).
#  - OFFLINE env on the subprocess (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) -> no network HANG (model is local).
#  - SWAP cushion so the ~175 GB convert peak can't OOM your 167 GB RAM.
#  - Every cache/temp/output PINNED to scratch so the main disk can't fill.
#  - Saves to BOTH your private HF repo AND Google Drive; live [mon] + full log on scratch.
# NOTE: int4 export is CPU+RAM+disk -- the GPU sits idle during conversion (that is correct).
import os, shutil, subprocess, sys, threading, time
SD = "/mnt/local-scratch"
assert os.path.isdir(SD), "/mnt/local-scratch is missing -- do not run this on /content."
os.environ.update({"HF_HUB_DISABLE_XET": "1", "HF_HOME": SD + "/hf", "HF_HUB_CACHE": SD + "/hf/hub",
                   "XDG_CACHE_HOME": SD + "/xdg", "TMPDIR": SD + "/tmp"})
for d in ("/hf", "/xdg", "/tmp"): os.makedirs(SD + d, exist_ok=True)
HF_USERNAME = "Wondernutts"
HF_TOKEN    = "YOUR_HF_TOKEN_HERE"
SOURCE_REPO = "llmfan46/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic"
WORK = SD + "/work"; SRC_DIR, OUT_DIR = WORK + "/src_model", WORK + "/ov_out"
LOG = WORK + "/convert.log"
assert os.path.exists(SRC_DIR + "/config.json"), "No download -- run CELL 2 first."
shutil.rmtree(OUT_DIR, ignore_errors=True)

# --- swap so the ~175 GB convert peak can't OOM (reset any old swapfile first) ---
SWAP = SD + "/swapfile"
subprocess.run("swapoff %s 2>/dev/null; rm -f %s" % (SWAP, SWAP), shell=True)
free_gb = shutil.disk_usage(SD).free // (1024**3)
assert free_gb > 130, "scratch too full before convert: only %d GB free" % free_gb
print(">>> swap 64 GB on %s (disk free=%d GB)" % (SD, free_gb), flush=True)
subprocess.run("fallocate -l 64G %s && chmod 600 %s && mkswap %s && swapon %s" % (SWAP, SWAP, SWAP, SWAP), shell=True, check=True)
subprocess.run("free -g; swapon --show 2>/dev/null", shell=True)

# --- OFFLINE env for the convert subprocess ONLY (no network = no hang; model is local) ---
conv_env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

# --- live monitor: real RAM + swap + both disks every 15s ---
def mon():
    while True:
        try:
            lines = subprocess.run("free -g", shell=True, capture_output=True, text=True).stdout.splitlines()
            mem = next(l.split() for l in lines if l.startswith("Mem:"))
            swp = next(l.split() for l in lines if l.startswith("Swap:"))
            print("[mon] RAM %s/%sGB | swap %sGB | MAIN-disk %dGB | scratch %dGB" % (
                mem[2], mem[1], swp[2],
                shutil.disk_usage("/content").used // (1024**3),
                shutil.disk_usage(SD).used // (1024**3)), flush=True)
        except Exception as e:
            print("[mon] err:", e, flush=True)
        time.sleep(15)
threading.Thread(target=mon, daemon=True).start()

# --- THE FIX: write a Python conversion script that EXCLUDES the MoE router from quantization ---
# optimum-cli has no flag for ignored_scope, so we must use the Python API. group_size 64 = Intel's value.
# ignored_scope .*router.* keeps the heretic's OWN router UNQUANTIZED -> GPU can fuse the MoE -> it LOADS.
CONV = WORK + "/convert.py"
with open(CONV, "w") as cf:
    cf.write(
'import os, shutil\n'
'SRC, OUT = os.environ["CONV_SRC"], os.environ["CONV_OUT"]\n'
'shutil.rmtree(OUT, ignore_errors=True)\n'
'from optimum.intel import OVModelForVisualCausalLM, OVWeightQuantizationConfig\n'
'q = OVWeightQuantizationConfig(bits=4, sym=False, group_size=64, dq_group_size=64, ratio=1.0,\n'
'        quant_method="awq",                          # AWQ (Intel recipe, data-free) -- LOAD-BEARING for coherence; awq=True is SILENTLY IGNORED, plain int4 -> garbage\n'
'        group_size_fallback="adjust",\n'
'        ignored_scope={"patterns": [".*router.*"]})  # keep the heretic router UNQUANTIZED  [matches Intel openvino_config.json field-for-field]\n'
'print(">>> exporting + AWQ int4, ROUTER EXCLUDED (this adds time vs plain int4) ...", flush=True)\n'
'm = OVModelForVisualCausalLM.from_pretrained(SRC, export=True, quantization_config=q, trust_remote_code=True)\n'
'm.save_pretrained(OUT)\n'
'# the Python export does NOT build the OV tokenizer -- build it from the source so the model is complete\n'
'from transformers import AutoTokenizer\n'
'import openvino_tokenizers, openvino as _ov\n'
'_tok = AutoTokenizer.from_pretrained(SRC)\n'
'_t, _d = openvino_tokenizers.convert_tokenizer(_tok, with_detokenizer=True)\n'
'_ov.save_model(_t, OUT + "/openvino_tokenizer.xml"); _ov.save_model(_d, OUT + "/openvino_detokenizer.xml")\n'
'_tok.save_pretrained(OUT)\n'
'print(">>> SAVED with AWQ + OWN tokenizer:", OUT, flush=True)\n'
    )
conv_env = dict(conv_env, CONV_SRC=SRC_DIR, CONV_OUT=OUT_DIR)
print(">>> converting (router EXCLUDED, offline+swap) -- live output; full log:", LOG, flush=True)
with open(LOG, "w", buffering=1) as f:
    p = subprocess.Popen([sys.executable, CONV], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=conv_env)
    for line in p.stdout:
        print(line, end=""); f.write(line)
    rc = p.wait()
print(">>> rc =", rc, flush=True)
if rc != 0:
    print(">>> last 160 log lines:", flush=True)
    subprocess.run("tail -160 %s" % LOG, shell=True)
    raise SystemExit("convert failed (rc=%d); see %s" % (rc, LOG))

# --- VERIFY router excluded + AWQ applied + tokenizer built, BEFORE wasting an upload ---
import json
_qc = json.load(open(OUT_DIR + "/openvino_config.json")).get("quantization_config", {})
_lm = _qc.get("quantization_configs", {}).get("lm_model", _qc)
_isc = _qc.get("ignored_scope") or _lm.get("ignored_scope")
_qm  = str(_lm.get("quant_method") or _qc.get("default_config", {}).get("quant_method") or "")
_tok = os.path.exists(OUT_DIR + "/openvino_tokenizer.xml")
print(">>> CHECK: ignored_scope=%s | quant_method=%s | tokenizer=%s" % (json.dumps(_isc), _qm, _tok), flush=True)
assert _isc, "ROUTER NOT EXCLUDED (ignored_scope null) -- aborting before upload"
assert "awq" in _qm.lower(), "AWQ NOT APPLIED (quant_method=%s) -- aborting before upload" % _qm
assert _tok, "OV TOKENIZER not built -- aborting before upload"

# --- save to HF (private, ONLINE) ---
print(">>> converted. uploading to HF...", flush=True)
from huggingface_hub import login, create_repo, upload_folder
login(token=HF_TOKEN)
out_repo = HF_USERNAME + "/" + SOURCE_REPO.split("/")[-1] + "-int4-ov"
create_repo(out_repo, private=True, exist_ok=True, token=HF_TOKEN)
upload_folder(folder_path=OUT_DIR, repo_id=out_repo, repo_type="model", token=HF_TOKEN)
print(">>> SAVED TO HF:", out_repo, flush=True)

# --- ALSO save to Google Drive ---
try:
    from google.colab import drive; drive.mount('/content/drive')
    dst = "/content/drive/MyDrive/ov_heretics/" + SOURCE_REPO.split("/")[-1] + "-int4-ov"
    os.makedirs("/content/drive/MyDrive/ov_heretics", exist_ok=True)
    shutil.copytree(OUT_DIR, dst, dirs_exist_ok=True)
    print(">>> ALSO SAVED TO DRIVE:", dst, flush=True)
except Exception as e:
    print(">>> Drive save skipped:", str(e)[:80], flush=True)
print("\n>>> DONE.  HF:", out_repo, flush=True)
