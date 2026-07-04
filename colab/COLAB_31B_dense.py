########################################################################
#  31B HERETIC -> OPENVINO INT4   *** THE RIGHT WAY ***
#  Routes EVERYTHING (download + convert + temp) to the BIGGEST disk so
#  /content can't fill up. Needs lots of RAM.
#
#  RUNTIME:  Runtime > Change runtime type > A100 GPU > **High-RAM** (required!)
#  Run CELL 1 -> 2 -> 3.  CELL 3 prints which disk it's using -- should be the 368 GB one.
########################################################################


# ============================ CELL 1 : install =======================================
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
!pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
!pip install -q "transformers==5.5.0" "nncf" "openvino==2026.2.0" "openvino-tokenizers==2026.2.0" "Pillow" "huggingface_hub"
import transformers, openvino, nncf
print("transformers", transformers.__version__, "| openvino", openvino.__version__, "| nncf", nncf.__version__)


# ============================ CELL 2 : settings ======================================
HF_USERNAME = "Wondernutts"
HF_TOKEN    = "YOUR_HF_TOKEN_HERE"
SOURCE_REPO = "llmfan46/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic"
from huggingface_hub import login
login(token=HF_TOKEN)
print("Logged in. Converting:", SOURCE_REPO.split("/")[-1])


# ============================ CELL 3 : convert on the BIGGEST disk ====================
import os, shutil, subprocess, json
os.environ["HF_HUB_DISABLE_XET"] = "1"

# --- find the biggest WRITABLE disk and put everything there ---
best, best_avail = "/content", 0
for line in subprocess.run("df -B1 --output=avail,target", shell=True, capture_output=True, text=True).stdout.strip().split("\n")[1:]:
    p = line.split(None, 1)
    if len(p) != 2: continue
    avail, mnt = int(p[0]), p[1].strip()
    if mnt.startswith(("/proc", "/sys", "/dev")): continue
    try:
        t = os.path.join(mnt, ".wt"); os.makedirs(t, exist_ok=True); os.rmdir(t)
    except Exception:
        continue
    if avail > best_avail:
        best, best_avail = mnt, avail
if os.path.isdir("/mnt/local-scratch"):          # the known 368 GB scratch disk -- use it explicitly
    best = "/mnt/local-scratch"
    best_avail = shutil.disk_usage(best)[2]
WORK = os.path.join(best, "work")
os.makedirs(WORK, exist_ok=True)
os.environ["TMPDIR"] = os.path.join(WORK, "tmp"); os.makedirs(os.environ["TMPDIR"], exist_ok=True)
print(">>> USING DISK:", best, "(%.0f GB free) -- download + convert + temp all go here" % (best_avail/1e9), flush=True)

def report(tag):
    _, u, f = shutil.disk_usage(best)
    print("   [%-11s] %s : %.0f GB used / %.0f GB free" % (tag, best, u/1e9, f/1e9), flush=True)

from huggingface_hub import snapshot_download, create_repo, upload_folder
name     = SOURCE_REPO.split("/")[-1]
out_repo = HF_USERNAME + "/" + name + "-int4-ov"
src_dir  = os.path.join(WORK, "src_model")
out_dir  = os.path.join(WORK, "ov_out")
shutil.rmtree(src_dir, ignore_errors=True); shutil.rmtree(out_dir, ignore_errors=True)
report("start")

print("\n>>> 1/4 downloading " + name + " ...", flush=True)
snapshot_download(SOURCE_REPO, local_dir=src_dir, token=HF_TOKEN)
shutil.rmtree(src_dir + "/.cache", ignore_errors=True)
shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)
report("downloaded")

mt = json.load(open(src_dir + "/config.json")).get("model_type")
assert mt == "gemma4", "model_type is '%s'" % mt

print("\n>>> 2/4 converting to int4 (~40-70 min). Watch the RAM gauge.", flush=True)
rc = subprocess.run(["optimum-cli","export","openvino","--model",src_dir,"--task","image-text-to-text",
    "--weight-format","int4","--group-size","128","--ratio","1.0", out_dir]).returncode
assert rc == 0, "convert failed (rc=%d) -- if rc=-9 you need High-RAM" % rc
report("converted")

print("\n>>> 3/4 uploading to HF: " + out_repo, flush=True)
create_repo(out_repo, private=True, exist_ok=True, token=HF_TOKEN)
upload_folder(folder_path=out_dir, repo_id=out_repo, repo_type="model", token=HF_TOKEN)
print(">>> SAVED TO HF:", out_repo, flush=True)

print("\n>>> 4/4 Drive backup ...", flush=True)
try:
    from google.colab import drive; drive.mount('/content/drive')
    os.makedirs('/content/drive/MyDrive/ov_heretics', exist_ok=True)
    shutil.copytree(out_dir, '/content/drive/MyDrive/ov_heretics/' + name + '-int4-ov', dirs_exist_ok=True)
    print(">>> ALSO saved to Drive", flush=True)
except Exception as e:
    print(">>> Drive skipped:", str(e)[:60], flush=True)

shutil.rmtree(src_dir, ignore_errors=True); shutil.rmtree(out_dir, ignore_errors=True)
report("done")
print("\n>>> DONE.  Send Claude this repo:  " + out_repo)
