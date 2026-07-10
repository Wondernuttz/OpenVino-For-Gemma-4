########################################################################
#  12B HERETIC (gemma4_unified)  ->  OPENVINO INT4   *** EXPERIMENTAL ***
#  This is the 12B ONLY. It needs a DIFFERENT toolchain than the 26B/31B:
#  the unmerged openvino-agent fork (PR #1770) + transformers >= 5.10.
#
#  >> RUN THIS IN A FRESH COLAB SESSION (different transformers version than
#     the 26B/31B notebook -- they can't share a runtime). A100 GPU. <<
#
#  Heads up: this path is experimental. The export may fail, and even if it
#  succeeds we still have to test whether it RUNS on the box. Lowest priority.
########################################################################


# ============================ CELL 1 :  install (experimental fork) ==================
!pip install -q "git+https://github.com/openvino-agent/optimum-intel.git@support_gemma-4-12B" --extra-index-url https://download.pytorch.org/whl/cpu
!pip install -q "transformers==5.10.2" "nncf" "openvino==2026.2.0" "openvino-tokenizers==2026.2.0" "Pillow" "huggingface_hub"
import transformers, openvino, nncf
print("transformers", transformers.__version__, "| openvino", openvino.__version__, "| nncf", nncf.__version__)
print("Need transformers >= 5.10 for gemma4_unified. If lower: Runtime > Restart session, re-run.")

from google.colab import drive
drive.mount('/content/drive')
import os; os.makedirs('/content/drive/MyDrive/ov_heretics', exist_ok=True)
print("Drive mounted.")


# ============================ CELL 2 :  settings (already filled) ====================
HF_USERNAME = "Wondernutts"
HF_TOKEN    = "YOUR_HF_TOKEN_HERE"   # temporary - revoke when done
SOURCE_REPO = "llmfan46/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic"
from huggingface_hub import login
login(token=HF_TOKEN)
print("Logged in. Converting:", SOURCE_REPO)


# ============================ CELL 3 :  convert -> Drive + HF ========================
import shutil, subprocess, json
from huggingface_hub import snapshot_download, create_repo, upload_folder
name = SOURCE_REPO.split("/")[-1]
out_repo = HF_USERNAME + "/" + name + "-int4-ov"
src_dir, out_dir = "/content/src_model", "/content/ov_out"
shutil.rmtree(src_dir, ignore_errors=True); shutil.rmtree(out_dir, ignore_errors=True)

print("[1/5] downloading ...", flush=True)
snapshot_download(SOURCE_REPO, local_dir=src_dir, token=HF_TOKEN)
print("      model_type:", json.load(open(src_dir + "/config.json")).get("model_type"), flush=True)

print("[2/5] converting (EXPERIMENTAL gemma4_unified path, ~20-40 min) ...", flush=True)
rc = subprocess.run(["optimum-cli","export","openvino","--model",src_dir,
    "--task","image-text-to-text","--weight-format","int4","--group-size","128","--ratio","1.0",
    out_dir]).returncode
if rc != 0:
    print("\n  CONVERT FAILED (rc=" + str(rc) + ") -- the 12B experimental path didn't take. Tell Claude.")
else:
    print("[3/5] Drive backup ...", flush=True)
    drive_dir = "/content/drive/MyDrive/ov_heretics/" + name + "-int4-ov"
    try:
        shutil.rmtree(drive_dir, ignore_errors=True); shutil.copytree(out_dir, drive_dir)
        print("      Drive ->", drive_dir, flush=True)
    except Exception as e:
        print("      Drive skipped:", str(e)[:70], flush=True)

    print("[4/5] uploading to HF:", out_repo, "...", flush=True)
    create_repo(out_repo, private=True, exist_ok=True, token=HF_TOKEN)
    upload_folder(folder_path=out_dir, repo_id=out_repo, repo_type="model", token=HF_TOKEN)
    print("[5/5] DONE ->", out_repo)
    print("\n>>> Tell Claude this repo:", out_repo, "<<<")
