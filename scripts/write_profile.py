"""Render the pinned serving defaults as shell-quoted values at build time."""
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
serving, model = manifest["serving"], manifest["model"]
flags = ["--" + key.replace("_", "-") for key in ("dense_fp8", "kv_fp8", "spec") if serving[key]]
if serving["spec"]:
    flags += ["--spec-threshold", str(serving["spec_threshold"])]
if "prefill_chunk" in serving:
    flags += ["--prefill-chunk", str(serving["prefill_chunk"])]
if "park_host_mib" in serving:
    flags += ["--park-host-mib", str(serving["park_host_mib"])]
# Files the server reads; the ablation file is checked on its own because INGINE_ABLATION may replace it.
skip = {"README.md", "LICENSE", "NOTICE", "SHA256SUMS", ".gitattributes", manifest["ablation"]}
required = [name for name in dict.fromkeys(["config.json", *model["files"]]) if name not in skip]
print("# Generated from runtime-manifest.json for scripts/ingine.")
for name, value in (("host", serving["host"]), ("port", serving["port"]), ("max_ctx", serving["max_ctx"]),
                    ("model_id", serving["model_id"]), ("ablation", manifest["ablation"]),
                    ("directory", model["directory"])):
    print(f"profile_{name}={shlex.quote(str(value))}")
print("profile_flags=(" + " ".join(map(shlex.quote, flags)) + ")")
print("profile_required=(" + " ".join(map(shlex.quote, required)) + ")")
