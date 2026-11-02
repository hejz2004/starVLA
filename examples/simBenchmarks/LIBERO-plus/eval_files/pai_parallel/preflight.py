"""Check existing shared inputs and an existing simulator environment; never install/copy assets."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["inputs","simulation"]);a=p.parse_args()
    run=Path(os.environ["RUN_ROOT"]);preflight=run/"preflight"
    if a.mode=="simulation":
        import importlib.metadata as md
        import mujoco,robosuite,numpy
        from libero.libero import benchmark
        if mujoco.__version__!="3.2.3" or robosuite.__version__!="1.4.0":
            raise ValueError("Use the verified LIBERO protocol: mujoco 3.2.3 and robosuite 1.4.0")
        suites={name:benchmark.get_benchmark_dict()[name]().n_tasks for name in ("libero_10","libero_goal","libero_object","libero_spatial")}
        if suites!={"libero_10":2519,"libero_goal":2591,"libero_object":2518,"libero_spatial":2402}:
            raise ValueError("LIBERO-plus task sizes differ: "+str(suites))
        import cv2,gym,glfw,OpenGL,imageio,scipy
        import wand.version
        extra={"opencv":cv2.__version__,"gym":gym.__version__,"glfw":glfw.__version__,
               "PyOpenGL":OpenGL.__version__,"imageio":imageio.__version__,"scipy":scipy.__version__,
               "Wand":wand.version.VERSION,"ImageMagick":wand.version.MAGICK_VERSION}
        result={"status":"passed","python":sys.executable,"python_version":sys.version,"mujoco":mujoco.__version__,
                "robosuite":robosuite.__version__,"numpy":numpy.__version__,"suites":suites,"extra":extra}
        (preflight/"simulation_runtime.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True);return
    import yaml
    from starVLA.dataloader.libero_action_contract import VERSION
    checkpoint=Path(os.environ["CKPT_PATH"]).resolve(strict=True)
    cfgpath=checkpoint.parent.parent/"config.yaml";statspath=checkpoint.parent.parent/"dataset_statistics.json"
    cfg=yaml.safe_load(cfgpath.read_text());am=cfg["framework"]["action_model"]
    if cfg["framework"]["name"]!="DualStreamQwenPI" or am["action_dim"]!=16 or am["action_horizon"]!=16 or am["num_inference_timesteps"]!=4 or am.get("use_state",False) or cfg["datasets"]["vla_data"].get("action_type")!=VERSION:
        raise ValueError("This adapter requires the latest LIBERO anchored-goal binary/chunk16/flow4 checkpoint")
    stats=json.loads(statspath.read_text())
    if set(stats)!={"franka"}:raise ValueError("Expected matched franka statistics")
    libero=Path(os.environ["LIBERO_HOME"]).resolve(strict=True)
    assets=Path(os.environ["LIBERO_ASSETS"]).resolve(strict=True)
    classification=libero/"libero/libero/benchmark/task_classification.json"
    expected=Path(__file__).parent/"launchers/task_classification.json"
    if digest(classification)!=digest(expected):raise ValueError("LIBERO-plus task classification changed")
    sentinel=assets/"scenes/tabletop250/tabletop_table_FabricCarpet001_NRM_3K.xml"
    if not sentinel.is_file():raise ValueError("Missing Plus asset sentinel: "+str(sentinel))
    benchmark_root=libero/"libero/libero"
    for sub in ("bddl_files","init_files"):
        if not (benchmark_root/sub).is_dir():raise ValueError("Missing shared Plus source folder "+sub)
    # Copy only source code: Plus resolves __file__ through realpath.
    # BDDL/init-state/assets remain shared read-only links.
    def directory(path):
        path.mkdir(parents=True,exist_ok=True)
    def link(target,source):
        if target.is_symlink():
            if target.resolve()!=source.resolve(): raise ValueError("Existing resource link changed: "+str(target))
        elif target.exists(): raise ValueError("Existing resource is not a link: "+str(target))
        else: target.symlink_to(source,target_is_directory=source.is_dir())
    asset_view=run/"asset_links";directory(asset_view)
    base=os.environ.get("LIBERO_BASE_ASSETS")
    roots=([Path(base).resolve(strict=True)] if base else [])+[assets]
    names={child.name for root in roots for child in root.iterdir()}
    for name in sorted(names):
        candidates=[root/name for root in roots if (root/name).exists()]
        link(asset_view/name,candidates[-1])
    if not (asset_view/"articulated_objects/wooden_cabinet.xml").is_file():
        raise ValueError("Missing standard LIBERO articulated_objects; set LIBERO_BASE_ASSETS to the existing standard asset root")
    view=run/"libero_source"
    shutil.copytree(libero,view,dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git","__pycache__","assets","bddl_files","init_files"))
    for sub in ("bddl_files","init_files"):
        link(view/"libero/libero"/sub,benchmark_root/sub)
    link(view/"libero/libero/assets",asset_view)
    original_libero=libero;libero=view;benchmark_root=view/"libero/libero"
    configdir=run/"libero_config";directory(configdir)
    config={"benchmark_root":str(benchmark_root),"bddl_files":str(benchmark_root/"bddl_files"),
            "init_states":str(benchmark_root/"init_files"),"datasets":str(run/"unused_datasets"),"assets":str(asset_view)}
    (configdir/"config.yaml").write_text("".join(f"{k}: {json.dumps(v)}\n" for k,v in config.items()))
    result={"status":"passed","checkpoint":str(checkpoint),"checkpoint_bytes":checkpoint.stat().st_size,
            "config_sha256":digest(cfgpath),"statistics_sha256":digest(statspath),"source_root":os.environ["STARVLA_DIR"]}
    (preflight/"input_artifacts.json").write_text(json.dumps(result,indent=2)+"\n")
    (preflight/"resources.json").write_text(json.dumps({"status":"passed","libero_home":str(libero),"shared_libero_home":str(original_libero),"assets":str(asset_view),"shared_plus_assets":str(assets),"classification_sha256":digest(classification),"asset_copy":False},indent=2)+"\n")
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
