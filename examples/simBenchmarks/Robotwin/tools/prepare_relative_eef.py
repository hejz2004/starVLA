"""Index native clean50 HDF5 and fit UMI-style relative EEF statistics offline."""
import argparse
import json
from pathlib import Path
from starVLA.dataloader.robotwin_hdf5 import prepare_assets

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--image-encoding', choices=['legacy_cv2_rgb','standard_rgb'], default='legacy_cv2_rgb')
    p.add_argument('--task-manifest', help='JSON with an explicit task_names list')
    args = p.parse_args()
    tasks = json.loads(Path(args.task_manifest).read_text())['task_names'] if args.task_manifest else None
    result = prepare_assets(args.source_root, args.output, image_encoding=args.image_encoding,
                            task_names=tasks, expected_tasks=len(tasks) if tasks is not None else 50)
    print(result['summary'])
