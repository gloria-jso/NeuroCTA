# skeletonize_worker.py
import sys
import json
import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from scipy.ndimage import distance_transform_edt

results = {}

def main():
    args = json.loads(sys.argv[1])
    volume = np.load(args["volume_path"])
    labels_info = json.loads(open(args["labels_path"]).read())
    total = len(labels_info)
    voxel_spacing = args.get("voxel_spacing", [1.0, 1.0, 1.0])

    print(f"Starting skeletonization of {len(labels_info)} segments", flush=True)

    full_skeleton = skeletonize((volume > 0).astype(np.uint8))
    full_dist     = distance_transform_edt(volume > 0, sampling=voxel_spacing)
    full_skel_obj = Skeleton(full_skeleton, spacing=voxel_spacing)
    full_stats    = summarize(skel=full_skel_obj, separator='-')

    # all branch point coords — extracted once
    all_bp_coords = np.array(full_skel_obj.coordinates[
        np.where(full_skel_obj.degrees > 2)[0]
    ].tolist(), dtype=int)

    # map branches to segments via midpoint
    branch_coords = np.array([
        full_skel_obj.path_coordinates(i)[len(full_skel_obj.path_coordinates(i))//2].astype(int)
        for i in range(len(full_stats))
    ])
    branch_labels = volume[branch_coords[:, 0], branch_coords[:, 1], branch_coords[:, 2]]
    full_stats["segment_label"] = branch_labels

    results = {}
    for i, label_info in enumerate(labels_info):
        label        = label_info["label"]
        segment_name = label_info["name"]
        mask         = (volume == label)
        skel_mask    = full_skeleton & mask
        coords       = np.argwhere(skel_mask)

        features = {}
        if coords.shape[0] > 1:
            seg_stats = full_stats[full_stats["segment_label"] == label]
            if len(seg_stats) > 0:
                features["length"]          = float(seg_stats["branch-distance"].sum())
                features["tortuosity"]      = float(seg_stats["branch-distance"].sum() /
                                                    seg_stats["euclidean-distance"].sum())
                features["n_branches"]      = int(len(seg_stats))
                features["n_branch_points"] = int((seg_stats["branch-type"] == 2).sum())

            radii                       = full_dist[skel_mask]
            features["mean_radius"]     = float(radii.mean())
            features["min_radius"]      = float(radii.min())
            features["max_radius"]      = float(radii.max())

            # branch points for this segment
            if len(all_bp_coords) > 0:
                bp_mask  = mask[all_bp_coords[:, 0], all_bp_coords[:, 1], all_bp_coords[:, 2]]
                seg_bp   = all_bp_coords[bp_mask]
                features["branch_point_coords"] = seg_bp.tolist()
            else:
                features["branch_point_coords"] = []

        results[segment_name] = {
            "coords":   coords.tolist(),
            "color":    label_info["color"],
            "features": features
        }
        print(f"PROGRESS:{i+1}:{total}\t[{i+1}/{total}] {segment_name} — {len(coords)} points", flush=True)

    with open(args["output_path"], "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
