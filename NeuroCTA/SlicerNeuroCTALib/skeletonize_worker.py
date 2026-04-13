# skeletonize_worker.py
import sys
import json
import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import csv

def compute_soam(coords, voxel_spacing):
    """
    Compute the Sum of Angles Metric (SOAM) for a sequence of 3D coordinates.
    Coords are in voxel (ZYX) space; voxel_spacing converts to physical units.
    Returns SOAM in radians per unit length.
    """
    if len(coords) < 4:
        return 0.0

    # Convert to physical space (ZYX * spacing)
    pts = coords * np.array(voxel_spacing)

    total_angle = 0.0
    valid_points = 0

    for k in range(1, len(pts) - 2):
        T1 = pts[k] - pts[k - 1]
        T2 = pts[k + 1] - pts[k]
        T3 = pts[k + 2] - pts[k + 1]

        n1 = np.linalg.norm(T1)
        n2 = np.linalg.norm(T2)
        n3 = np.linalg.norm(T3)

        if n1 < 1e-6 or n2 < 1e-6 or n3 < 1e-6:
            continue

        T1n = T1 / n1
        T2n = T2 / n2
        T3n = T3 / n3

        # In-plane angle at Pk
        dot_ip = np.clip(T1n @ T2n, -1.0, 1.0)
        IP = np.arccos(dot_ip)

        # Torsional angle at Pk
        cross1 = np.cross(T1n, T2n)
        cross2 = np.cross(T2n, T3n)
        nc1 = np.linalg.norm(cross1)
        nc2 = np.linalg.norm(cross2)

        if nc1 < 1e-6 or nc2 < 1e-6:
            TP = 0.0
        else:
            cross1n = cross1 / nc1
            cross2n = cross2 / nc2
            dot_tp = np.clip(cross1n @ cross2n, -1.0, 1.0)
            TP = np.arccos(dot_tp)
            # Set to 0 at inflection points (TP == pi)
            if np.isclose(TP, np.pi):
                TP = 0.0

        # Combined angle at Pk
        CP = np.sqrt(IP**2 + TP**2)
        total_angle += CP
        valid_points += 1

    if valid_points == 0:
        return 0.0

    # Normalize by total curve length
    deltas = np.diff(pts, axis=0)
    total_length = np.sum(np.linalg.norm(deltas, axis=1))

    if total_length < 1e-6:
        return 0.0

    return total_angle / total_length

def main():
    args = json.loads(sys.argv[1])
    volume = np.load(args["volume_path"])
    labels_info = json.loads(open(args["labels_path"]).read())
    total_progress_steps = len(labels_info) + 2  # +2 for skeletonization and distance transform steps
    voxel_spacing = args.get("voxel_spacing", [1.0, 1.0, 1.0])

    print(f"Starting skeletonization of {len(labels_info)} segments", flush=True)

    full_skeleton = skeletonize((volume > 0).astype(np.uint8))

    # Updates progress bar (see _onProgressInfo in Logic.py)
    print(f"PROGRESS:{1}:{total_progress_steps}\t[{1}/{total_progress_steps}] Skeletonization", flush=True)

    # Compute distance transform for all segments at once to avoid redundant calculations
    full_dist_transform = distance_transform_edt(volume > 0, sampling=voxel_spacing)
    full_skel_obj = Skeleton(full_skeleton, spacing=voxel_spacing)
    full_skel_stats = summarize(skel=full_skel_obj, separator='-')

    # Updates progress bar (see _onProgressInfo in Logic.py)
    print(f"PROGRESS:{2}:{total_progress_steps}\t[{2}/{total_progress_steps}] Distance Transform", flush=True)

    # Get all branch point and end points
    bp_coords = full_skel_obj.coordinates[np.where(full_skel_obj.degrees > 2)[0]]
    ep_coords = full_skel_obj.coordinates[np.where(full_skel_obj.degrees == 1)[0]]
    bp_coords = np.array(bp_coords.tolist(), dtype=int)
    ep_coords = np.array(ep_coords.tolist(), dtype=int)

    # Assigns each branch to a segment (vessel class) based on the label at the midpoint of the branch
    branch_coords = np.array([
        full_skel_obj.path_coordinates(i)[len(full_skel_obj.path_coordinates(i))//2].astype(int)
        for i in range(len(full_skel_stats))
    ])
    branch_labels = volume[branch_coords[:, 0], branch_coords[:, 1], branch_coords[:, 2]]
    full_skel_stats["segment_label"] = branch_labels

    # Iterate through each segment to extract skeleton coordinates, features, and branch/end points
    results = {}
    for i, label_info in enumerate(labels_info):
        label = label_info["label"]
        segment_name = label_info["name"]
        segment_mask = (volume == label)
        skel_mask = full_skeleton & segment_mask
        skel_coords = np.argwhere(skel_mask)

        features = {}
        if skel_coords.shape[0] > 1:
            seg_stats = full_skel_stats[full_skel_stats["segment_label"] == label]
            if len(seg_stats) > 0:
                features["length"] = float(seg_stats["branch-distance"].sum())
                features["n_branches"] = int(len(seg_stats))
                features["n_branch_points"] = int((seg_stats["branch-type"] == 2).sum())

                # Radius
                skeleton_radii = full_dist_transform[skel_mask]
                features["mean_radius"] = float(skeleton_radii.mean())
                features["min_radius"]  = float(skeleton_radii.min())
                features["max_radius"]  = float(skeleton_radii.max())

                # Tortuosity - Distance Metric (DM) = branch distance / euclidean distance [CITE]
                features["tortuosity_dm"] = float(seg_stats["branch-distance"].sum() / seg_stats["euclidean-distance"].sum())

                # Tortuosity - Sum of Angle Metric (SOAM) = length-weighted average across all branches in segment [CITE]
                if skel_coords.shape[0] < 4:
                    features["tortuosity_soam"] = 0.0
                else:
                    soam_total = 0.0
                    length_total = 0.0
                    for row_idx in seg_stats.index:
                        path = full_skel_obj.path_coordinates(int(row_idx)).astype(float)
                        branch_length = float(seg_stats.loc[row_idx, "branch-distance"])
                        soam_total += compute_soam(path, voxel_spacing) * branch_length
                        length_total += branch_length
                    features["tortuosity_soam"] = soam_total / length_total if length_total > 0 else 0.0

            # branch points for this segment
            if len(bp_coords) > 0:
                bp_mask  = segment_mask[bp_coords[:, 0], bp_coords[:, 1], bp_coords[:, 2]]
                seg_bp   = bp_coords[bp_mask]
                features["branch_point_coords"] = seg_bp.tolist()
            else:
                features["branch_point_coords"] = []

            # end points for this segment
            if len(ep_coords) > 0:
                ep_mask  = segment_mask[ep_coords[:, 0], ep_coords[:, 1], ep_coords[:, 2]]
                seg_ep   = ep_coords[ep_mask]
                features["end_point_coords"] = seg_ep.tolist()
            else:
                features["end_point_coords"] = []            


        results[segment_name] = {
            "coords":   skel_coords.tolist(),
            "color":    label_info["color"],
            "features": features
        }

        print(f"PROGRESS:{i+3}:{total_progress_steps}\t[{i+3}/{total_progress_steps}] {segment_name} - {len(skel_coords)} points", flush=True)

    # Save centerline points CSV (ZYX voxel coords converted to physical space)
    cl_csv_path = args["output_path"].replace("results.json", "centerline_points.csv")
    with open(cl_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "x", "y", "z", "radius"])
        for segment_name, data in results.items():
            coords = data["coords"]
            features = data.get("features", {})
            # get per-point radii from dist transform if available
            # for coord in coords:
            #     z, y, x = coord
            #     # convert ZYX voxel to physical mm
            #     x_mm = x * voxel_spacing[0]
            #     y_mm = y * voxel_spacing[1]
            #     z_mm = z * voxel_spacing[2]
            #     writer.writerow([segment_name, x_mm, y_mm, z_mm, ""])
            label_info_match = next((l for l in labels_info if l["name"] == segment_name), None)
            if label_info_match:
                for coord in coords:
                    z, y, x = coord
                    radius = float(full_dist_transform[z, y, x])
                    writer.writerow([segment_name, x * voxel_spacing[0], y * voxel_spacing[1], z * voxel_spacing[2], radius])

    # Save per-segment features CSV
    seg_csv_path = args["output_path"].replace("results.json", "segment_features.csv")
    with open(seg_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "segment", "length", "n_branches", "n_branch_points",
            "tortuosity_dm", "tortuosity_soam",
            "mean_radius", "min_radius", "max_radius"
        ])
        writer.writeheader()
        for segment_name, data in results.items():
            features = data.get("features", {})
            if features:
                writer.writerow({
                    "segment":          segment_name,
                    "length":           features.get("length", ""),
                    "n_branches":       features.get("n_branches", ""),
                    "n_branch_points":  features.get("n_branch_points", ""),
                    "tortuosity_dm":    features.get("tortuosity_dm", ""),
                    "tortuosity_soam":  features.get("tortuosity_soam", ""),
                    "mean_radius":      features.get("mean_radius", ""),
                    "min_radius":       features.get("min_radius", ""),
                    "max_radius":       features.get("max_radius", ""),
                })

    with open(args["output_path"], "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
