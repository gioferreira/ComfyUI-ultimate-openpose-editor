import math
import json
import numpy as np
import matplotlib
import cv2
from comfy.utils import ProgressBar
from typing import List, Dict

eps = 0.01


def extend_scalelist(
    scalelist_behavior,
    pose_json,
    hands_scale,
    body_scale,
    head_scale,
    overall_scale,
    match_scalelist_method,
    only_scale_pose_index,
) -> List[list]:
    try:
        if pose_json.startswith("{"):
            pose_json = "[{}]".format(pose_json)
        poses = json.loads(pose_json)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing pose JSON: {e}")
        return [], [], [], []

    if not isinstance(poses, list):
        poses = [poses] if isinstance(poses, dict) else []

    # initialize scale lists
    hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = [], [], [], []
    num_imgs = 0
    num_poses = 0
    scale_values = [hands_scale, body_scale, head_scale, overall_scale]
    scale_lists = [hands_scalelist, body_scalelist, head_scalelist, overall_scalelist]

    for img in poses:
        if not isinstance(img, dict):
            continue

        default_scale = 0.0
        default_num_person = 1
        if "people" in img and img["people"] and isinstance(img["people"], list):
            default_scale = 1.0
            default_num_person = len(img["people"])
            subscales = [default_scale] * default_num_person
            if scalelist_behavior == "poses":
                for i, scales in enumerate(scale_values):
                    if isinstance(scales, (list, tuple)):
                        if len(scales) >= num_poses + default_num_person:
                            if (
                                only_scale_pose_index < default_num_person
                                and only_scale_pose_index >= -default_num_person
                            ):
                                subscales[only_scale_pose_index] = scales[
                                    num_poses + only_scale_pose_index
                                ]
                            else:
                                subscales = scales[
                                    num_poses : num_poses + default_num_person
                                ]
                        else:
                            if match_scalelist_method == "no extend":
                                subscales = [default_scale] * default_num_person
                            elif match_scalelist_method == "loop extend":
                                extend_scaleslist = scales * math.ceil(
                                    (num_poses + default_num_person) / len(scales)
                                )
                                if (
                                    only_scale_pose_index < default_num_person
                                    and only_scale_pose_index >= -default_num_person
                                ):
                                    subscales[only_scale_pose_index] = (
                                        extend_scaleslist[
                                            num_poses + only_scale_pose_index
                                        ]
                                    )
                                else:
                                    subscales = extend_scaleslist[
                                        num_poses : num_poses + default_num_person
                                    ]
                            elif match_scalelist_method == "clamp extend":
                                if (
                                    only_scale_pose_index < default_num_person
                                    and only_scale_pose_index >= -default_num_person
                                ):
                                    subscales[only_scale_pose_index] = scales[-1]
                                else:
                                    subscales = [scales[-1]] * default_num_person
                    else:
                        if (
                            only_scale_pose_index < default_num_person
                            and only_scale_pose_index >= -default_num_person
                        ):
                            subscales[only_scale_pose_index] = scales
                        else:
                            subscales = [scales] * default_num_person

                    scale_lists[i].append(subscales.copy())
            else:
                for i, scales in enumerate(scale_values):
                    if isinstance(scales, (list, tuple)):
                        if len(scales) >= num_imgs + 1:
                            if (
                                only_scale_pose_index < default_num_person
                                and only_scale_pose_index >= -default_num_person
                            ):
                                subscales[only_scale_pose_index] = scales[num_imgs]
                            else:
                                subscales = [scales[num_imgs]] * default_num_person
                        else:
                            if match_scalelist_method == "no extend":
                                subscales = [default_scale] * default_num_person
                            elif match_scalelist_method == "loop extend":
                                extend_scaleslist = scales * math.ceil(
                                    (num_imgs + 1) / len(scales)
                                )
                                if (
                                    only_scale_pose_index < default_num_person
                                    and only_scale_pose_index >= -default_num_person
                                ):
                                    subscales[only_scale_pose_index] = (
                                        extend_scaleslist[num_imgs]
                                    )
                                else:
                                    subscales = [
                                        extend_scaleslist[num_imgs]
                                    ] * default_num_person
                            elif match_scalelist_method == "clamp extend":
                                if (
                                    only_scale_pose_index < default_num_person
                                    and only_scale_pose_index >= -default_num_person
                                ):
                                    subscales[only_scale_pose_index] = scales[-1]
                                else:
                                    subscales = [scales[-1]] * default_num_person
                    else:
                        if (
                            only_scale_pose_index < default_num_person
                            and only_scale_pose_index >= -default_num_person
                        ):
                            subscales[only_scale_pose_index] = scales
                        else:
                            subscales = [scales] * default_num_person

                    scale_lists[i].append(subscales.copy())

            num_poses += default_num_person
            num_imgs += 1
        else:
            # if no people in image
            for i in range(len(scale_values)):
                scale_lists[i].append([default_scale])

    return scale_lists


def pose_normalized(pose_json):
    try:
        if pose_json.startswith("{"):
            pose_json = "[{}]".format(pose_json)
        images = json.loads(pose_json)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing pose JSON in pose_normalized: {e}")
        return "[]"

    if not isinstance(images, list):
        images = [images] if isinstance(images, dict) else []

    for image in images:
        if not isinstance(image, dict) or "people" not in image:
            continue
        figures = image["people"]
        if not figures or not isinstance(figures, list):
            continue

        H = image.get("canvas_height", 1)
        W = image.get("canvas_width", 1)

        # Prevent division by zero
        if H <= 0 or W <= 0:
            continue

        normalized = 0.0
        for figure in figures:
            if not isinstance(figure, dict):
                continue

            if "pose_keypoints_2d" in figure:
                body = figure["pose_keypoints_2d"]
                if body and isinstance(body, list):
                    normalized = max(body)
                    if normalized > 2.0:
                        break
            if "face_keypoints_2d" in figure:
                face = figure["face_keypoints_2d"]
                if face and isinstance(face, list):
                    normalized = max(face)
                    if normalized > 2.0:
                        break
            if "hand_left_keypoints_2d" in figure:
                lhand = figure["hand_left_keypoints_2d"]
                if lhand and isinstance(lhand, list):
                    normalized = max(lhand)
                    if normalized > 2.0:
                        break
            if "hand_right_keypoints_2d" in figure:
                rhand = figure["hand_right_keypoints_2d"]
                if rhand and isinstance(rhand, list):
                    normalized = max(rhand)
                    if normalized > 2.0:
                        break

        if normalized > 2.0:
            for figure in figures:
                if not isinstance(figure, dict):
                    continue

                if "pose_keypoints_2d" in figure:
                    body = figure["pose_keypoints_2d"]
                    if body and isinstance(body, list):
                        for i in range(0, len(body), 3):
                            if i + 1 < len(body):
                                body[i] = body[i] / float(W)
                                body[i + 1] = body[i + 1] / float(H)

                if "face_keypoints_2d" in figure:
                    face = figure["face_keypoints_2d"]
                    if face and isinstance(face, list):
                        for i in range(0, len(face), 3):
                            if i + 1 < len(face):
                                face[i] = face[i] / float(W)
                                face[i + 1] = face[i + 1] / float(H)

                if "hand_left_keypoints_2d" in figure:
                    lhand = figure["hand_left_keypoints_2d"]
                    if lhand and isinstance(lhand, list):
                        for i in range(0, len(lhand), 3):
                            if i + 1 < len(lhand):
                                lhand[i] = lhand[i] / float(W)
                                lhand[i + 1] = lhand[i + 1] / float(H)

                if "hand_right_keypoints_2d" in figure:
                    rhand = figure["hand_right_keypoints_2d"]
                    if rhand and isinstance(rhand, list):
                        for i in range(0, len(rhand), 3):
                            if i + 1 < len(rhand):
                                rhand[i] = rhand[i] / float(W)
                                rhand[i + 1] = rhand[i + 1] / float(H)
    return json.dumps(images)


def scale(point, scale_factor, pivot):
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return point
    if not isinstance(pivot, (list, tuple)) or len(pivot) < 2:
        return point
    return [
        (point[i] - pivot[i]) * scale_factor + pivot[i]
        for i in range(min(len(point), len(pivot)))
    ]


def draw_pose_json(
    pose_json,
    resolution_x,
    show_body,
    show_face,
    show_hands,
    pose_marker_size,
    face_marker_size,
    hand_marker_size,
    hands_scalelist,
    body_scalelist,
    head_scalelist,
    overall_scalelist,
    auto_fix_connections=True,
    head_alignment="eyes",  # New parameter: "eyes" or "neck"
    shift_x=0,  # Add shift_x with default value
    shift_y=0,  # Add shift_y with default value
):
    """
    Draw pose JSON with optional automatic hand and head tracking.

    Args:
        auto_fix_connections (bool): If True, automatically moves hands to follow wrists
                                   and head to follow neck (or eye-based pivot for head)
                                   when different scales are applied.
                                   If False, uses the original behavior.
        head_alignment (str): Specifies the head alignment mode when auto_fix_connections is True.
                              "eyes": Uses eye keypoints as the primary reference for head scaling.
                              "neck": Uses the neck keypoint as the primary reference for head scaling.
    """
    pose_imgs = []
    pose_scaled = []

    if pose_json:
        try:
            if pose_json.startswith("{"):
                pose_json = "[{}]".format(pose_json)
            images = json.loads(pose_json)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing pose JSON in draw_pose_json: {e}")
            return pose_imgs, pose_scaled

        if not isinstance(images, list):
            images = [images] if isinstance(images, dict) else []

        pbar = ProgressBar(len(images))
        for img_idx, image in enumerate(images):
            if not isinstance(image, dict) or "people" not in image:
                pbar.update(1)
                continue

            figures = image["people"]
            if not figures or not isinstance(figures, list):
                pbar.update(1)
                continue

            H = image.get("canvas_height", 768)
            W = image.get("canvas_width", 512)

            # Apply shifts to all keypoints
            if shift_x != 0 or shift_y != 0:
                for figure in figures:
                    if not isinstance(figure, dict):
                        continue
                    for keypoint_type in [
                        "pose_keypoints_2d",
                        "face_keypoints_2d",
                        "hand_left_keypoints_2d",
                        "hand_right_keypoints_2d",
                    ]:
                        if keypoint_type in figure and isinstance(
                            figure[keypoint_type], list
                        ):
                            keypoints = figure[keypoint_type]
                            for i in range(0, len(keypoints), 3):
                                keypoints[i] += shift_x
                                keypoints[i + 1] += shift_y

            # These lists will store data for the current image
            current_image_candidate_parts = []  # For body keypoints used in drawing
            current_image_subset_parts = [[]]  # For body connections used in drawing
            current_image_faces_parts = []  # For face keypoints used in drawing
            current_image_hands_parts = []  # For hand keypoints used in drawing

            openpose_json_people = []  # For the output JSON structure for this image

            for pose_idx, figure in enumerate(figures):
                if not isinstance(figure, dict):
                    continue

                body_scale_val = 1.0
                hands_scale_val = 1.0
                head_scale_val = 1.0
                overall_scale_val = 1.0

                if img_idx < len(body_scalelist) and pose_idx < len(
                    body_scalelist[img_idx]
                ):
                    body_scale_val = body_scalelist[img_idx][pose_idx]
                if img_idx < len(hands_scalelist) and pose_idx < len(
                    hands_scalelist[img_idx]
                ):
                    hands_scale_val = hands_scalelist[img_idx][pose_idx]
                if img_idx < len(head_scalelist) and pose_idx < len(
                    head_scalelist[img_idx]
                ):
                    head_scale_val = head_scalelist[img_idx][pose_idx]
                if img_idx < len(overall_scalelist) and pose_idx < len(
                    overall_scalelist[img_idx]
                ):
                    overall_scale_val = overall_scalelist[img_idx][pose_idx]

                body_orig = figure.get("pose_keypoints_2d", [])
                face_orig = figure.get("face_keypoints_2d", [])
                lhand_orig = figure.get("hand_left_keypoints_2d", [])
                rhand_orig = figure.get("hand_right_keypoints_2d", [])

                body_scaled_output = (
                    body_orig.copy()
                    if body_orig and isinstance(body_orig, list)
                    else []
                )
                face_scaled_output = (
                    face_orig.copy()
                    if face_orig and isinstance(face_orig, list)
                    else []
                )
                lhand_scaled_output = (
                    lhand_orig.copy()
                    if lhand_orig and isinstance(lhand_orig, list)
                    else []
                )
                rhand_scaled_output = (
                    rhand_orig.copy()
                    if rhand_orig and isinstance(rhand_orig, list)
                    else []
                )

                person_candidate_points = []  # Keypoints for this person for drawing

                if auto_fix_connections:
                    original_body_keypoints = (
                        body_orig.copy()
                        if body_orig and isinstance(body_orig, list)
                        else []
                    )

                    original_neck = None
                    if (
                        original_body_keypoints
                        and len(original_body_keypoints) > 1 * 3 + 2
                        and original_body_keypoints[1 * 3 + 2] > 0
                    ):
                        original_neck = [
                            original_body_keypoints[1 * 3],
                            original_body_keypoints[1 * 3 + 1],
                        ]

                    original_left_wrist = None
                    if (
                        original_body_keypoints
                        and len(original_body_keypoints) > 7 * 3 + 2
                        and original_body_keypoints[7 * 3 + 2] > 0
                    ):
                        original_left_wrist = [
                            original_body_keypoints[7 * 3],
                            original_body_keypoints[7 * 3 + 1],
                        ]

                    original_right_wrist = None
                    if (
                        original_body_keypoints
                        and len(original_body_keypoints) > 4 * 3 + 2
                        and original_body_keypoints[4 * 3 + 2] > 0
                    ):
                        original_right_wrist = [
                            original_body_keypoints[4 * 3],
                            original_body_keypoints[4 * 3 + 1],
                        ]

                    overall_pivot = [
                        0.5,
                        0.5,
                    ]  # Default pivot for overall and body scaling

                    # --- Stage 1: Apply body_scale and overall_scale to all body parts ---
                    temp_body_scaled_for_person = []  # For drawing candidate list
                    if original_body_keypoints:
                        for i in range(0, len(original_body_keypoints), 3):
                            if i + 1 < len(original_body_keypoints):
                                p_o = original_body_keypoints[i : i + 2]
                                p_body_s = scale(p_o, body_scale_val, overall_pivot)
                                p_overall_s = scale(
                                    p_body_s, overall_scale_val, overall_pivot
                                )
                                if i + 1 < len(body_scaled_output):
                                    body_scaled_output[i : i + 2] = p_overall_s
                                temp_body_scaled_for_person.append(p_overall_s)
                            elif (
                                i < len(original_body_keypoints)
                            ):  # Handle case where only x is present (should not happen for valid points)
                                temp_body_scaled_for_person.append(
                                    [original_body_keypoints[i]]
                                )

                    # --- Determine Original Head Reference Point (from original_body_keypoints) ---
                    original_head_ref_point = None
                    if head_alignment == "eyes":
                        # Try midpoint of eyes
                        if (
                            original_body_keypoints
                            and len(original_body_keypoints) > 14 * 3 + 2
                            and original_body_keypoints[14 * 3 + 2] > 0
                            and len(original_body_keypoints) > 15 * 3 + 2
                            and original_body_keypoints[15 * 3 + 2] > 0
                        ):
                            original_head_ref_point = [
                                (
                                    original_body_keypoints[14 * 3]
                                    + original_body_keypoints[15 * 3]
                                )
                                / 2,
                                (
                                    original_body_keypoints[14 * 3 + 1]
                                    + original_body_keypoints[15 * 3 + 1]
                                )
                                / 2,
                            ]
                        # Fallback to REye
                        elif (
                            original_body_keypoints
                            and len(original_body_keypoints) > 14 * 3 + 2
                            and original_body_keypoints[14 * 3 + 2] > 0
                        ):
                            original_head_ref_point = [
                                original_body_keypoints[14 * 3],
                                original_body_keypoints[14 * 3 + 1],
                            ]
                        # Fallback to LEye
                        elif (
                            original_body_keypoints
                            and len(original_body_keypoints) > 15 * 3 + 2
                            and original_body_keypoints[15 * 3 + 2] > 0
                        ):
                            original_head_ref_point = [
                                original_body_keypoints[15 * 3],
                                original_body_keypoints[15 * 3 + 1],
                            ]
                        # Fallback to Nose (if eyes not available but alignment is 'eyes')
                        elif (
                            original_body_keypoints
                            and len(original_body_keypoints) > 0 * 3 + 2
                            and original_body_keypoints[0 * 3 + 2] > 0
                        ):
                            original_head_ref_point = [
                                original_body_keypoints[0 * 3],
                                original_body_keypoints[0 * 3 + 1],
                            ]
                        # Final fallback to Neck if eyes/nose not found
                        elif original_neck:
                            original_head_ref_point = original_neck
                    elif head_alignment == "neck":
                        # Prioritize Neck
                        if original_neck:
                            original_head_ref_point = original_neck
                        # Fallback to Nose (if neck not available but alignment is 'neck')
                        elif (
                            original_body_keypoints
                            and len(original_body_keypoints) > 0 * 3 + 2
                            and original_body_keypoints[0 * 3 + 2] > 0
                        ):
                            original_head_ref_point = [
                                original_body_keypoints[0 * 3],
                                original_body_keypoints[0 * 3 + 1],
                            ]
                        # Fallback to midpoint of eyes if neck/nose not available
                        elif (
                            original_body_keypoints
                            and len(original_body_keypoints) > 14 * 3 + 2
                            and original_body_keypoints[14 * 3 + 2] > 0
                            and len(original_body_keypoints) > 15 * 3 + 2
                            and original_body_keypoints[15 * 3 + 2] > 0
                        ):
                            original_head_ref_point = [
                                (
                                    original_body_keypoints[14 * 3]
                                    + original_body_keypoints[15 * 3]
                                )
                                / 2,
                                (
                                    original_body_keypoints[14 * 3 + 1]
                                    + original_body_keypoints[15 * 3 + 1]
                                )
                                / 2,
                            ]

                    if (
                        original_head_ref_point is None
                    ):  # Default if no valid point found based on mode
                        original_head_ref_point = [0.5, 0.5]

                    # --- Determine Head Transform Pivot (from body_scaled_output, i.e., after body/overall scale) ---
                    head_transform_pivot = None
                    if head_alignment == "eyes":
                        # Try midpoint of scaled eyes
                        if (
                            len(body_scaled_output) > 14 * 3 + 2
                            and original_body_keypoints[14 * 3 + 2] > 0
                            and len(body_scaled_output) > 15 * 3 + 2
                            and original_body_keypoints[15 * 3 + 2] > 0
                        ):  # Check original confidence
                            head_transform_pivot = [
                                (
                                    body_scaled_output[14 * 3]
                                    + body_scaled_output[15 * 3]
                                )
                                / 2,
                                (
                                    body_scaled_output[14 * 3 + 1]
                                    + body_scaled_output[15 * 3 + 1]
                                )
                                / 2,
                            ]
                        # Fallback to scaled REye
                        elif (
                            len(body_scaled_output) > 14 * 3 + 2
                            and original_body_keypoints[14 * 3 + 2] > 0
                        ):
                            head_transform_pivot = [
                                body_scaled_output[14 * 3],
                                body_scaled_output[14 * 3 + 1],
                            ]
                        # Fallback to scaled LEye
                        elif (
                            len(body_scaled_output) > 15 * 3 + 2
                            and original_body_keypoints[15 * 3 + 2] > 0
                        ):
                            head_transform_pivot = [
                                body_scaled_output[15 * 3],
                                body_scaled_output[15 * 3 + 1],
                            ]
                        # Fallback to scaled Nose
                        elif (
                            len(body_scaled_output) > 0 * 3 + 2
                            and original_body_keypoints[0 * 3 + 2] > 0
                        ):
                            head_transform_pivot = [
                                body_scaled_output[0 * 3],
                                body_scaled_output[0 * 3 + 1],
                            ]
                        # Fallback to scaled Neck
                        elif (
                            len(body_scaled_output) > 1 * 3 + 2
                            and original_body_keypoints[1 * 3 + 2] > 0
                        ):
                            head_transform_pivot = [
                                body_scaled_output[1 * 3],
                                body_scaled_output[1 * 3 + 1],
                            ]
                    elif head_alignment == "neck":
                        # Prioritize scaled Neck
                        if (
                            len(body_scaled_output) > 1 * 3 + 2
                            and original_body_keypoints[1 * 3 + 2] > 0
                        ):
                            head_transform_pivot = [
                                body_scaled_output[1 * 3],
                                body_scaled_output[1 * 3 + 1],
                            ]
                        # Fallback to scaled Nose
                        elif (
                            len(body_scaled_output) > 0 * 3 + 2
                            and original_body_keypoints[0 * 3 + 2] > 0
                        ):
                            head_transform_pivot = [
                                body_scaled_output[0 * 3],
                                body_scaled_output[0 * 3 + 1],
                            ]
                        # Fallback to midpoint of scaled eyes
                        elif (
                            len(body_scaled_output) > 14 * 3 + 2
                            and original_body_keypoints[14 * 3 + 2] > 0
                            and len(body_scaled_output) > 15 * 3 + 2
                            and original_body_keypoints[15 * 3 + 2] > 0
                        ):
                            head_transform_pivot = [
                                (
                                    body_scaled_output[14 * 3]
                                    + body_scaled_output[15 * 3]
                                )
                                / 2,
                                (
                                    body_scaled_output[14 * 3 + 1]
                                    + body_scaled_output[15 * 3 + 1]
                                )
                                / 2,
                            ]

                    if (
                        head_transform_pivot is None
                    ):  # Default if no valid pivot found based on mode
                        head_transform_pivot = [0.5, 0.5]

                    # --- Apply head scaling to face keypoints ---
                    if face_orig and isinstance(face_orig, list):
                        person_face_points = []
                        for i in range(0, len(face_orig), 3):
                            if i + 1 < len(face_orig) and (
                                i + 2 < len(face_orig) and face_orig[i + 2] > 0
                            ):  # Check confidence
                                p_f_orig = face_orig[i : i + 2]
                                rel_x = p_f_orig[0] - original_head_ref_point[0]
                                rel_y = p_f_orig[1] - original_head_ref_point[1]
                                p_repositioned = [
                                    head_transform_pivot[0] + rel_x,
                                    head_transform_pivot[1] + rel_y,
                                ]
                                p_f_scaled = scale(
                                    p_repositioned, head_scale_val, head_transform_pivot
                                )
                                if i + 1 < len(face_scaled_output):
                                    face_scaled_output[i : i + 2] = p_f_scaled
                                person_face_points.append(p_f_scaled)
                        if person_face_points:
                            current_image_faces_parts.append(person_face_points)

                    # --- Apply head scaling to head-related BODY keypoints (Nose, Eyes, Ears) ---
                    head_keypoint_indices = [0, 14, 15, 16, 17]
                    for head_kp_idx in head_keypoint_indices:
                        kp_flat_idx = head_kp_idx * 3
                        if (
                            original_body_keypoints
                            and len(original_body_keypoints) > kp_flat_idx + 2
                            and original_body_keypoints[kp_flat_idx + 2] > 0
                        ):
                            p_kp_orig = original_body_keypoints[
                                kp_flat_idx : kp_flat_idx + 2
                            ]
                            rel_x = p_kp_orig[0] - original_head_ref_point[0]
                            rel_y = p_kp_orig[1] - original_head_ref_point[1]
                            p_repositioned = [
                                head_transform_pivot[0] + rel_x,
                                head_transform_pivot[1] + rel_y,
                            ]
                            p_kp_scaled = scale(
                                p_repositioned, head_scale_val, head_transform_pivot
                            )

                            if kp_flat_idx + 1 < len(body_scaled_output):
                                body_scaled_output[kp_flat_idx : kp_flat_idx + 2] = (
                                    p_kp_scaled
                                )
                            if head_kp_idx < len(
                                temp_body_scaled_for_person
                            ):  # Update drawing candidate list
                                temp_body_scaled_for_person[head_kp_idx] = p_kp_scaled

                    person_candidate_points.extend(temp_body_scaled_for_person)

                    # --- Apply hand scaling with wrist tracking ---
                    new_left_wrist_pos = (
                        [body_scaled_output[7 * 3], body_scaled_output[7 * 3 + 1]]
                        if len(body_scaled_output) > 7 * 3 + 1 and original_left_wrist
                        else None
                    )
                    new_right_wrist_pos = (
                        [body_scaled_output[4 * 3], body_scaled_output[4 * 3 + 1]]
                        if len(body_scaled_output) > 4 * 3 + 1 and original_right_wrist
                        else None
                    )

                    if (
                        lhand_orig
                        and isinstance(lhand_orig, list)
                        and original_left_wrist
                        and new_left_wrist_pos
                    ):
                        person_lhand_points = []
                        for i in range(0, len(lhand_orig), 3):
                            if i + 1 < len(lhand_orig) and (
                                i + 2 < len(lhand_orig) and lhand_orig[i + 2] > 0
                            ):
                                p_lh_orig = lhand_orig[i : i + 2]
                                rel_x = p_lh_orig[0] - original_left_wrist[0]
                                rel_y = p_lh_orig[1] - original_left_wrist[1]
                                p_repositioned = [
                                    new_left_wrist_pos[0] + rel_x,
                                    new_left_wrist_pos[1] + rel_y,
                                ]
                                p_lh_scaled = scale(
                                    p_repositioned, hands_scale_val, new_left_wrist_pos
                                )
                                # Overall scale is not applied again to hands here, as they follow wrists scaled by overall
                                if i + 1 < len(lhand_scaled_output):
                                    lhand_scaled_output[i : i + 2] = p_lh_scaled
                                person_lhand_points.append(p_lh_scaled)
                        if person_lhand_points:
                            current_image_hands_parts.append(person_lhand_points)

                    if (
                        rhand_orig
                        and isinstance(rhand_orig, list)
                        and original_right_wrist
                        and new_right_wrist_pos
                    ):
                        person_rhand_points = []
                        for i in range(0, len(rhand_orig), 3):
                            if i + 1 < len(rhand_orig) and (
                                i + 2 < len(rhand_orig) and rhand_orig[i + 2] > 0
                            ):
                                p_rh_orig = rhand_orig[i : i + 2]
                                rel_x = p_rh_orig[0] - original_right_wrist[0]
                                rel_y = p_rh_orig[1] - original_right_wrist[1]
                                p_repositioned = [
                                    new_right_wrist_pos[0] + rel_x,
                                    new_right_wrist_pos[1] + rel_y,
                                ]
                                p_rh_scaled = scale(
                                    p_repositioned, hands_scale_val, new_right_wrist_pos
                                )
                                if i + 1 < len(rhand_scaled_output):
                                    rhand_scaled_output[i : i + 2] = p_rh_scaled
                                person_rhand_points.append(p_rh_scaled)
                        if person_rhand_points:
                            current_image_hands_parts.append(person_rhand_points)

                    # Populate subset for drawing
                    current_person_subset = []
                    if original_body_keypoints:
                        start_idx_in_current_image_candidate = len(
                            current_image_candidate_parts
                        )
                        for i in range(0, len(original_body_keypoints), 3):
                            # Check original confidence for subset
                            if (
                                i + 2 < len(original_body_keypoints)
                                and original_body_keypoints[i + 2] > 0
                            ):
                                current_person_subset.append(
                                    start_idx_in_current_image_candidate + (i // 3)
                                )
                            else:
                                current_person_subset.append(-1)

                    if not current_image_subset_parts[0]:  # First person in image
                        current_image_subset_parts[0] = current_person_subset
                    else:  # Subsequent people
                        current_image_subset_parts.append(current_person_subset)

                    current_image_candidate_parts.extend(person_candidate_points)

                else:  # Original non-auto_fix_connections behavior
                    # ... (original logic for non-auto_fix_connections)
                    # This part needs to be carefully preserved or adapted if it also populates
                    # current_image_candidate_parts, current_image_subset_parts etc.
                    # For brevity, assuming this part correctly populates the drawing lists.
                    # The key is that person_candidate_points should be populated for drawing.
                    overall_pivot = [0.5, 0.5]
                    lhand_pivot = [0.25, 0.5]  # Default pivots if not tracked
                    rhand_pivot = [0.75, 0.5]
                    face_pivot = [0.5, 0.5]

                    face_offset = [0, 0]
                    lhand_offset = [0, 0]
                    rhand_offset = [0, 0]

                    candidate_start_idx_for_person = len(current_image_candidate_parts)

                    if (
                        body_orig
                        and isinstance(body_orig, list)
                        and len(body_orig) >= 3
                    ):
                        for i in range(0, len(body_orig), 3):
                            if i + 1 < len(body_orig):
                                p_s = scale(
                                    body_orig[i : i + 2], body_scale_val, overall_pivot
                                )
                                p_s = scale(p_s, overall_scale_val, overall_pivot)
                                if i + 1 < len(body_scaled_output):
                                    body_scaled_output[i : i + 2] = p_s
                                person_candidate_points.append(p_s)

                        # Simplified offset logic from original for non-autofix
                        if (
                            len(person_candidate_points) > 0 and len(body_orig) >= 2
                        ):  # Nose
                            factor = 0.8
                            face_offset = [
                                (person_candidate_points[0][0] - body_orig[0]) * factor,
                                (person_candidate_points[0][1] - body_orig[1]) * factor,
                            ]
                            face_pivot = person_candidate_points[0]
                        if (
                            len(person_candidate_points) > 7
                            and len(body_orig) > 7 * 3 + 2
                        ):  # LWrist
                            lhand_offset = [
                                person_candidate_points[7][0] - body_orig[7 * 3],
                                person_candidate_points[7][1] - body_orig[7 * 3 + 1],
                            ]
                            lhand_pivot = person_candidate_points[7]
                        if (
                            len(person_candidate_points) > 4
                            and len(body_orig) > 4 * 3 + 2
                        ):  # RWrist
                            rhand_offset = [
                                person_candidate_points[4][0] - body_orig[4 * 3],
                                person_candidate_points[4][1] - body_orig[4 * 3 + 1],
                            ]
                            rhand_pivot = person_candidate_points[4]

                        current_person_subset = []
                        for i in range(0, len(body_orig), 3):
                            if i + 2 < len(body_orig) and body_orig[i + 2] > 0:
                                current_person_subset.append(
                                    candidate_start_idx_for_person + (i // 3)
                                )
                            else:
                                current_person_subset.append(-1)
                        if not current_image_subset_parts[0]:
                            current_image_subset_parts[0] = current_person_subset
                        else:
                            current_image_subset_parts.append(current_person_subset)
                        current_image_candidate_parts.extend(person_candidate_points)

                    if face_orig and isinstance(face_orig, list):
                        person_face_points = []
                        for i in range(0, len(face_orig), 3):
                            if i + 1 < len(face_orig):
                                p = face_orig[i : i + 2]
                                p_offset = [
                                    p[0] + face_offset[0],
                                    p[1] + face_offset[1],
                                ]
                                p_s = scale(p_offset, head_scale_val, face_pivot)
                                p_s = scale(p_s, overall_scale_val, overall_pivot)
                                if i + 1 < len(face_scaled_output):
                                    face_scaled_output[i : i + 2] = p_s
                                person_face_points.append(p_s)
                        if person_face_points:
                            current_image_faces_parts.append(person_face_points)

                    if lhand_orig and isinstance(lhand_orig, list):
                        person_lhand_points = []
                        for i in range(0, len(lhand_orig), 3):
                            if i + 1 < len(lhand_orig):
                                p = lhand_orig[i : i + 2]
                                p_offset = [
                                    p[0] + lhand_offset[0],
                                    p[1] + lhand_offset[1],
                                ]
                                p_s = scale(p_offset, hands_scale_val, lhand_pivot)
                                p_s = scale(p_s, overall_scale_val, overall_pivot)
                                if i + 1 < len(lhand_scaled_output):
                                    lhand_scaled_output[i : i + 2] = p_s
                                person_lhand_points.append(p_s)
                        if person_lhand_points:
                            current_image_hands_parts.append(person_lhand_points)

                    if rhand_orig and isinstance(rhand_orig, list):
                        person_rhand_points = []
                        for i in range(0, len(rhand_orig), 3):
                            if i + 1 < len(rhand_orig):
                                p = rhand_orig[i : i + 2]
                                p_offset = [
                                    p[0] + rhand_offset[0],
                                    p[1] + rhand_offset[1],
                                ]
                                p_s = scale(p_offset, hands_scale_val, rhand_pivot)
                                p_s = scale(p_s, overall_scale_val, overall_pivot)
                                if i + 1 < len(rhand_scaled_output):
                                    rhand_scaled_output[i : i + 2] = p_s
                                person_rhand_points.append(p_s)
                        if person_rhand_points:
                            current_image_hands_parts.append(person_rhand_points)
                    # End of original non-auto_fix_connections behavior

                openpose_json_people.append(
                    dict(
                        pose_keypoints_2d=body_scaled_output,
                        face_keypoints_2d=face_scaled_output,
                        hand_left_keypoints_2d=lhand_scaled_output,
                        hand_right_keypoints_2d=rhand_scaled_output,
                    )
                )
            # End of loop over figures (people) in one image

            # Consolidate drawing data for the image
            bodies_for_drawing = dict(
                candidate=current_image_candidate_parts,
                subset=current_image_subset_parts,
            )

            pose_for_drawing = dict(
                bodies=bodies_for_drawing
                if show_body
                else {"candidate": [], "subset": []},
                faces=current_image_faces_parts if show_face else [],
                hands=current_image_hands_parts if show_hands else [],
            )

            W_scaled = resolution_x
            if resolution_x < 64:
                W_scaled = W
            H_scaled = int(H * (W_scaled * 1.0 / W)) if W > 0 else H

            current_openpose_json_output = {
                "people": openpose_json_people,
                "canvas_height": H_scaled,
                "canvas_width": W_scaled,
            }

            pose_img = draw_pose(
                pose_for_drawing,
                H_scaled,
                W_scaled,
                pose_marker_size,
                face_marker_size,
                hand_marker_size,
            )
            pose_imgs.append(pose_img)
            pose_scaled.append(current_openpose_json_output)
            pbar.update(1)
        # End of loop over images
    return pose_imgs, pose_scaled


def draw_pose(pose, H, W, pose_marker_size, face_marker_size, hand_marker_size):
    if not isinstance(pose, dict):
        return np.zeros(shape=(H, W, 3), dtype=np.uint8)

    bodies = pose.get("bodies", {})
    faces = pose.get("faces", [])
    hands = pose.get("hands", [])

    if not isinstance(bodies, dict):
        bodies = {"candidate": [], "subset": []}

    candidate = bodies.get("candidate", [])
    subset = bodies.get("subset", [])

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if len(candidate) > 0:
        canvas = draw_bodypose(canvas, candidate, subset, pose_marker_size)

    if len(hands) > 0:
        canvas = draw_handpose(canvas, hands, hand_marker_size)

    if len(faces) > 0:
        canvas = draw_facepose(canvas, faces, face_marker_size)

    return canvas


def draw_bodypose(canvas, candidate, subset, pose_marker_size):
    if (
        not isinstance(candidate, list)
        or not isinstance(subset, list)
        or len(candidate) == 0
    ):
        return canvas

    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    # stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(min(17, len(limbSeq))):
        for n in range(len(subset)):
            if len(subset[n]) <= max(limbSeq[i]) - 1:
                continue
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index or any(idx >= len(candidate) for idx in index):
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if length > 0:
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)),
                    (int(length / 2), pose_marker_size),
                    int(angle),
                    0,
                    360,
                    1,
                )
                cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            if len(subset[n]) <= i:
                continue
            index = int(subset[n][i])
            if index == -1 or index >= len(candidate):
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(
                canvas,
                (int(x), int(y)),
                pose_marker_size,
                colors[i % len(colors)],
                thickness=-1,
            )

    return canvas


def draw_handpose(canvas, all_hand_peaks, hand_marker_size):
    if not isinstance(all_hand_peaks, list):
        return canvas

    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        if not isinstance(peaks, list) or len(peaks) == 0:
            continue

        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            if max(e) >= len(peaks):
                continue
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    * 255,
                    thickness=1 if hand_marker_size == 0 else hand_marker_size,
                )

        joint_size = 0
        if hand_marker_size < 2:
            joint_size = hand_marker_size + 1
        else:
            joint_size = hand_marker_size + 2
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), joint_size, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks, face_marker_size):
    if not isinstance(all_lmks, list):
        return canvas

    H, W, C = canvas.shape
    for lmks in all_lmks:
        if not isinstance(lmks, list):
            continue
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(
                    canvas, (x, y), face_marker_size, (255, 255, 255), thickness=-1
                )
    return canvas
