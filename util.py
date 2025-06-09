import math
import json
import numpy as np
import matplotlib
import cv2
from comfy.utils import ProgressBar
from typing import List, Dict

eps = 0.01

def extend_scalelist(scalelist_behavior, pose_json, hands_scale, body_scale, head_scale, overall_scale, match_scalelist_method, only_scale_pose_index) -> List[list]:
    try:
        if pose_json.startswith('{'):
            pose_json = '[{}]'.format(pose_json)
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
        if 'people' in img and img['people'] and isinstance(img['people'], list):
            default_scale = 1.0
            default_num_person = len(img['people'])
            subscales = [default_scale]*default_num_person
            if scalelist_behavior == 'poses':
                for i, scales in enumerate(scale_values):
                    if isinstance(scales, (list, tuple)):
                        if len(scales) >= num_poses + default_num_person:
                            if only_scale_pose_index<default_num_person and only_scale_pose_index >= -default_num_person:
                                subscales[only_scale_pose_index] = scales[num_poses + only_scale_pose_index]
                            else:
                                subscales = scales[num_poses:num_poses + default_num_person]
                        else:
                            if match_scalelist_method == 'no extend':
                                subscales = [default_scale]*default_num_person
                            elif match_scalelist_method == 'loop extend':
                                extend_scaleslist = scales*math.ceil((num_poses+default_num_person) / len(scales))
                                if only_scale_pose_index<default_num_person and only_scale_pose_index >= -default_num_person:
                                    subscales[only_scale_pose_index] = extend_scaleslist[num_poses + only_scale_pose_index]
                                else:
                                    subscales = extend_scaleslist[num_poses:num_poses + default_num_person]
                            elif match_scalelist_method == 'clamp extend':
                                if only_scale_pose_index<default_num_person and only_scale_pose_index >= -default_num_person:
                                    subscales[only_scale_pose_index] = scales[-1]
                                else:
                                    subscales = [scales[-1]] * default_num_person
                    else:
                        if only_scale_pose_index<default_num_person and only_scale_pose_index >= -default_num_person:
                            subscales[only_scale_pose_index] = scales
                        else:
                            subscales = [scales] * default_num_person

                    scale_lists[i].append(subscales.copy())
            else:
                for i, scales in enumerate(scale_values):
                    if isinstance(scales, (list, tuple)):
                        if len(scales) >= num_imgs + 1:
                            if only_scale_pose_index<default_num_person and only_scale_pose_index >= -default_num_person:
                                subscales[only_scale_pose_index] = scales[num_imgs]
                            else:
                                subscales = [scales[num_imgs]]*default_num_person
                        else:
                            if match_scalelist_method == 'no extend':
                                subscales = [default_scale]*default_num_person
                            elif match_scalelist_method == 'loop extend':
                                extend_scaleslist = scales*math.ceil((num_imgs+1) / len(scales))
                                if only_scale_pose_index<default_num_person and only_scale_pose_index >= -default_num_person:
                                    subscales[only_scale_pose_index] = extend_scaleslist[num_imgs]
                                else:
                                    subscales = [extend_scaleslist[num_imgs]]*default_num_person
                            elif match_scalelist_method == 'clamp extend':
                                if only_scale_pose_index<default_num_person and only_scale_pose_index >= -default_num_person:
                                    subscales[only_scale_pose_index] = scales[-1]
                                else:
                                    subscales = [scales[-1]] * default_num_person
                    else:
                        if only_scale_pose_index<default_num_person and only_scale_pose_index >= -default_num_person:
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
        if pose_json.startswith('{'):
            pose_json = '[{}]'.format(pose_json)
        images = json.loads(pose_json)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing pose JSON in pose_normalized: {e}")
        return "[]"
    
    if not isinstance(images, list):
        images = [images] if isinstance(images, dict) else []
    
    for image in images:
        if not isinstance(image, dict) or 'people' not in image:
            continue
        figures = image['people']
        if not figures or not isinstance(figures, list):
            continue
            
        H = image.get('canvas_height', 1)
        W = image.get('canvas_width', 1)
        
        # Prevent division by zero
        if H <= 0 or W <= 0:
            continue
            
        normalized = 0.0
        for figure in figures:
            if not isinstance(figure, dict):
                continue
                
            if 'pose_keypoints_2d' in figure:
                body = figure['pose_keypoints_2d']
                if body and isinstance(body, list):
                    normalized = max(body)
                    if normalized > 2.0:
                        break
            if 'face_keypoints_2d' in figure:
                face = figure['face_keypoints_2d']
                if face and isinstance(face, list):
                    normalized = max(face)
                    if normalized > 2.0:
                        break
            if 'hand_left_keypoints_2d' in figure:
                lhand = figure['hand_left_keypoints_2d']
                if lhand and isinstance(lhand, list):
                    normalized = max(lhand)
                    if normalized > 2.0:
                        break
            if 'hand_right_keypoints_2d' in figure:
                rhand = figure['hand_right_keypoints_2d']
                if rhand and isinstance(rhand, list):
                    normalized = max(rhand)
                    if normalized > 2.0:
                        break
                        
        if normalized > 2.0:
            for figure in figures:
                if not isinstance(figure, dict):
                    continue
                    
                if 'pose_keypoints_2d' in figure:
                    body = figure['pose_keypoints_2d']
                    if body and isinstance(body, list):
                        for i in range(0, len(body), 3):
                            if i + 1 < len(body):
                                body[i] = body[i] / float(W)
                                body[i+1] = body[i+1] / float(H)
                                
                if 'face_keypoints_2d' in figure:
                    face = figure['face_keypoints_2d']
                    if face and isinstance(face, list):
                        for i in range(0, len(face), 3):
                            if i + 1 < len(face):
                                face[i] = face[i] / float(W)
                                face[i+1] = face[i+1] / float(H)
                                
                if 'hand_left_keypoints_2d' in figure:
                    lhand = figure['hand_left_keypoints_2d']
                    if lhand and isinstance(lhand, list):
                        for i in range(0, len(lhand), 3):
                            if i + 1 < len(lhand):
                                lhand[i] = lhand[i] / float(W)
                                lhand[i+1] = lhand[i+1] / float(H)
                                
                if 'hand_right_keypoints_2d' in figure:
                    rhand = figure['hand_right_keypoints_2d']
                    if rhand and isinstance(rhand, list):
                        for i in range(0, len(rhand), 3):
                            if i + 1 < len(rhand):
                                rhand[i] = rhand[i] / float(W)
                                rhand[i+1] = rhand[i+1] / float(H)
    return json.dumps(images)

def scale(point, scale_factor, pivot):
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return point
    if not isinstance(pivot, (list, tuple)) or len(pivot) < 2:
        return point
    return [(point[i] - pivot[i])*scale_factor + pivot[i] for i in range(min(len(point), len(pivot)))]

def draw_pose_json(pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist, auto_fix_connections=True):
    """
    Draw pose JSON with optional automatic hand and head tracking.
    
    Args:
        auto_fix_connections (bool): If True, automatically moves hands to follow wrists 
                                   and head to follow neck when different scales are applied.
                                   If False, uses the original behavior.
    """
    pose_imgs = []
    pose_scaled = []

    if pose_json:
        try:
            if pose_json.startswith('{'):
                pose_json = '[{}]'.format(pose_json)
            images = json.loads(pose_json)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing pose JSON in draw_pose_json: {e}")
            return pose_imgs, pose_scaled
            
        if not isinstance(images, list):
            images = [images] if isinstance(images, dict) else []
            
        pbar = ProgressBar(len(images))
        for img_idx, image in enumerate(images):
            if not isinstance(image, dict) or 'people' not in image:
                pbar.update(1)
                continue
                
            figures = image['people']
            if not figures or not isinstance(figures, list):
                pbar.update(1)
                continue
                
            H = image.get('canvas_height', 768)
            W = image.get('canvas_width', 512)

            bodies = []
            candidate = []
            subset = [[]]
            faces = []
            hands = []

            openpose_json = []
            for pose_idx, figure in enumerate(figures):
                if not isinstance(figure, dict):
                    continue
                    
                # Safe access to scale lists with bounds checking
                body_scale = 1.0
                hands_scale = 1.0
                head_scale = 1.0
                overall_scale = 1.0
                
                if (img_idx < len(body_scalelist) and pose_idx < len(body_scalelist[img_idx])):
                    body_scale = body_scalelist[img_idx][pose_idx]
                if (img_idx < len(hands_scalelist) and pose_idx < len(hands_scalelist[img_idx])):
                    hands_scale = hands_scalelist[img_idx][pose_idx]
                if (img_idx < len(head_scalelist) and pose_idx < len(head_scalelist[img_idx])):
                    head_scale = head_scalelist[img_idx][pose_idx]
                if (img_idx < len(overall_scalelist) and pose_idx < len(overall_scalelist[img_idx])):
                    overall_scale = overall_scalelist[img_idx][pose_idx]
                
                body = []
                face = []
                lhand = []
                rhand = []
                
                if 'pose_keypoints_2d' in figure:
                    body = figure['pose_keypoints_2d']
                    body_scaled = body.copy() if body and isinstance(body, list) else []
                if 'face_keypoints_2d' in figure:
                    face = figure['face_keypoints_2d']
                    face_scaled = face.copy() if face and isinstance(face, list) else []
                if 'hand_left_keypoints_2d' in figure:
                    lhand = figure['hand_left_keypoints_2d']
                    lhand_scaled = lhand.copy() if lhand and isinstance(lhand, list) else []
                if 'hand_right_keypoints_2d' in figure:
                    rhand = figure['hand_right_keypoints_2d']
                    rhand_scaled = rhand.copy() if rhand and isinstance(rhand, list) else []

                if auto_fix_connections:
                    # NEW TRACKING SYSTEM: Store original positions before scaling
                    original_neck = None
                    original_left_wrist = None
                    original_right_wrist = None
                    
                    if body and isinstance(body, list) and len(body) >= 3:
                        # Store original neck position (index 1 = neck)
                        if len(body) > 1*3 + 2 and body[1*3+2] > 0:
                            original_neck = [body[1*3], body[1*3+1]]
                        
                        # Store original wrist positions
                        if len(body) > 7*3 + 2 and body[7*3+2] > 0:  # left wrist
                            original_left_wrist = [body[7*3], body[7*3+1]]
                        if len(body) > 4*3 + 2 and body[4*3+2] > 0:  # right wrist
                            original_right_wrist = [body[4*3], body[4*3+1]]

                    overall_pivot = [0.5, 0.5]

                    # Apply body scaling first
                    if body and isinstance(body, list) and len(body) >= 3:
                        candidate_start_idx = len(candidate)

                        for i in range(0, len(body), 3):
                            if i + 1 < len(body):
                                p_scaled = scale(body[i:i+2], body_scale, overall_pivot)
                                p_scaled = scale(p_scaled, overall_scale, overall_pivot)
                                if i + 1 < len(body_scaled):
                                    body_scaled[i:i+2] = p_scaled
                                candidate.append(p_scaled)

                        # Calculate how much neck and wrists moved after body scaling
                        neck_offset = [0, 0]
                        left_wrist_offset = [0, 0]
                        right_wrist_offset = [0, 0]
                        
                        if original_neck and len(body_scaled) > 1*3 + 1:
                            new_neck = [body_scaled[1*3], body_scaled[1*3+1]]
                            neck_offset = [new_neck[0] - original_neck[0], new_neck[1] - original_neck[1]]
                        
                        if original_left_wrist and len(body_scaled) > 7*3 + 1:
                            new_left_wrist = [body_scaled[7*3], body_scaled[7*3+1]]
                            left_wrist_offset = [new_left_wrist[0] - original_left_wrist[0], new_left_wrist[1] - original_left_wrist[1]]
                        
                        if original_right_wrist and len(body_scaled) > 4*3 + 1:
                            new_right_wrist = [body_scaled[4*3], body_scaled[4*3+1]]
                            right_wrist_offset = [new_right_wrist[0] - original_right_wrist[0], new_right_wrist[1] - original_right_wrist[1]]

                        # Apply head scaling with neck tracking
                        if face and isinstance(face, list):
                            f = []
                            # Use the new neck position as pivot for face scaling
                            face_pivot = [body_scaled[1*3], body_scaled[1*3+1]] if len(body_scaled) > 1*3 + 1 else [0.5, 0.5]
                            
                            for i in range(0, len(face), 3):
                                if i + 1 < len(face):
                                    p = face[i:i+2]
                                    # Calculate the relative position from original neck to face point
                                    if original_neck:
                                        relative_x = p[0] - original_neck[0]
                                        relative_y = p[1] - original_neck[1]
                                        # Apply the relative position to the new neck location
                                        p_relative_to_new_neck = [face_pivot[0] + relative_x, face_pivot[1] + relative_y]
                                    else:
                                        p_relative_to_new_neck = p
                                    
                                    # Then apply head scaling around the new neck position
                                    p_scaled = scale(p_relative_to_new_neck, head_scale, face_pivot)
                                    p_scaled = scale(p_scaled, overall_scale, overall_pivot)
                                    if i + 1 < len(face_scaled):
                                        face_scaled[i:i+2] = p_scaled
                                    f.append(p_scaled)
                            faces.append(f)

                        # Apply hand scaling with wrist tracking
                        if lhand and isinstance(lhand, list):
                            lh = []
                            # Use the new left wrist position as pivot for left hand scaling
                            lhand_pivot = [body_scaled[7*3], body_scaled[7*3+1]] if len(body_scaled) > 7*3 + 1 else [0.25, 0.5]
                            
                            for i in range(0, len(lhand), 3):
                                if i + 1 < len(lhand):
                                    p = lhand[i:i+2]
                                    # Calculate the relative position from original wrist to hand point
                                    if original_left_wrist:
                                        relative_x = p[0] - original_left_wrist[0]
                                        relative_y = p[1] - original_left_wrist[1]
                                        # Apply the relative position to the new wrist location
                                        p_relative_to_new_wrist = [lhand_pivot[0] + relative_x, lhand_pivot[1] + relative_y]
                                    else:
                                        p_relative_to_new_wrist = p
                                    
                                    # Then apply hand scaling around the new wrist position
                                    p_scaled = scale(p_relative_to_new_wrist, hands_scale, lhand_pivot)
                                    p_scaled = scale(p_scaled, overall_scale, overall_pivot)
                                    if i + 1 < len(lhand_scaled):
                                        lhand_scaled[i:i+2] = p_scaled
                                    lh.append(p_scaled)
                            hands.append(lh)

                        if rhand and isinstance(rhand, list):
                            rh = []
                            # Use the new right wrist position as pivot for right hand scaling
                            rhand_pivot = [body_scaled[4*3], body_scaled[4*3+1]] if len(body_scaled) > 4*3 + 1 else [0.75, 0.5]
                            
                            for i in range(0, len(rhand), 3):
                                if i + 1 < len(rhand):
                                    p = rhand[i:i+2]
                                    # Calculate the relative position from original wrist to hand point
                                    if original_right_wrist:
                                        relative_x = p[0] - original_right_wrist[0]
                                        relative_y = p[1] - original_right_wrist[1]
                                        # Apply the relative position to the new wrist location
                                        p_relative_to_new_wrist = [rhand_pivot[0] + relative_x, rhand_pivot[1] + relative_y]
                                    else:
                                        p_relative_to_new_wrist = p
                                    
                                    # Then apply hand scaling around the new wrist position
                                    p_scaled = scale(p_relative_to_new_wrist, hands_scale, rhand_pivot)
                                    p_scaled = scale(p_scaled, overall_scale, overall_pivot)
                                    if i + 1 < len(rhand_scaled):
                                        rhand_scaled[i:i+2] = p_scaled
                                    rh.append(p_scaled)
                            hands.append(rh)

                        if not subset[0]:
                            subset[0].extend([candidate_start_idx+(i//3) if i+2 < len(body) and body[i+2]>0 else -1 for i in range(0,len(body),3)])
                        else:
                            new_subset = [candidate_start_idx+(i//3) if i+2 < len(body) and body[i+2]>0 else -1 for i in range(0,len(body),3)]
                            subset.append(new_subset)

                else:
                    # ORIGINAL SYSTEM: Keep the old behavior for backward compatibility
                    face_offset = [0, 0]
                    lhand_offset = [0, 0]
                    rhand_offset = [0, 0]

                    overall_pivot = [0.5, 0.5]
                    lhand_pivot = [0.25, 0.5]
                    rhand_pivot = [0.75, 0.5]
                    face_pivot = [0.5, 0.5]

                    if body and isinstance(body, list) and len(body) >= 3:
                        candidate_start_idx = len(candidate)

                        for i in range(0, len(body), 3):
                            if i + 1 < len(body):
                                p_scaled = scale(body[i:i+2], body_scale, overall_pivot)
                                p_scaled = scale(p_scaled, overall_scale, overall_pivot)
                                if i + 1 < len(body_scaled):
                                    body_scaled[i:i+2] = p_scaled
                                candidate.append(p_scaled)

                        figure_head_idx = candidate_start_idx
                        if figure_head_idx < len(candidate) and len(body) >= 2:
                            factor = 0.8
                            face_offset = [(candidate[figure_head_idx][0] - body[0])*factor, (candidate[figure_head_idx][1] - body[1])*factor]
                            face_pivot = candidate[figure_head_idx]

                        wrist_left_idx = candidate_start_idx + 7
                        wrist_right_idx = candidate_start_idx + 4

                        if wrist_left_idx < len(candidate) and len(body) > 22:
                            lhand_offset = [candidate[wrist_left_idx][0] - body[21], candidate[wrist_left_idx][1] - body[22]]
                            lhand_pivot = candidate[wrist_left_idx]

                        if wrist_right_idx < len(candidate) and len(body) > 13:
                            rhand_offset = [candidate[wrist_right_idx][0] - body[12], candidate[wrist_right_idx][1] - body[13]]
                            rhand_pivot = candidate[wrist_right_idx]

                        if not subset[0]:
                            subset[0].extend([candidate_start_idx+(i//3) if i+2 < len(body) and body[i+2]>0 else -1 for i in range(0,len(body),3)])
                        else:
                            new_subset = [candidate_start_idx+(i//3) if i+2 < len(body) and body[i+2]>0 else -1 for i in range(0,len(body),3)]
                            subset.append(new_subset)

                    if face and isinstance(face, list):
                        f = []
                        for i in range(0, len(face), 3):
                            if i + 1 < len(face):
                                p = face[i:i+2]
                                p_offset = [p[0] + face_offset[0], p[1] + face_offset[1]]
                                p_scaled = scale(p_offset, head_scale, face_pivot)
                                p_scaled = scale(p_scaled, overall_scale, overall_pivot)
                                if i + 1 < len(face_scaled):
                                    face_scaled[i:i+2] = p_scaled
                                f.append(p_scaled)
                        faces.append(f)

                    if lhand and isinstance(lhand, list):
                        lh = []
                        for i in range(0, len(lhand), 3):
                            if i + 1 < len(lhand):
                                p = lhand[i:i+2]
                                p_offset = [p[0] + lhand_offset[0], p[1] + lhand_offset[1]]
                                p_scaled = scale(p_offset, hands_scale, lhand_pivot)
                                p_scaled = scale(p_scaled, overall_scale, overall_pivot)
                                if i + 1 < len(lhand_scaled):
                                    lhand_scaled[i:i+2] = p_scaled
                                lh.append(p_scaled)
                        hands.append(lh)

                    if rhand and isinstance(rhand, list):
                        rh = []
                        for i in range(0, len(rhand), 3):
                            if i + 1 < len(rhand):
                                p = rhand[i:i+2]
                                p_offset = [p[0] + rhand_offset[0], p[1] + rhand_offset[1]]
                                p_scaled = scale(p_offset, hands_scale, rhand_pivot)
                                p_scaled = scale(p_scaled, overall_scale, overall_pivot)
                                if i + 1 < len(rhand_scaled):
                                    rhand_scaled[i:i+2] = p_scaled
                                rh.append(p_scaled)
                        hands.append(rh)
                    
                openpose_json.append(dict(pose_keypoints_2d=body_scaled, face_keypoints_2d=face_scaled, hand_left_keypoints_2d=lhand_scaled, hand_right_keypoints_2d=rhand_scaled))

            bodies = dict(candidate=candidate, subset=subset)
            pose = dict(bodies=bodies, faces=faces, hands=hands)
            pose = dict(bodies=bodies if show_body else {'candidate':[], 'subset':[]}, faces=faces if show_face else [], hands=hands if show_hands else [])

            W_scaled = resolution_x
            if resolution_x < 64:
                W_scaled = W
            H_scaled = int(H*(W_scaled*1.0/W)) if W > 0 else H
            openpose_json = {
                            'people': openpose_json,
                            'canvas_height': H_scaled,
                            'canvas_width': W_scaled,
                            }

            pose_img = draw_pose(pose, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)
            pose_imgs.append(pose_img)
            pose_scaled.append(openpose_json)
            pbar.update(1)

    return pose_imgs, pose_scaled

def draw_pose(pose, H, W, pose_marker_size, face_marker_size, hand_marker_size):
    if not isinstance(pose, dict):
        return np.zeros(shape=(H, W, 3), dtype=np.uint8)
        
    bodies = pose.get('bodies', {})
    faces = pose.get('faces', [])
    hands = pose.get('hands', [])
    
    if not isinstance(bodies, dict):
        bodies = {'candidate': [], 'subset': []}
    
    candidate = bodies.get('candidate', [])
    subset = bodies.get('subset', [])
    
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if len(candidate) > 0:
        canvas = draw_bodypose(canvas, candidate, subset, pose_marker_size)

    if len(hands) > 0:
        canvas = draw_handpose(canvas, hands, hand_marker_size)

    if len(faces) > 0:
        canvas = draw_facepose(canvas, faces, face_marker_size)

    return canvas

def draw_bodypose(canvas, candidate, subset, pose_marker_size):
    if not isinstance(candidate, list) or not isinstance(subset, list) or len(candidate) == 0:
        return canvas
        
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    # stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

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
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), pose_marker_size), int(angle), 0, 360, 1)
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
            cv2.circle(canvas, (int(x), int(y)), pose_marker_size, colors[i % len(colors)], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, hand_marker_size):
    if not isinstance(all_hand_peaks, list):
        return canvas
        
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

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
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=1 if hand_marker_size == 0 else hand_marker_size)

        joint_size=0
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
                cv2.circle(canvas, (x, y), face_marker_size, (255, 255, 255), thickness=-1)
    return canvas