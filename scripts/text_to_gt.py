"""
Text-to-GT: Convert free-text accident description to structured GT schema v2.

Usage:
1. LLM-based: Send text to LLM, get structured YAML back
2. Rule-based fallback: Parse known patterns from VLM descriptions
3. Human-in-the-loop: Show structured result, user confirms/edits
"""
import json
import re
import yaml
import sys
import os

# Add LiteLLM support
LITELLM_BASE = "http://192.168.0.199/v1"
LITELLM_KEY = "sk-1234"

GT_SCHEMA_PROMPT = """You are an accident analysis GT annotator. Given a text description of a traffic accident (from dashcam video), extract structured information into the following YAML format.

Rules:
- Only include what is explicitly stated or clearly implied in the description
- Use "unclear" or null for anything not determinable
- Vehicle IDs: "ego" for dashcam vehicle, "car1"/"car2" for others
- Contact zones: front_center, front_left, front_right, left_side, right_side, rear_center, rear_left, rear_right, unclear
- Collision types: rear_end, side_swipe, head_on, t_bone, single_vehicle, unclear
- Causes: rear_end_ego_at_fault, rear_end_other_at_fault, unsafe_lane_change_ego, cut_in_other, solo_collision, sudden_stop_front, side_collision_intersection, observed_accident
- ego_action: approaching, lane_changing, braking, stationary, evading, unclear
- vehicle_b_behavior: stationary, decelerating, lane_changing, approaching_from_side, approaching_from_rear, unclear

Output ONLY valid YAML, no explanation.

Schema:
```yaml
involvement: ego/observed/unclear
n_collisions: N
lane_info:
  available: true/false
  road_type: highway/urban/intersection/parking/rural/unclear
  n_lanes_visible: N or null
  ego_lane_at_start: N or null
collisions:
  - collision_id: 1
    time_sec: N.N or null
    vehicle_a: "ego"/"car1"/...
    vehicle_a_type: car/truck/bus/motorcycle/bicycle/pedestrian/structure/unknown
    contact_zone_a: 8-zone
    lane_a: N or null
    lane_change_a: true/false
    vehicle_b: "car1"/"guardrail"/...
    vehicle_b_type: car/truck/bus/motorcycle/bicycle/pedestrian/structure/unknown
    contact_zone_b: 8-zone
    lane_b: N or null
    lane_change_b: true/false
    collision_type: type
    severity_visible: minor/moderate/severe/unclear
    pre_collision:
      ego_action: action
      vehicle_b_behavior: behavior
cause: cause_type
camera_shake_visible: true/false
```"""


def text_to_gt_llm(description, stem=None, collision_time=None, model="qwen-vl-heavy"):
    """Use LLM to convert text description to GT schema."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LITELLM_BASE, api_key=LITELLM_KEY)

        user_msg = f"Accident video: {stem}\n" if stem else ""
        if collision_time:
            user_msg += f"Collision time: {collision_time}s\n"
        user_msg += f"\nDescription:\n{description}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GT_SCHEMA_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()

        # Extract YAML from response
        if "```yaml" in content:
            content = content.split("```yaml")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        gt = yaml.safe_load(content)
        if stem:
            gt["stem"] = stem
        return gt

    except Exception as e:
        print(f"LLM error: {e}")
        return None


def text_to_gt_rules(description, stem=None, collision_time=None,
                      vlm_subject=None, vlm_impacts=None, vlm_cause=None):
    """Rule-based fallback: parse known patterns from VLM descriptions."""
    desc_lower = description.lower() if description else ""

    # Involvement
    involvement = vlm_subject or "unclear"
    if any(w in desc_lower for w in ["관찰", "직접 관여하지 않", "observed"]):
        involvement = "observed"
    elif any(w in desc_lower for w in ["블랙박스 차량이 충돌", "ego", "급정지"]):
        involvement = "ego"

    # Road type
    road_type = "unclear"
    if any(w in desc_lower for w in ["고속도로", "highway"]):
        road_type = "highway"
    elif any(w in desc_lower for w in ["교차로", "intersection"]):
        road_type = "intersection"
    elif any(w in desc_lower for w in ["시내", "도심", "urban"]):
        road_type = "urban"

    # Camera shake
    camera_shake = any(w in desc_lower for w in ["흔들", "충격", "카메라", "shake"])

    # Build collisions from vlm_impacts
    collisions = []
    if vlm_impacts:
        for i, imp in enumerate(vlm_impacts):
            collision = {
                "collision_id": i + 1,
                "time_sec": collision_time,
                "vehicle_a": "ego" if involvement == "ego" else imp.get("vehicle", f"car{i+1}"),
                "vehicle_a_type": "car",
                "contact_zone_a": _normalize_zone(imp.get("zone", "unclear")),
                "lane_a": None,
                "lane_change_a": "차선 변경" in desc_lower,
                "vehicle_b": imp.get("vehicle", f"car{i+1}"),
                "vehicle_b_type": _detect_vehicle_type(imp.get("vehicle", ""), desc_lower),
                "contact_zone_b": "unclear",
                "lane_b": None,
                "lane_change_b": False,
                "collision_type": _infer_collision_type(imp.get("zone", ""), desc_lower),
                "severity_visible": "unclear",
                "pre_collision": {
                    "ego_action": _infer_ego_action(desc_lower),
                    "vehicle_b_behavior": _infer_other_behavior(desc_lower),
                },
            }
            collisions.append(collision)

    gt = {
        "stem": stem,
        "involvement": involvement,
        "n_collisions": len(collisions),
        "lane_info": {
            "available": "차선" in desc_lower or "차로" in desc_lower,
            "road_type": road_type,
            "n_lanes_visible": None,
            "ego_lane_at_start": None,
        },
        "collisions": collisions,
        "cause": vlm_cause or "unclear",
        "camera_shake_visible": camera_shake,
    }
    return gt


def _normalize_zone(zone):
    zone_map = {
        "front": "front_center", "front_center": "front_center",
        "front_left": "front_left", "front_right": "front_right",
        "left": "left_side", "left_side": "left_side",
        "right": "right_side", "right_side": "right_side",
        "rear": "rear_center", "rear_center": "rear_center",
        "rear_left": "rear_left", "rear_right": "rear_right",
    }
    return zone_map.get(zone, "unclear")


def _detect_vehicle_type(vehicle_str, desc):
    if "truck" in desc or "트럭" in desc: return "truck"
    if "bus" in desc or "버스" in desc: return "bus"
    if "motorcycle" in desc or "오토바이" in desc: return "motorcycle"
    if "pedestrian" in desc or "보행자" in desc: return "pedestrian"
    if "guardrail" in vehicle_str or "가드레일" in desc: return "structure"
    if "wall" in vehicle_str or "벽" in desc: return "structure"
    return "car"


def _infer_collision_type(zone, desc):
    if "추돌" in desc or "rear" in zone: return "rear_end"
    if "측면" in desc or "side" in zone: return "side_swipe"
    if "정면" in desc or "head" in desc: return "head_on"
    if "교차" in desc or "t_bone" in desc: return "t_bone"
    if "가드레일" in desc or "전복" in desc: return "single_vehicle"
    return "unclear"


def _infer_ego_action(desc):
    if "급정지" in desc or "정지" in desc: return "braking"
    if "차선 변경" in desc: return "lane_changing"
    if "회피" in desc: return "evading"
    if "접근" in desc or "가까워" in desc: return "approaching"
    if "정차" in desc or "멈" in desc: return "stationary"
    return "unclear"


def _infer_other_behavior(desc):
    if "급정거" in desc or "급정지" in desc: return "decelerating"
    if "끼어들" in desc or "차선 변경" in desc: return "lane_changing"
    if "정지" in desc or "멈춰" in desc: return "stationary"
    if "측면에서" in desc or "교차" in desc: return "approaching_from_side"
    if "뒤에서" in desc: return "approaching_from_rear"
    return "unclear"


def demo():
    """Demo: convert existing VLM descriptions to GT schema."""
    with open("/home/ktl/projects/accident_analysis/collision_analysis_results.json") as f:
        coll = json.load(f)
    with open("/home/ktl/projects/accident_analysis/cause_classification_results.json") as f:
        causes = json.load(f)

    # Test cases
    test_stems = [
        "20170911_SEQ_S_F_D_1_H_1_9",  # observed, multi-impact
        "20170313_SEQ_S_F_D_1_O_1_9",  # solo collision
        "20170126_SEQ_S_R_N_1_O_1_6",  # rear_end ego
    ]

    for stem in test_stems:
        c = coll[stem]
        ca = causes[stem]
        desc = c["verified"]["description"]
        impacts = c["verified"]["impacts"]
        subject = c["verified"]["accident_subject"]
        cause = ca["cause"]
        col_time = c["collision_time"]

        print(f"\n{'='*60}")
        print(f"Video: {stem}")
        print(f"Description: {desc}")
        print(f"\n--- Rule-based GT ---")
        gt = text_to_gt_rules(desc, stem, col_time, subject, impacts, cause)
        print(yaml.dump(gt, allow_unicode=True, default_flow_style=False))


if __name__ == "__main__":
    demo()
