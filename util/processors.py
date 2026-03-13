"""
processors.py

센서별 MAT 변환 프로세서 및 토픽 상수.
BagReader / Synchronizer / 헬퍼는 utils.py 참조.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import BagReader, Synchronizer, _fval, _extract_flat

# ─────────────────────────────────────────────────────────────
# 토픽 / 필드 상수
# ─────────────────────────────────────────────────────────────

_GNSS_TOPICS: Dict[str, str] = {
    "GNSS":               "/gnss",
    "Filter_Positionlla": "/filter/positionlla",
    "Filter_Velocity":    "/filter/velocity",
    "IMU_data":           "/imu/data",
    "IMU_time_ref":       "/imu/time_ref",
}

_ME_LANE_TOPICS: Dict[str, str] = {
    "Add_info_1":        "/MELaneAddinfo1",
    "Additional_Data_1": "/MELaneAdditionalData1",
    "Additional_Data_2": "/MELaneAdditionalData2",
    "Additional_Data_3": "/MELaneAdditionalData3",
    "Left_Lane_A":       "/MELeftLaneA",
    "Left_Lane_B":       "/MELeftLaneB",
    "Right_Lane_A":      "/MERightLaneA",
    "Right_Lane_B":      "/MERightLaneB",
}

# (MATLAB 출력명, ROS2 snake_case 속성명)
_RADAR_FIELDS: List[Tuple[str, str]] = [
    ("TrackStatus",       "track_status"),
    ("TrackAngle",        "track_angle"),
    ("TrackRange",        "track_range"),
    ("TrackWidth",        "track_width"),
    ("TrackRollingCount", "track_rolling_count"),
    ("TrackRangeAccel",   "track_range_accel"),
    ("TrackRangeRate",    "track_range_rate"),
]

_FT3_FIELD_MAP: List[Tuple[str, str]] = [
    ("Measure_Rel_Pos_Y",              "measure_rel_pos_y"),
    ("Measure_Rel_Pos_X",              "measure_rel_pos_x"),
    ("Measure_Rel_Vel_Y",              "measure_rel_vel_y"),
    ("Measure_Rel_Vel_X",              "measure_rel_vel_x"),
    ("Measure_Abs_Vel_Y",              "measure_abs_vel_y"),
    ("Measure_Abs_Vel_X",              "measure_abs_vel_x"),
    ("Measure_Abs_Vel",                "measure_abs_vel"),
    ("Measure_Rel_Acc_Y",              "measure_rel_acc_y"),
    ("Measure_Rel_Acc_X",              "measure_rel_acc_x"),
    ("Measure_Width",                  "measure_width"),
    ("Measure_Length",                 "measure_length"),
    ("Measure_Heading_Angle",          "measure_heading_angle"),
    ("Association_Fused_FVT_ID",       "association_fused_fvt_id"),
    ("Association_Fused_LDT_ID",       "association_fused_ldt_id"),
    ("Association_Fused_FRT_ID",       "association_fused_frt_id"),
    ("Association_Asso_Hist_FVT_ID",   "association_asso_hist_fvt_id"),
    ("Association_Asso_Hist_LDT_ID",   "association_asso_hist_ldt_id"),
    ("Association_Asso_Hist_FRT_ID",   "association_asso_hist_frt_id"),
    ("Association_Asso_Creation_Flag", "association_asso_creation_flag"),
    ("Association_Asso_Assoc_Flag",    "association_asso_association_flag"),
    ("Motion_Attribute_Abs_Vel",       "motion_attribute_abs_vel"),
    ("Motion_Attribute_Motion",        "motion_attribute_motion"),
    ("Shape_Attribute_shape",          "shape_attribute_shape"),
    ("Tracking_ID",                    "tracking_id"),
    ("Tracking_Rel_Pos_Y",             "tracking_rel_pos_y"),
    ("Tracking_Rel_Pos_X",             "tracking_rel_pos_x"),
    ("Tracking_Rel_Vel_Y",             "tracking_rel_vel_y"),
    ("Tracking_Rel_Vel_X",             "tracking_rel_vel_x"),
    ("Tracking_Rel_Acc_Y",             "tracking_rel_acc_y"),
    ("Tracking_Rel_Acc_X",             "tracking_rel_acc_x"),
    ("Tracking_Updated_Age",           "tracking_updated_age"),
    ("Tracking_Coasting_Age",          "tracking_coasting_age"),
    ("Tracking_Life_Time",             "tracking_life_time"),
    ("Tracking_Rel_Pos_Normal",        "tracking_rel_pos_normal"),
    ("Tracking_Rel_Vel_Tangential",    "tracking_rel_vel_tangential"),
    ("Tracking_Rel_Vel_Normal",        "tracking_rel_vel_normal"),
    ("Tracking_Abs_Vel_X",             "tracking_abs_vel_x"),
    ("Tracking_Abs_Vel_Y",             "tracking_abs_vel_y"),
    ("Tracking_Moving_Direction",      "tracking_moving_direction"),
    ("Tracking_Heading_Angle",         "tracking_heading_angle"),
    ("Tracking_Range",                 "tracking_range"),
    ("Tracking_Angle",                 "tracking_angle"),
    ("Tracking_Threat_TTC",            "tracking_threat_ttc"),
    ("Tracking_Threat_I_Long",         "tracking_threat_i_long"),
    ("Tracking_Threat_I_LAT",          "tracking_threat_i_lat"),
    ("Tracking_Threat_RSS_X",          "tracking_threat_rss_x"),
    ("Tracking_Threat_RSS_Y",          "tracking_threat_rss_y"),
    ("Tracking_Threat_FCW_HONDA",      "tracking_threat_fcw_honda"),
    ("Tracking_Threat_CP",             "tracking_threat_cp"),
]

# Mobileye Track → Additional Data 그룹 매핑
_TRACK_TO_GROUP: Dict[int, int] = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5}


# ─────────────────────────────────────────────────────────────
# 센서별 프로세서
# ─────────────────────────────────────────────────────────────

def process_chassis(reader: BagReader, syncer: Synchronizer) -> Dict[str, np.ndarray]:
    """/LOGBYTE0~3 → Chassis"""
    result: Dict[str, np.ndarray] = {}
    for idx in range(4):
        topic = f"/LOGBYTE{idx}"
        msgs, ts = reader.get(topic)
        if not msgs:
            print(f"  [WARN] {topic} 없음")
            continue
        result.update(_extract_flat(syncer.sync(msgs, ts), syncer.signal_length,
                                    prefix=f"LOG_BYTE{idx}_"))
    print("  Chassis 완료")
    return result


def process_odd_monitor(reader: BagReader, syncer: Synchronizer) -> Dict[str, np.ndarray]:
    msgs, ts = reader.get("/ODD_monitor")
    if not msgs:
        print("  [WARN] ODD_monitor 없음")
        return {}
    result = _extract_flat(syncer.sync(msgs, ts), syncer.signal_length)
    print("  ODD Monitor 완료")
    return result


def process_gnss(reader: BagReader, syncer: Synchronizer) -> Dict[str, Any]:
    """
    MATLAB 구조:
      GNSS.GNSS_Latitude / GNSS_Longitude / GNSS_Altitude   ← /gnss (flat)
      GNSS.Filter_Positionlla_Vector.X/Y/Z                  ← /filter/positionlla
      GNSS.Filter_Velocity_Vector.X/Y/Z                     ← /filter/velocity
      GNSS.IMU_data_AngularVelocity.X/Y/Z                   ← /imu/data
      GNSS.IMU_data_LinearAcceleration.X/Y/Z
      GNSS.IMU_data_Orientation.W/X/Y/Z
      GNSS.IMU_time_ref_TimeRef.Sec/Nsec                    ← /imu/time_ref
    """
    result: Dict[str, Any] = {}
    N = syncer.signal_length

    # /gnss → flat fields with GNSS_ prefix
    msgs, ts = reader.get("/gnss")
    if msgs:
        result.update(_extract_flat(syncer.sync(msgs, ts), N, prefix="GNSS_"))
    else:
        print("  [WARN] /gnss 없음")

    # /filter/positionlla → Filter_Positionlla_Vector: {X, Y, Z}
    msgs, ts = reader.get("/filter/positionlla")
    if msgs:
        synced = syncer.sync(msgs, ts)
        vec = {"X": np.zeros(N), "Y": np.zeros(N), "Z": np.zeros(N)}
        for t, msg in enumerate(synced):
            if msg is not None:
                vec["X"][t] = _fval(msg, "x")
                vec["Y"][t] = _fval(msg, "y")
                vec["Z"][t] = _fval(msg, "z")
        result["Filter_Positionlla_Vector"] = vec
    else:
        print("  [WARN] /filter/positionlla 없음")

    # /filter/velocity → Filter_Velocity_Vector: {X, Y, Z}
    msgs, ts = reader.get("/filter/velocity")
    if msgs:
        synced = syncer.sync(msgs, ts)
        vec = {"X": np.zeros(N), "Y": np.zeros(N), "Z": np.zeros(N)}
        for t, msg in enumerate(synced):
            if msg is not None:
                vec["X"][t] = _fval(msg, "x")
                vec["Y"][t] = _fval(msg, "y")
                vec["Z"][t] = _fval(msg, "z")
        result["Filter_Velocity_Vector"] = vec
    else:
        print("  [WARN] /filter/velocity 없음")

    # /imu/data → 3 nested sub-structs
    msgs, ts = reader.get("/imu/data")
    if msgs:
        synced = syncer.sync(msgs, ts)
        ang = {"X": np.zeros(N), "Y": np.zeros(N), "Z": np.zeros(N)}
        lin = {"X": np.zeros(N), "Y": np.zeros(N), "Z": np.zeros(N)}
        ori = {"W": np.zeros(N), "X": np.zeros(N), "Y": np.zeros(N), "Z": np.zeros(N)}
        for t, msg in enumerate(synced):
            if msg is None:
                continue
            av = getattr(msg, "angular_velocity", None)
            if av:
                ang["X"][t] = _fval(av, "x")
                ang["Y"][t] = _fval(av, "y")
                ang["Z"][t] = _fval(av, "z")
            la = getattr(msg, "linear_acceleration", None)
            if la:
                lin["X"][t] = _fval(la, "x")
                lin["Y"][t] = _fval(la, "y")
                lin["Z"][t] = _fval(la, "z")
            ot = getattr(msg, "orientation", None)
            if ot:
                ori["W"][t] = _fval(ot, "w")
                ori["X"][t] = _fval(ot, "x")
                ori["Y"][t] = _fval(ot, "y")
                ori["Z"][t] = _fval(ot, "z")
        result["IMU_data_AngularVelocity"]    = ang
        result["IMU_data_LinearAcceleration"] = lin
        result["IMU_data_Orientation"]        = ori
    else:
        print("  [WARN] /imu/data 없음")

    # /imu/time_ref → IMU_time_ref_TimeRef: {Sec, Nsec}
    msgs, ts = reader.get("/imu/time_ref")
    if msgs:
        synced = syncer.sync(msgs, ts)
        tref = {"Sec": np.zeros(N), "Nsec": np.zeros(N)}
        for t, msg in enumerate(synced):
            if msg is None:
                continue
            tr = getattr(msg, "time_ref", None)
            if tr:
                tref["Sec"][t]  = _fval(tr, "sec")
                tref["Nsec"][t] = _fval(tr, "nanosec")
        result["IMU_time_ref_TimeRef"] = tref
    else:
        print("  [WARN] /imu/time_ref 없음")

    print("  GNSS 완료")
    return result


def process_lidar_detection(
    reader: BagReader,
    syncer: Synchronizer,
    range_right: float,
    range_left:  float,
    range_front: float,
    range_rear:  float,
    max_objects: int,
) -> Dict[str, Any]:
    msgs, ts = reader.get("/detection/lidar_objects")
    if not msgs:
        print("  [WARN] Lidar Detection 없음")
        return {}

    synced   = syncer.sync(msgs, ts)
    N, MAX   = syncer.signal_length, max_objects
    obj_fields = ["Rel_Pos_X","Rel_Pos_Y","Yaw","Length","Width","Height","Class","Score"]
    # MATLAB 구조: Lidar_Detection.Object1.Rel_Pos_X (nested struct)
    result: Dict[str, Any] = {
        f"Object{i}": {f: np.zeros(N) for f in obj_fields}
        for i in range(1, MAX + 1)
    }
    _CLASS_MAP = {1: 2, 2: 1, 3: 7}

    for t, msg in enumerate(synced):
        if msg is None:
            continue
        objs = list(getattr(msg, "objects", []))
        if not objs:
            continue
        in_roi = [
            o for o in objs
            if range_rear  <= _nested_xy(o, "x") <= range_front
            and range_right <= _nested_xy(o, "y") <= range_left
        ]
        selected = (in_roi + [o for o in objs if o not in in_roi])[:MAX]

        for k, obj in enumerate(selected, start=1):
            pose = getattr(obj, "pose", None)
            pos  = getattr(pose, "position", None) if pose else None
            dims = getattr(obj, "dimensions", None)
            ob = result[f"Object{k}"]
            ob["Rel_Pos_X"][t] = _fval(pos, "x") if pos else 0.0
            ob["Rel_Pos_Y"][t] = _fval(pos, "y") if pos else 0.0
            ob["Yaw"][t]       = _fval(obj, "angle")
            ob["Length"][t]    = _fval(dims, "x") if dims else 0.0
            ob["Width"][t]     = _fval(dims, "y") if dims else 0.0
            ob["Height"][t]    = _fval(dims, "z") if dims else 0.0
            ob["Class"][t]     = _CLASS_MAP.get(int(getattr(obj, "indicator_state", 0)), 15)
            ob["Score"][t]     = _fval(obj, "score")

    print("  Lidar Detection 완료")
    return result


def _nested_xy(obj: Any, axis: str) -> float:
    """pose.position.x/y 안전 추출 헬퍼"""
    pose = getattr(obj, "pose", None)
    pos  = getattr(pose, "position", None) if pose else None
    return _fval(pos, axis, 999.0) if pos else 999.0


def process_lidar_tracking(
    reader:      BagReader,
    syncer:      Synchronizer,
    max_objects: int,
) -> Dict[str, Any]:
    msgs, ts = reader.get("/lidar_tracking")
    if not msgs:
        print("  [WARN] Lidar Tracking 없음")
        return {}

    synced = syncer.sync(msgs, ts)
    N, MAX = syncer.signal_length, max_objects
    trk_fields = ["Rel_Pos_X","Rel_Pos_Y","Rel_Vel_X","Rel_Vel_Y",
                  "Yaw","Length","Width","Height","ID","Class","Life_Time"]
    # MATLAB 구조: Lidar_Track.Object1.ID (nested struct)
    result: Dict[str, Any] = {
        f"Object{i}": {f: np.zeros(N) for f in trk_fields}
        for i in range(1, MAX + 1)
    }
    _ATTR = dict(zip(trk_fields, [
        "rel_pos_x","rel_pos_y","rel_vel_x","rel_vel_y",
        "yaw","length","width","height","id","class_","life_time",
    ]))

    for t, msg in enumerate(synced):
        if msg is None:
            continue
        for k, trk in enumerate(list(getattr(msg, "tracks", []))[:MAX], start=1):
            ob = result[f"Object{k}"]
            for fname, attr in _ATTR.items():
                ob[fname][t] = _fval(trk, attr)

    print("  Lidar Tracking 완료")
    return result


def process_front_radar_track(
    reader:     BagReader,
    syncer:     Synchronizer,
    max_tracks: int,
) -> Dict[str, Any]:
    """
    MATLAB 구조: Front_Radar_Track.Track1.TrackStatus (nested struct)
    """
    N = syncer.signal_length
    result: Dict[str, Any] = {}

    for i in range(1, max_tracks + 1):
        msgs, ts = reader.get(f"/Track{i}")
        if not msgs:
            continue
        synced = syncer.sync(msgs, ts)
        track_data: Dict[str, np.ndarray] = {}
        for mat_name, ros_attr in _RADAR_FIELDS:
            arr = np.zeros(N)
            for t, msg in enumerate(synced):
                if msg is not None:
                    arr[t] = _fval(msg, ros_attr)
            track_data[mat_name] = arr
        result[f"Track{i}"] = track_data

    print("  Front Radar Track 완료")
    return result


def process_mobileye_lane(reader: BagReader, syncer: Synchronizer) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for key, topic in _ME_LANE_TOPICS.items():
        msgs, ts = reader.get(topic)
        if not msgs:
            print(f"  [WARN] {topic} 없음")
            continue
        result.update(_extract_flat(syncer.sync(msgs, ts), syncer.signal_length,
                                    prefix=f"ME_Lane_{key}_"))
    print("  Mobileye Lane 완료")
    return result


def process_mobileye_track(
    reader:     BagReader,
    syncer:     Synchronizer,
    max_tracks: int,
) -> Dict[str, Any]:
    """
    MATLAB 구조: Mobileye_Track.Track1.ObjectClass (nested struct)
    """
    N, MAX = syncer.signal_length, max_tracks

    obs: Dict[str, List[Optional[Any]]] = {}
    for i in range(1, MAX + 1):
        for suf in ("A", "B", "C"):
            msgs, ts = reader.get(f"/ObstacleData{i}{suf}")
            obs[f"{i}{suf}"] = syncer.sync(msgs, ts) if msgs else [None] * N

    add: Dict[int, List[Optional[Any]]] = {}
    for i in range(1, 6):
        msgs, ts = reader.get(f"/ObstacleAdditionalData{i}")
        add[i] = syncer.sync(msgs, ts) if msgs else [None] * N

    out_keys = [
        "ObjectClass","LongitudinalDistance","MotionCategory","LateralDistance",
        "AliveCounter","AbsoluteLongVelocity","AbsoluteLateralVelocity","VD3DValid",
        "ID","PredictedObject","AngleRight","MotionStatus","AngleLeft",
        "MotionOrientation","AngleRateMean","AbsoluteLongAcc","LaneAssignment",
        "Length","BrakeLight","Width","SyncFrameIndex","ObjectAge",
        "TurnIndicatorLeft","TurnIndicatorRight","ObstacleHeading",
        "ObstacleHeight","ObstacleSide",
    ]
    # MATLAB 구조: Mobileye_Track.Track1.ObjectClass (nested struct)
    result: Dict[str, Any] = {
        f"Track{i}": {k: np.zeros(N) for k in out_keys}
        for i in range(1, MAX + 1)
    }

    for i in range(1, MAX + 1):
        g = _TRACK_TO_GROUP[i]
        tr = result[f"Track{i}"]
        for t in range(N):
            ma, mb, mc, md = obs[f"{i}A"][t], obs[f"{i}B"][t], obs[f"{i}C"][t], add[g][t]

            if ma:
                tr["ObjectClass"][t]              = _fval(ma, "object_class")
                tr["LongitudinalDistance"][t]     = _fval(ma, "longitudinal_distance")
                tr["MotionCategory"][t]           = _fval(ma, "motion_category")
                tr["LateralDistance"][t]          = _fval(ma, "lateral_distance")
                tr["AliveCounter"][t]             = _fval(ma, "alive_counter")
                tr["AbsoluteLongVelocity"][t]     = _fval(ma, "absolute_long_velocity")
                tr["AbsoluteLateralVelocity"][t]  = _fval(ma, "absolute_lateral_velocity")
                tr["VD3DValid"][t]                = _fval(ma, "vd3_d_valid")
                tr["ID"][t]                       = _fval(ma, "id")
            if mb:
                tr["PredictedObject"][t]          = _fval(mb, "predicted_object")
                tr["AngleRight"][t]               = _fval(mb, "angle_right")
                tr["MotionStatus"][t]             = _fval(mb, "motion_status")
                tr["AngleLeft"][t]                = _fval(mb, "angle_left")
                tr["MotionOrientation"][t]        = _fval(mb, "motion_orientation")
                tr["AngleRateMean"][t]            = _fval(mb, "angle_rate_mean")
                tr["AbsoluteLongAcc"][t]          = _fval(mb, "absolute_long_acc")
            if mc:
                tr["LaneAssignment"][t]           = _fval(mc, "lane_assignment")
                tr["Length"][t]                   = _fval(mc, "length")
                tr["BrakeLight"][t]               = _fval(mc, "brake_light")
                tr["Width"][t]                    = _fval(mc, "width")
                tr["SyncFrameIndex"][t]           = _fval(mc, "sync_frame_index")
                tr["ObjectAge"][t]                = _fval(mc, "object_age")
                tr["TurnIndicatorLeft"][t]        = _fval(mc, "turn_indicator_left")
                tr["TurnIndicatorRight"][t]       = _fval(mc, "turn_indicator_right")
                tr["ObstacleHeading"][t]          = _fval(mc, "obstacle_heading")
            if md:
                tr["ObstacleHeight"][t] = _fval(md, f"obstacle_height{i}")
                tr["ObstacleSide"][t]   = _fval(md, f"obstacle_side{i}")

    print("  Mobileye Track 완료")
    return result


def process_fusion_track_v3(
    reader:     BagReader,
    syncer:     Synchronizer,
    max_tracks: int,
) -> Dict[str, Any]:
    """
    MATLAB 구조: Fusion_Track_v3.Track1.Measure_Rel_Pos_Y (nested struct)
    """
    topic = "/sensorfusion/situationalawareness/fusiontrackmaneuver"
    msgs, ts = reader.get(topic)
    if not msgs:
        print("  [WARN] Fusion Track v3 없음")
        return {}

    synced = syncer.sync(msgs, ts)
    N, MAX = syncer.signal_length, max_tracks

    # MATLAB 구조에 맞춰 nested dict로 초기화
    result: Dict[str, Any] = {
        f"Track{i}": {mat: np.zeros(N) for mat, _ in _FT3_FIELD_MAP}
        for i in range(1, MAX + 1)
    }

    for t, msg in enumerate(synced):
        if msg is None:
            continue
        for k, trk in enumerate(list(getattr(msg, "tracks", []))[:MAX], start=1):
            tr = result[f"Track{k}"]
            for mat_name, ros_attr in _FT3_FIELD_MAP:
                tr[mat_name][t] = _fval(trk, ros_attr)

    print("  Fusion Track v3 완료")
    return result


def process_fallback_decision(reader: BagReader, syncer: Synchronizer) -> Dict[str, np.ndarray]:
    msgs, ts = reader.get("/fallback_result_topic_Orin")
    if not msgs:
        print("  [WARN] Fallback Decision 없음")
        return {}
    N   = syncer.signal_length
    arr = np.zeros(N)
    for t, msg in enumerate(syncer.sync(msgs, ts)):
        if msg is not None:
            arr[t] = _fval(msg, "data")
    print("  Fallback Decision 완료")
    return {"Data": arr}


def process_collision_mode(
    reader: BagReader,
    syncer: Synchronizer,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Returns: (collision_mode dict, collision_image dict)"""
    msgs, ts = reader.get("/CM_combined_topic_Orin")
    if not msgs:
        print("  [WARN] Collision Mode 없음")
        return {}, {}

    synced = syncer.sync(msgs, ts)
    N      = syncer.signal_length
    prob   = np.zeros(N)
    cls_   = np.zeros(N)
    bev_data: List[Optional[np.ndarray]] = [None] * N

    for t, msg in enumerate(synced):
        if msg is None:
            continue
        cm = getattr(msg, "collision_mode", None)
        if cm:
            prob[t] = _fval(cm, "cm_probability")
            cls_[t] = _fval(cm, "cm_classification")
        bev = getattr(msg, "bev_data", None)
        if bev:
            raw = np.array(list(getattr(bev, "data", [])), dtype=np.uint8)
            if raw.size == 201 * 101 * 3:
                bev_data[t] = raw.reshape(201, 101, 3)

    print("  Collision Mode 완료")
    return {"Probability": prob, "Classification": cls_}, {"Data": bev_data}


def process_vision_avi(bag_path: Path, output_path: Path) -> bool:
    """
    /front_cam/image_raw → AVI 파일 저장 (스트리밍, 메모리 효율적).
    Returns True on success.
    """
    try:
        import cv2
    except ImportError:
        print("  [WARN] opencv-python 없음.  pip install opencv-python")
        return False

    try:
        from rosbags.rosbag2 import Reader as _R
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        print("  [WARN] rosbags 없음")
        return False

    TOPIC = "/front_cam/image_raw"

    # 1) 타임스탬프 수집 (FPS 계산)
    with _R(str(bag_path)) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        if not conns:
            print(f"  [WARN] {TOPIC} 없음")
            return False
        ts_list = [ts_ns * 1e-9 for _, ts_ns, _ in reader.messages(connections=conns)]

    if not ts_list:
        print(f"  [WARN] {TOPIC} 메시지 없음")
        return False

    n_frames = len(ts_list)
    fps = float(n_frames - 1) / (ts_list[-1] - ts_list[0]) if n_frames >= 2 else 10.0
    fps = max(1.0, min(fps, 120.0))

    # 2) 프레임 순차 기록
    ts_store = get_typestore(Stores.ROS2_FOXY)
    writer: Any = None

    _ENC_TABLE = {
        "rgb8":   lambda a, h, w: cv2.cvtColor(a.reshape(h, w, 3), cv2.COLOR_RGB2BGR),
        "bgr8":   lambda a, h, w: a.reshape(h, w, 3),
        "mono8":  lambda a, h, w: cv2.cvtColor(a.reshape(h, w),    cv2.COLOR_GRAY2BGR),
        "8uc1":   lambda a, h, w: cv2.cvtColor(a.reshape(h, w),    cv2.COLOR_GRAY2BGR),
        "yuv422": lambda a, h, w: cv2.cvtColor(a.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV),
        "yuyv":   lambda a, h, w: cv2.cvtColor(a.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV),
    }

    import numpy as np
    saved = 0
    with _R(str(bag_path)) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        for conn, _, rawdata in reader.messages(connections=conns):
            try:
                msg = ts_store.deserialize_cdr(rawdata, conn.msgtype)
            except Exception:
                continue
            h    = int(getattr(msg, "height",   0))
            w    = int(getattr(msg, "width",    0))
            step = int(getattr(msg, "step",     0))
            enc  = str(getattr(msg, "encoding", "bgr8")).lower().replace("-", "")
            data = getattr(msg, "data", None)
            if data is None or h == 0 or w == 0:
                continue
            raw_arr = data if isinstance(data, np.ndarray) else np.frombuffer(bytes(data), dtype=np.uint8)
            try:
                conv = _ENC_TABLE.get(enc)
                if conv:
                    frame = conv(raw_arr, h, w)
                else:
                    ch = step // w if w > 0 else 3
                    frame = raw_arr[: h * w * ch].reshape(h, w, ch)
                    if ch == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
                continue
            if writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (fw, fh))
                if not writer.isOpened():
                    print(f"  [WARN] AVI 열기 실패: {output_path}")
                    return False
            writer.write(frame)
            saved += 1

    if writer is None:
        print("  [WARN] 기록된 프레임 없음")
        return False
    writer.release()
    print(f"  저장: {output_path}  ({saved}프레임 / {fps:.1f}fps)")
    return True


# ─────────────────────────────────────────────────────────────
# 필요 토픽 목록 수집
# ─────────────────────────────────────────────────────────────

def collect_needed_topics(cfg: Any) -> List[str]:
    """Config 토글에 따라 읽어야 할 topic 목록을 반환한다."""
    topics: List[str] = ["/timestamp"]  # 기준 시간 (없어도 무방)

    if cfg.toggle_chassis:
        topics += [f"/LOGBYTE{i}" for i in range(4)]
    if cfg.toggle_mobileye:
        topics += list(_ME_LANE_TOPICS.values())
        topics += [f"/ObstacleData{i}{s}" for i in range(1, 11) for s in ("A","B","C")]
        topics += [f"/ObstacleAdditionalData{i}" for i in range(1, 6)]
    if cfg.toggle_front_radar_track:
        topics += [f"/Track{i}" for i in range(1, cfg.max_radar_tracks + 1)]
    if cfg.toggle_gnss:
        topics += list(_GNSS_TOPICS.values())
    if cfg.toggle_lidar_detection:
        topics.append("/detection/lidar_objects")
    if cfg.toggle_lidar_tracking:
        topics.append("/lidar_tracking")
    if cfg.toggle_odd_monitor:
        topics.append("/ODD_monitor")
    if cfg.toggle_fusion_track_v3:
        topics.append("/sensorfusion/situationalawareness/fusiontrackmaneuver")
    if cfg.toggle_fallback_decision:
        topics.append("/fallback_result_topic_Orin")
    if cfg.toggle_collision_mode_sfcpp:
        topics.append("/CM_combined_topic_Orin")

    return topics
