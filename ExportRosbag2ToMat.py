#!/usr/bin/env python3
"""
ExportRosbag2ToMat.py  ─  진입점

ROS2 bag (.db3) → MATLAB (.mat) 변환기
원본: ExportRosbagToMat_v23_9_ros2migration.m

사용법:
    python ExportRosbag2ToMat.py

의존성 설치 (Windows/Linux 공통, ROS2 설치 불필요):
    pip install rosbags numpy scipy

파일 구성:
    ExportRosbag2ToMat.py  ← 이 파일 (Config · 오케스트레이터 · main)
    util/processors.py     ← 센서별 MAT 변환 프로세서
    util/utils.py          ← BagReader · Synchronizer · 공통 헬퍼
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.io import savemat

from util.utils import BagReader, Synchronizer, _BACKEND
from util.processors import (
    collect_needed_topics,
    process_chassis,
    process_collision_mode,
    process_fallback_decision,
    process_front_radar_track,
    process_fusion_track_v3,
    process_gnss,
    process_lidar_detection,
    process_lidar_tracking,
    process_mobileye_lane,
    process_mobileye_track,
    process_odd_monitor,
    process_road_barrier,
    process_target,
    process_vision_avi,
)

print(f"[INFO] 백엔드: {_BACKEND}")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    사용자 설정 블록.
    MATLAB 코드 상단 파라미터 섹션에 해당.
    """

    # ── 경로 ───────────────────────────────────
    vehicle_date:   str = "CN7_030526"
    rosbag_dir:     str = "D:/FOT_Avante Data_1/Rosbag"
    rosbag2mat_dir: str = "D:/FOT_Avante Data_1/Rosbag2Mat"

    # ── 샘플링 주기 (20 Hz) ────────────────────
    sample_time: float = 0.05

    # ── Lidar ROI 범위 (m) ─────────────────────
    range_right: float = -20.0
    range_left:  float =  20.0
    range_front: float =  50.0
    range_rear:  float = -20.0

    # ── 저장 토글 (On) ─────────────────────────
    toggle_chassis:              bool = True
    toggle_mobileye:             bool = True
    toggle_front_radar_track:    bool = True
    toggle_gnss:                 bool = False
    toggle_lidar_detection:      bool = False
    toggle_lidar_tracking:       bool = False
    toggle_odd_monitor:          bool = False
    toggle_fusion_track_v3:      bool = True
    toggle_target:               bool = True
    toggle_road_barrier:         bool = True
    toggle_fallback_decision:    bool = False
    toggle_collision_mode_sfcpp: bool = False
    toggle_vision_avi:           bool = True
    toggle_save_mat:             bool = True

    # ── 저장 토글 (Off) ────────────────────────
    toggle_morai:                bool = False
    toggle_aes:                  bool = False
    toggle_sf:                   bool = False
    toggle_fusion_track_v1:      bool = False
    toggle_recognition:          bool = False
    toggle_sf_bypass:            bool = False
    toggle_pnt:                  bool = False
    toggle_me_lane_pp:           bool = False
    toggle_me_track_pp:          bool = False
    toggle_radar_track_pp:       bool = False
    toggle_in_vehicle_sensor_pp: bool = False
    toggle_collision_mode_sfpy:  bool = False

    # ── 처리 범위 (1-base, 0 = 전체) ──────────
    start_bag_number: int = 1
    end_bag_number:   int = 0

    # ── 최대 객체 수 ───────────────────────────
    max_detection_objects: int = 32
    max_tracking_objects:  int = 32
    max_radar_tracks:      int = 32
    max_mobileye_tracks:   int = 10
    max_fusion_tracks:     int = 64

    # ── 파생 경로 ──────────────────────────────
    @property
    def rosbag_path(self) -> Path:
        return Path(self.rosbag_dir) / self.vehicle_date

    @property
    def rosbag2mat_path(self) -> Path:
        return Path(self.rosbag2mat_dir) / self.vehicle_date


# ─────────────────────────────────────────────────────────────
# 단일 BAG 처리 오케스트레이터
# ─────────────────────────────────────────────────────────────

def process_single_bag(bag_path: Path, cfg: Config, output_path: Path) -> bool:
    """
    단일 bag → 동기화 → 변환 → .mat 저장.

    Returns:
        True  : 정상 처리
        False : 에러 (건너뜀)
    """
    print(f"\n{'='*60}")
    print(f"  Bag : {bag_path.name}")
    print(f"{'='*60}")
    t_total = time.perf_counter()

    # ── 1) 단일 패스 읽기 ───────────────────────
    reader    = BagReader(bag_path)
    available = reader.available_topics
    needed    = collect_needed_topics(cfg)

    missing = [t for t in needed if t not in available]
    if missing:
        print(f"  [INFO] bag에 없는 topic {len(missing)}개 (처리 계속)")
        print(f"  bag 내 topic 목록: {sorted(available.keys())}")

    print(f"  topic 읽는 중... ({len(needed)}개 요청)", end=" ", flush=True)
    t0 = time.perf_counter()
    reader.read_all(needed)
    print(f"({time.perf_counter()-t0:.1f}s)")

    # ── 2) 동기화 기준 시간 설정 ─────────────────
    _, ts_times = reader.get("/timestamp")
    if ts_times.size == 0:
        print("  [ERROR] /timestamp 토픽 없음 → 건너뜀")
        return False
    start_time = float(ts_times[0])
    end_time   = float(ts_times[-1])

    start_time    = float(start_time)
    end_time      = float(end_time)
    signal_length = max(1, int((end_time - start_time) / cfg.sample_time))
    syncer        = Synchronizer(start_time, cfg.sample_time, signal_length)

    print(f"  시간: {start_time:.2f}~{end_time:.2f}s  |  "
          f"{signal_length} steps @ {1/cfg.sample_time:.0f} Hz")

    # ── 3) 센서별 처리 ──────────────────────────
    save_data: Dict[str, Any] = {
        "time":        syncer.sync_time,
        "Sample_Time": np.array([cfg.sample_time]),
    }

    def run(key: str, data: Dict) -> None:
        if data:
            save_data[key] = data

    if cfg.toggle_chassis:
        _timed("Chassis",
               lambda: run("Chassis", process_chassis(reader, syncer)))

    if cfg.toggle_mobileye:
        _timed("Mobileye Lane",
               lambda: run("Mobileye_Lane", process_mobileye_lane(reader, syncer)))
        _timed("Mobileye Track",
               lambda: run("Mobileye_Track",
                           process_mobileye_track(reader, syncer, cfg.max_mobileye_tracks)))

    if cfg.toggle_front_radar_track:
        _timed("Front Radar Track",
               lambda: run("Front_Radar_Track",
                           process_front_radar_track(reader, syncer, cfg.max_radar_tracks)))

    if cfg.toggle_gnss:
        _timed("GNSS",
               lambda: run("GNSS", process_gnss(reader, syncer)))

    if cfg.toggle_lidar_detection:
        _timed("Lidar Detection",
               lambda: run("Lidar_Detection",
                           process_lidar_detection(reader, syncer,
                                                   cfg.range_right, cfg.range_left,
                                                   cfg.range_front, cfg.range_rear,
                                                   cfg.max_detection_objects)))

    if cfg.toggle_lidar_tracking:
        _timed("Lidar Tracking",
               lambda: run("Lidar_Track",
                           process_lidar_tracking(reader, syncer, cfg.max_tracking_objects)))

    if cfg.toggle_odd_monitor:
        _timed("ODD Monitor",
               lambda: run("ODD_Monitor", process_odd_monitor(reader, syncer)))

    if cfg.toggle_fusion_track_v3:
        _timed("Fusion Track v3",
               lambda: run("Fusion_Track_v3",
                           process_fusion_track_v3(bag_path, syncer, cfg.max_fusion_tracks)))

    if cfg.toggle_target:
        _timed("Target",
               lambda: save_data.update(process_target(reader, syncer)))

    if cfg.toggle_road_barrier:
        _timed("Road Barrier",
               lambda: save_data.update(process_road_barrier(reader, syncer)))

    if cfg.toggle_fallback_decision:
        _timed("Fallback Decision",
               lambda: run("Fallback_decision", process_fallback_decision(reader, syncer)))

    if cfg.toggle_collision_mode_sfcpp:
        _timed("Collision Mode", lambda: _run_collision(reader, syncer, save_data))

    # ── 4) .mat 저장 ─────────────────────────────
    if cfg.toggle_save_mat:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(output_path), save_data, long_field_names=True)
        print(f"\n  저장: {output_path}")

    # ── 5) AVI 저장 ──────────────────────────────
    if cfg.toggle_vision_avi:
        avi_path = output_path.with_suffix(".avi")
        _timed("Front Camera AVI",
               lambda: process_vision_avi(bag_path, avi_path))

    print(f"  총 처리 시간: {(time.perf_counter()-t_total)/60:.2f} min")
    return True


def _timed(label: str, fn) -> None:
    print(f"  {label}...", end=" ")
    t0 = time.perf_counter()
    fn()
    print(f"[{time.perf_counter()-t0:5.1f}s]")


def _run_collision(reader: BagReader, syncer: Synchronizer, save_data: Dict) -> None:
    cm, ci = process_collision_mode(reader, syncer)
    if cm:
        save_data["Collision_Mode"]  = cm
    if ci:
        save_data["Collision_image"] = ci


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def find_bags(root: Path) -> List[Path]:
    """
    root 폴더에서 bag을 찾아 정렬 반환.

    Case 1: root 자체가 bag 폴더 (metadata.yaml 직접 포함)
        D:/Rosbag/CN7_030526/metadata.yaml  → [root]
    Case 2: root 안에 여러 bag 하위 폴더
        D:/Rosbag/CN7_030526/bag_001/metadata.yaml  → [bag_001, ...]
    """
    if not root.exists():
        return []
    if (root / "metadata.yaml").exists():
        return [root]
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "metadata.yaml").exists()
    )


def _select_folder(title: str, initial: str = "") -> Optional[Path]:
    """Windows 폴더 선택 다이얼로그. 취소 시 None 반환."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title=title, initialdir=initial or "/")
        root.destroy()
        return Path(folder) if folder else None
    except Exception as e:
        print(f"[WARN] 폴더 다이얼로그 오류: {e}")
        return None


def main() -> None:
    cfg = Config()

    # ── Rosbag 루트 폴더 선택 ────────────────────────────────
    # 루트 폴더(예: D:/FOT_Avante Data_1/Rosbag) 를 선택하면
    # 그 안의 모든 bag 하위 폴더(CN7_260312_001, 002, ...)를 자동 탐색
    print("Rosbag 루트 폴더를 선택하세요.")
    rosbag_root = _select_folder("Rosbag 루트 폴더 선택", cfg.rosbag_dir)
    if rosbag_root is None:
        print("[취소] 폴더를 선택하지 않았습니다.")
        return

    # ── 출력(.mat/.avi) 폴더 선택 ────────────────────────────
    print("출력 폴더를 선택하세요.")
    mat_root = _select_folder("출력 폴더 선택", cfg.rosbag2mat_dir)
    if mat_root is None:
        print("[취소] 폴더를 선택하지 않았습니다.")
        return

    print(f"  Rosbag : {rosbag_root}")
    print(f"  출력   : {mat_root}")

    bags = find_bags(rosbag_root)

    if not bags:
        print(f"[ERROR] bag 없음: {rosbag_root}")
        print("  metadata.yaml 이 있는 폴더가 있어야 합니다.")
        return

    end_num     = cfg.end_bag_number if cfg.end_bag_number > 0 else len(bags)
    target_bags = bags[cfg.start_bag_number - 1 : end_num]

    print(f"총 {len(bags)}개 bag 발견  →  {len(target_bags)}개 처리 예정")
    print(f"처리 범위: {cfg.start_bag_number} ~ {end_num}")

    mat_root.mkdir(parents=True, exist_ok=True)
    error_bags: List[str] = []

    for bag_path in target_bags:
        # 출력 파일명 = bag 폴더명 그대로 사용 (CN7_260312_001.mat 등)
        out_mat = mat_root / f"{bag_path.name}.mat"
        if not process_single_bag(bag_path, cfg, out_mat):
            error_bags.append(bag_path.name)

    print(f"\n{'='*60}")
    total = len(target_bags)
    print(f"처리 완료  성공: {total - len(error_bags)}/{total}")
    if error_bags:
        print(f"에러 bag: {error_bags}")


if __name__ == "__main__":
    main()
