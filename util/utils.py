"""
utils.py

ROS2 bag 읽기 / 동기화 / 공통 헬퍼
백엔드: rosbags (Windows/Linux, ROS2 불필요) → rosbag2_py (폴백)

설치:
    pip install rosbags numpy scipy
"""

from __future__ import annotations

import dataclasses
import sqlite3
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────
# 필드명 변환 헬퍼 (snake_case → PascalCase, MATLAB 참조 파일 호환)
# ─────────────────────────────────────────────────────────────

# 붙여쓴 복합어 → 언더스코어 분리 매핑 (긴 것부터 먼저 등록)
_COMPOUND_SPLITS: Dict[str, str] = {
    "lanemarkposition":       "lane_mark_position",
    "lanemarkheadingangle":   "lane_mark_heading_angle",
    "lanemarkmodela":         "lane_mark_model_a",
    "lanemarkmodelderiva":    "lane_mark_model_deriv_a",
    "lanemarkviewrangeavail": "lane_mark_view_range_avail",
    "lanemarkviewrange":      "lane_mark_view_range",
    "lanemark":               "lane_mark",
    "leftlanecolorinfo":      "left_lane_color_info",
    "rightlanecolorinfo":     "right_lane_color_info",
    "leftlaneprediction":     "left_lane_prediction",
    "rightlaneprediction":    "right_lane_prediction",
    "leftlane":               "left_lane",
    "rightlane":              "right_lane",
    "viewrange":              "view_range",
    "startpoint":             "start_point",
    "alivecnt":               "alive_cnt",
}

# 전체 대문자로 표기할 약어 (소문자 기준)
_ACRONYMS_UPPER: Set[str] = {"me", "ldw", "da", "rss", "fcw", "ttc", "cp"}


def _to_pascal(name: str) -> str:
    """snake_case → PascalCase (복합어 확장 + 약어 전체대문자 처리)."""
    result: List[str] = []
    for tok in name.split("_"):
        if not tok:
            continue
        if tok in _COMPOUND_SPLITS:
            for sub in _COMPOUND_SPLITS[tok].split("_"):
                if sub.lower() in _ACRONYMS_UPPER:
                    result.append(sub.upper())
                else:
                    result.append(sub[0].upper() + sub[1:])
        elif tok.lower() in _ACRONYMS_UPPER:
            result.append(tok.upper())
        else:
            result.append(tok[0].upper() + tok[1:])
    return "".join(result)

import numpy as np

# ─────────────────────────────────────────────────────────────
# 백엔드 선택
# ─────────────────────────────────────────────────────────────

_BACKEND: str = ""

# ros2_msg 커스텀 타입 .msg 파일 경로 (프로젝트 루트 기준)
_MSG_DIR: Path = Path(__file__).parent.parent / "ros2_msg" / "msg"

try:
    from rosbags.rosbag2 import Reader as _Rosbag2Reader          # type: ignore
    try:
        from rosbags.typesys import Stores as _Stores             # type: ignore
        from rosbags.typesys import get_typestore as _get_typestore  # type: ignore
        try:
            from rosbags.typesys import get_types_from_msg as _get_types_from_msg  # type: ignore
        except ImportError:
            _get_types_from_msg = None
        _ROSBAGS_NEW = True
    except ImportError:
        try:
            from rosbags.serde import deserialize_cdr as _serde_deser  # type: ignore
        except ImportError:
            from rosbags.serde.cdr import deserialize_cdr as _serde_deser  # type: ignore
        _ROSBAGS_NEW = False
    _BACKEND = "rosbags"

except ImportError:
    try:
        import rosbag2_py as _rosbag2_py                          # type: ignore
        from rclpy.serialization import deserialize_message as _rclpy_deser   # type: ignore
        from rosidl_runtime_py.utilities import get_message as _get_message   # type: ignore
        _BACKEND = "rosbag2_py"
    except ImportError:
        raise RuntimeError(
            "\n[오류] bag 읽기 라이브러리를 찾을 수 없습니다.\n\n"
            "  [권장] Windows / Linux (ROS2 불필요):\n"
            "      pip install rosbags\n\n"
            "  [대안] Linux + ROS2 환경:\n"
            "      source /opt/ros/foxy/setup.bash\n"
        )


# ─────────────────────────────────────────────────────────────
# BagReader
# ─────────────────────────────────────────────────────────────

class BagReader:
    """
    ROS2 .db3 bag 파일 단일 패스(single-pass) 읽기.
    rosbags(Windows 포함) 또는 rosbag2_py(ROS2 Linux) 백엔드 지원.
    """

    def __init__(self, bag_path: str | Path):
        self._bag_path  = Path(bag_path)
        self._type_map: Dict[str, str]                          = {}
        self._cache:    Dict[str, Tuple[List[Any], np.ndarray]] = {}

    # ── rosbags 헬퍼 ──────────────────────────────────────────

    def _rosbags_typestore(self, reader):
        if not _ROSBAGS_NEW:
            return None
        ts = _get_typestore(_Stores.ROS2_FOXY)
        if _get_types_from_msg is None:
            return ts

        add = {}

        # 1) connection msgdef에서 시도 (bag에 내장된 경우)
        for conn in reader.connections:
            msgdef = getattr(conn, "msgdef", None) or ""
            if msgdef:
                try:
                    add.update(_get_types_from_msg(msgdef, conn.msgtype))
                except Exception:
                    pass

        # 2) ros2_msg .msg 파일에서 커스텀 타입 직접 등록
        if _MSG_DIR.exists():
            for msg_file in sorted(_MSG_DIR.glob("*.msg")):
                msgtype = f"ros2_msg/msg/{msg_file.stem}"
                try:
                    content = msg_file.read_text(encoding="utf-8")
                    add.update(_get_types_from_msg(content, msgtype))
                except Exception:
                    pass
        else:
            print(f"  [WARN] ros2_msg .msg 디렉토리 없음: {_MSG_DIR}")

        if add:
            try:
                ts.register(add)
            except Exception:
                print(f"  [WARN] typestore 등록 실패:")
                print(traceback.format_exc())
        return ts

    def _rosbags_deser(self, ts, rawdata: bytes, msgtype: str):
        if _ROSBAGS_NEW:
            return ts.deserialize_cdr(rawdata, msgtype)
        return _serde_deser(rawdata, msgtype)

    # ── rosbag2_py 헬퍼 ───────────────────────────────────────

    def _rosbag2py_reader(self):
        reader = _rosbag2_py.SequentialReader()
        reader.open(
            _rosbag2_py.StorageOptions(uri=str(self._bag_path), storage_id="sqlite3"),
            _rosbag2_py.ConverterOptions("", ""),
        )
        self._type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
        return reader

    # ── 공개 인터페이스 ───────────────────────────────────────

    @property
    def available_topics(self) -> Dict[str, str]:
        if not self._type_map:
            if _BACKEND == "rosbags":
                with _Rosbag2Reader(str(self._bag_path)) as r:
                    self._type_map = {c.topic: c.msgtype for c in r.connections}
            else:
                self._rosbag2py_reader()
        return self._type_map

    def read_all(self, topics: List[str]) -> None:
        """지정한 topic 목록을 단일 패스로 읽어 내부 캐시에 저장한다."""
        if _BACKEND == "rosbags":
            self._read_rosbags(topics)
        else:
            self._read_rosbag2py(topics)

    def _read_rosbags(self, topics: List[str]) -> None:
        with _Rosbag2Reader(str(self._bag_path)) as reader:
            self._type_map = {c.topic: c.msgtype for c in reader.connections}
            valid = [t for t in topics if t in self._type_map]
            if not valid:
                self._cache = {}
                return

            ts    = self._rosbags_typestore(reader)
            conns = [c for c in reader.connections if c.topic in valid]
            raw: Dict[str, Tuple[List, List]] = {t: ([], []) for t in valid}

            _err_count: Dict[str, int] = {}

            for conn, ts_ns, rawdata in reader.messages(connections=conns):
                if conn.topic not in raw:
                    continue
                try:
                    msg = self._rosbags_deser(ts, rawdata, conn.msgtype)
                    raw[conn.topic][0].append(msg)
                    raw[conn.topic][1].append(ts_ns * 1e-9)
                except Exception:
                    cnt = _err_count.get(conn.topic, 0)
                    if cnt == 0:
                        print(f"\n  [WARN] 역직렬화 실패 ({conn.topic}): "
                              f"{traceback.format_exc().splitlines()[-1]}")
                    _err_count[conn.topic] = cnt + 1

            for topic, cnt in _err_count.items():
                if cnt > 0:
                    ok = len(raw.get(topic, ([],))[0])
                    print(f"  [WARN] {topic}: {cnt}개 메시지 스킵 (성공 {ok}개)")

        self._cache = {
            t: (msgs, np.array(tss, dtype=np.float64))
            for t, (msgs, tss) in raw.items()
        }

    def _read_rosbag2py(self, topics: List[str]) -> None:
        reader = self._rosbag2py_reader()
        valid  = [t for t in topics if t in self._type_map]
        if not valid:
            self._cache = {}
            return

        type_classes = {t: _get_message(self._type_map[t]) for t in valid}
        raw: Dict[str, Tuple[List, List]] = {t: ([], []) for t in valid}

        reader.set_filter(_rosbag2_py.StorageFilter(topics=valid))
        while reader.has_next():
            topic, data, ts_ns = reader.read_next()
            if topic in type_classes:
                raw[topic][0].append(_rclpy_deser(data, type_classes[topic]))
                raw[topic][1].append(ts_ns * 1e-9)

        self._cache = {
            t: (msgs, np.array(tss, dtype=np.float64))
            for t, (msgs, tss) in raw.items()
        }

    def get(self, topic: str) -> Tuple[List[Any], np.ndarray]:
        """캐시에서 topic 데이터 반환 (없으면 빈 결과)"""
        return self._cache.get(topic, ([], np.array([], dtype=np.float64)))


# ─────────────────────────────────────────────────────────────
# Synchronizer  (Zero-Order Hold)
# ─────────────────────────────────────────────────────────────

class Synchronizer:
    """모든 topic 메시지를 지정 Hz 기준 시간 그리드에 ZOH 동기화한다."""

    def __init__(self, start_time: float, sample_time: float, signal_length: int):
        self.start_time    = start_time
        self.sample_time   = sample_time
        self.signal_length = signal_length
        self.sync_time     = np.arange(signal_length, dtype=np.float64) * sample_time

    def sync(
        self,
        messages:   List[Any],
        timestamps: np.ndarray,
    ) -> List[Optional[Any]]:
        synced: List[Optional[Any]] = [None] * self.signal_length

        if not messages:
            return synced

        target_times = self.start_time + self.sync_time

        for j, t_target in enumerate(target_times):
            idx = int(np.argmin(np.abs(timestamps - t_target)))
            dt  = timestamps[idx] - t_target
            if abs(dt) <= self.sample_time * 2:
                synced[j] = messages[idx - 1] if (dt > 0 and idx > 0) else messages[idx]

        # ZOH forward-fill
        last: Optional[Any] = None
        for j in range(self.signal_length):
            if synced[j] is not None:
                last = synced[j]
            elif last is not None:
                synced[j] = last

        return synced


# ─────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────

def bag_start_from_db3(bag_path: Path) -> Optional[float]:
    """
    bag 폴더 안의 .db3 파일에서 첫 메시지 타임스탬프(초)를 읽는다.
    /timestamp 토픽이 없을 때 start_time 기준으로 사용.
    """
    db3_files = sorted(bag_path.glob("*.db3"))
    if not db3_files:
        return None
    try:
        con = sqlite3.connect(str(db3_files[0]))
        cur = con.execute("SELECT MIN(timestamp) FROM messages")
        row = cur.fetchone()
        con.close()
        if row and row[0] is not None:
            return row[0] * 1e-9   # nanosec → sec
    except Exception:
        pass
    return None


_SKIP = {"header", "Header", "MessageType"}


def _fval(msg: Any, name: str, default: float = 0.0) -> float:
    """메시지에서 단일 float 값을 안전하게 추출한다."""
    v = getattr(msg, name, default)
    if v is None:
        return default
    try:
        dataclasses.fields(v)  # type: ignore[arg-type]
        return default          # 중첩 struct → 기본값
    except TypeError:
        pass
    if isinstance(v, (np.ndarray, list, tuple, bytes, bytearray)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract_flat(
    synced: List[Optional[Any]],
    n:      int,
    prefix: str = "",
) -> Dict[str, np.ndarray]:
    """
    동기화된 flat 메시지 리스트에서 scalar 필드를 1D 배열로 추출한다.
    중첩 struct · 배열 필드는 건너뛴다.
    """
    sample = next((m for m in synced if m is not None), None)
    if sample is None:
        return {}

    try:
        all_fields = [f.name for f in dataclasses.fields(sample)]  # type: ignore[arg-type]
    except TypeError:
        all_fields = list(vars(sample).keys())

    scalar_fields: List[str] = []
    for fname in all_fields:
        if fname.startswith("_") or fname in _SKIP:   # _msgtype_ 등 내부 필드 제외
            continue
        val = getattr(sample, fname, None)
        if val is None or isinstance(val, str):        # string 필드 제외
            continue
        try:
            dataclasses.fields(val)  # type: ignore[arg-type]
            continue  # 중첩 struct
        except TypeError:
            pass
        if isinstance(val, (np.ndarray, list, tuple, bytes, bytearray)):
            continue
        scalar_fields.append(fname)

    out: Dict[str, np.ndarray] = {}
    for fname in scalar_fields:
        pascal = _to_pascal(fname)                     # snake_case → PascalCase
        key = f"{prefix}{pascal}" if prefix else pascal
        arr = np.zeros(n, dtype=np.float64)
        for i, msg in enumerate(synced):
            if msg is not None:
                arr[i] = _fval(msg, fname)
        out[key] = arr

    return out
