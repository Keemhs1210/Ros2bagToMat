"""
utils.py ─ BagReader / Synchronizer / 공통 헬퍼

역할:
  - BagReader   : ROS2 bag (.db3) 을 단일 패스로 읽어 메모리에 캐싱
  - Synchronizer: 모든 토픽 메시지를 지정 Hz 기준 시간 그리드에 ZOH 동기화
  - _fval       : 메시지 필드 → float 안전 추출
  - _extract_flat: flat 메시지 → scalar 필드를 PascalCase 이름의 1D 배열로 변환

지원 백엔드 (자동 감지):
  1순위 rosbags  ─ pip install rosbags  (Windows 포함, ROS2 불필요)
  2순위 rosbag2_py ─ ROS2 Linux 환경
"""

from __future__ import annotations

import dataclasses
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────
# PascalCase 변환 헬퍼
#   목적: ROS2 snake_case 필드명 → MATLAB 참조 파일과 호환되는 PascalCase 변환
# ─────────────────────────────────────────────────────────────

# Mobileye 토픽 전용 붙여쓴 복합어 → 언더스코어 분리 매핑
# 예) lanemarkposition → lane_mark_position → LaneMarkPosition
# 긴 토큰이 짧은 토큰을 포함하므로 긴 것부터 먼저 등록해야 올바르게 매핑됨
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

# 전체 대문자로 표기할 약어 목록 (소문자 기준)
# 예) me → ME, ttc → TTC, rss → RSS
_ACRONYMS_UPPER: Set[str] = {"me", "ldw", "da", "rss", "fcw", "ttc", "cp"}


def _to_pascal(name: str) -> str:
    """
    snake_case 문자열을 PascalCase 로 변환한다.
    - _COMPOUND_SPLITS 에 등록된 복합어는 먼저 분리 후 각 단어를 변환
    - _ACRONYMS_UPPER 에 등록된 약어는 전체 대문자로 변환
    예) "lanemarkposition_c0_lh_me" → "LaneMarkPositionC0LhME"
    """
    parts: List[str] = []
    for tok in name.split("_"):
        if not tok:
            continue
        if tok in _COMPOUND_SPLITS:
            # 복합어를 하위 토큰으로 분리하여 각각 처리
            for sub in _COMPOUND_SPLITS[tok].split("_"):
                parts.append(sub.upper() if sub.lower() in _ACRONYMS_UPPER
                              else sub[0].upper() + sub[1:])
        elif tok.lower() in _ACRONYMS_UPPER:
            parts.append(tok.upper())
        else:
            parts.append(tok[0].upper() + tok[1:])
    return "".join(parts)


# ─────────────────────────────────────────────────────────────
# 백엔드 자동 선택
#   rosbags 설치 여부를 확인하고, 없으면 rosbag2_py 로 폴백
# ─────────────────────────────────────────────────────────────

_BACKEND: str = ""

# 커스텀 .msg 파일 디렉토리 (프로젝트 루트/ros2_msg/msg/)
# typestore 등록 시 이 폴더의 *.msg 파일을 모두 로드함
_MSG_DIR: Path = Path(__file__).parent.parent / "ros2_msg" / "msg"

try:
    # ── 1순위: rosbags ──────────────────────────────────────
    from rosbags.rosbag2 import Reader as _Rosbag2Reader          # type: ignore
    try:
        # rosbags 신버전 (typesys 모듈 있음)
        from rosbags.typesys import Stores as _Stores             # type: ignore
        from rosbags.typesys import get_typestore as _get_typestore  # type: ignore
        try:
            from rosbags.typesys import get_types_from_msg as _get_types_from_msg  # type: ignore
        except ImportError:
            _get_types_from_msg = None
        _ROSBAGS_NEW = True
    except ImportError:
        # rosbags 구버전 (serde 모듈로 역직렬화)
        try:
            from rosbags.serde import deserialize_cdr as _serde_deser  # type: ignore
        except ImportError:
            from rosbags.serde.cdr import deserialize_cdr as _serde_deser  # type: ignore
        _ROSBAGS_NEW = False
    _BACKEND = "rosbags"

except ImportError:
    # ── 2순위: rosbag2_py (ROS2 Linux 환경 전용) ───────────
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
    ROS2 bag (.db3) 단일 패스(single-pass) 읽기 클래스.

    사용법:
        reader = BagReader(bag_path)
        reader.read_all(["/topic_a", "/topic_b"])   # 한 번만 파일 순회
        msgs, timestamps = reader.get("/topic_a")   # 캐시에서 꺼냄

    특징:
        - read_all() 한 번으로 여러 토픽을 동시에 읽어 I/O 최소화
        - rosbags / rosbag2_py 백엔드를 동일한 인터페이스로 지원
        - 역직렬화 실패 메시지는 경고 출력 후 스킵 (나머지 메시지 유지)
    """

    def __init__(self, bag_path: str | Path):
        self._bag_path = Path(bag_path)
        # {토픽명: 메시지 타입 문자열} — available_topics 또는 read_all 호출 시 채워짐
        self._type_map: Dict[str, str] = {}
        # {토픽명: (메시지 리스트, 타임스탬프 배열)} — read_all 호출 후 채워짐
        self._cache: Dict[str, Tuple[List[Any], np.ndarray]] = {}

    # ── rosbags 내부 헬퍼 ─────────────────────────────────────

    def _rosbags_typestore(self, reader):
        """
        rosbags typestore 를 생성하고 커스텀 .msg 타입을 등록한다.
        ROS2_FOXY 기본 타입 + bag 내장 msgdef + ros2_msg/msg/*.msg 파일 순서로 등록.
        """
        if not _ROSBAGS_NEW:
            return None  # 구버전은 typestore 없이 역직렬화

        ts = _get_typestore(_Stores.ROS2_FOXY)
        if _get_types_from_msg is None:
            return ts

        add = {}

        # 1) bag 에 내장된 msgdef 로 커스텀 타입 등록 시도
        for conn in reader.connections:
            msgdef = getattr(conn, "msgdef", None) or ""
            if msgdef:
                try:
                    add.update(_get_types_from_msg(msgdef, conn.msgtype))
                except Exception:
                    pass

        # 2) ros2_msg/msg/*.msg 파일에서 커스텀 타입 직접 등록
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
                print("  [WARN] typestore 등록 실패:")
                print(traceback.format_exc())
        return ts

    def _rosbags_deser(self, ts, rawdata: bytes, msgtype: str):
        """CDR 바이너리를 typestore 로 역직렬화한다."""
        if _ROSBAGS_NEW:
            return ts.deserialize_cdr(rawdata, msgtype)
        return _serde_deser(rawdata, msgtype)

    # ── rosbag2_py 내부 헬퍼 ─────────────────────────────────

    def _rosbag2py_reader(self):
        """rosbag2_py SequentialReader 를 열고 반환한다. _type_map 도 채운다."""
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
        """bag 에 존재하는 {토픽명: 메시지 타입} 딕셔너리를 반환한다."""
        if not self._type_map:
            if _BACKEND == "rosbags":
                with _Rosbag2Reader(str(self._bag_path)) as r:
                    self._type_map = {c.topic: c.msgtype for c in r.connections}
            else:
                self._rosbag2py_reader()
        return self._type_map

    def read_all(self, topics: List[str]) -> None:
        """
        지정한 토픽 목록을 단일 패스로 읽어 내부 캐시에 저장한다.
        이후 get() 으로 캐시된 데이터를 꺼낼 수 있다.
        """
        if _BACKEND == "rosbags":
            self._read_rosbags(topics)
        else:
            self._read_rosbag2py(topics)

    def _read_rosbags(self, topics: List[str]) -> None:
        """rosbags 백엔드로 단일 패스 읽기."""
        with _Rosbag2Reader(str(self._bag_path)) as reader:
            self._type_map = {c.topic: c.msgtype for c in reader.connections}
            # bag 에 실제로 존재하는 토픽만 필터링
            valid = [t for t in topics if t in self._type_map]
            if not valid:
                self._cache = {}
                return

            ts    = self._rosbags_typestore(reader)
            conns = [c for c in reader.connections if c.topic in valid]
            raw: Dict[str, Tuple[List, List]] = {t: ([], []) for t in valid}
            err_count: Dict[str, int] = {}

            for conn, ts_ns, rawdata in reader.messages(connections=conns):
                if conn.topic not in raw:
                    continue
                try:
                    msg = self._rosbags_deser(ts, rawdata, conn.msgtype)
                    raw[conn.topic][0].append(msg)
                    raw[conn.topic][1].append(ts_ns * 1e-9)  # ns → sec
                except Exception:
                    cnt = err_count.get(conn.topic, 0)
                    if cnt == 0:
                        # 첫 실패만 출력 (이후는 카운트만 집계)
                        print(f"\n  [WARN] 역직렬화 실패 ({conn.topic}): "
                              f"{traceback.format_exc().splitlines()[-1]}")
                    err_count[conn.topic] = cnt + 1

            # 토픽별 누적 실패 수 요약 출력
            for topic, cnt in err_count.items():
                ok = len(raw.get(topic, ([],))[0])
                print(f"  [WARN] {topic}: {cnt}개 메시지 스킵 (성공 {ok}개)")

        self._cache = {
            t: (msgs, np.array(tss, dtype=np.float64))
            for t, (msgs, tss) in raw.items()
        }

    def _read_rosbag2py(self, topics: List[str]) -> None:
        """rosbag2_py 백엔드로 단일 패스 읽기."""
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
        """
        캐시에서 topic 데이터를 반환한다.
        read_all() 에서 읽히지 않은 토픽은 빈 결과를 반환한다.
        """
        return self._cache.get(topic, ([], np.array([], dtype=np.float64)))


# ─────────────────────────────────────────────────────────────
# Synchronizer (Zero-Order Hold)
# ─────────────────────────────────────────────────────────────

class Synchronizer:
    """
    모든 토픽 메시지를 지정 Hz 기준 시간 그리드에 ZOH(Zero-Order Hold) 동기화한다.

    동작 방식:
      1) 각 목표 시각(t_target)에 가장 가까운 메시지 인덱스를 찾는다.
      2) 시간 오차가 sample_time × 2 이내인 경우에만 할당한다.
      3) 할당되지 않은 슬롯은 이전 유효 값으로 forward-fill (ZOH) 한다.
    """

    def __init__(self, start_time: float, sample_time: float, signal_length: int):
        self.start_time    = start_time    # 동기화 시작 시각 (초)
        self.sample_time   = sample_time   # 샘플 주기 (초), 예: 0.05 = 20 Hz
        self.signal_length = signal_length # 전체 스텝 수
        # 0부터 시작하는 상대 시간 배열 [0, dt, 2dt, ..., (N-1)*dt]
        self.sync_time = np.arange(signal_length, dtype=np.float64) * sample_time

    def sync(
        self,
        messages:   List[Any],
        timestamps: np.ndarray,
    ) -> List[Optional[Any]]:
        """
        messages 를 self.sync_time 기준 그리드에 동기화하여 반환한다.
        메시지가 없거나 시간 오차가 큰 슬롯은 ZOH forward-fill 로 채운다.
        """
        synced: List[Optional[Any]] = [None] * self.signal_length

        if not messages:
            return synced

        target_times = self.start_time + self.sync_time

        for j, t_target in enumerate(target_times):
            # 목표 시각과 가장 가까운 메시지 인덱스
            idx = int(np.argmin(np.abs(timestamps - t_target)))
            dt  = timestamps[idx] - t_target
            if abs(dt) <= self.sample_time * 2:
                # dt > 0 이면 목표 시각보다 미래 메시지 → 이전 메시지 사용 (ZOH)
                synced[j] = messages[idx - 1] if (dt > 0 and idx > 0) else messages[idx]

        # ZOH forward-fill: None 슬롯을 직전 유효 값으로 채움
        last: Optional[Any] = None
        for j in range(self.signal_length):
            if synced[j] is not None:
                last = synced[j]
            elif last is not None:
                synced[j] = last

        return synced


# ─────────────────────────────────────────────────────────────
# 공통 헬퍼 함수
# ─────────────────────────────────────────────────────────────

# _extract_flat 에서 제외할 내부 필드명 (Header, MessageType 등)
_SKIP = {"header", "Header", "MessageType"}


def _fval(msg: Any, name: str, default: float = 0.0) -> float:
    """
    ROS2 메시지 객체에서 단일 float 값을 안전하게 추출한다.
    - 중첩 struct, 배열, None, 문자열인 경우 default 를 반환
    - float 변환 불가 시 default 를 반환
    """
    v = getattr(msg, name, default)
    if v is None:
        return default
    try:
        dataclasses.fields(v)  # type: ignore[arg-type]
        return default          # 중첩 struct → 기본값 반환
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
    동기화된 flat 메시지 리스트에서 scalar 필드를 1D numpy 배열로 추출한다.

    Args:
        synced : Synchronizer.sync() 반환값 (Optional[msg] 리스트, 길이 n)
        n      : 신호 길이 (배열 크기)
        prefix : MATLAB 변수명 앞에 붙일 접두사 (예: "ME_Lane_Left_Lane_A_")

    Returns:
        {PascalCase 필드명: np.ndarray(n)} 딕셔너리
        중첩 struct, 배열 필드, 내부 필드(_로 시작), 문자열 필드는 제외됨
    """
    sample = next((m for m in synced if m is not None), None)
    if sample is None:
        return {}

    # 메시지 객체의 모든 필드명 수집 (dataclass 또는 일반 object)
    try:
        all_fields = [f.name for f in dataclasses.fields(sample)]  # type: ignore[arg-type]
    except TypeError:
        all_fields = list(vars(sample).keys())

    # scalar 필드만 추려냄
    scalar_fields: List[str] = []
    for fname in all_fields:
        if fname.startswith("_") or fname in _SKIP:
            continue  # 내부 필드 (_msgtype_ 등) 제외
        val = getattr(sample, fname, None)
        if val is None or isinstance(val, str):
            continue  # None 또는 문자열 제외
        try:
            dataclasses.fields(val)  # type: ignore[arg-type]
            continue  # 중첩 struct 제외
        except TypeError:
            pass
        if isinstance(val, (np.ndarray, list, tuple, bytes, bytearray)):
            continue  # 배열 타입 제외
        scalar_fields.append(fname)

    # 각 scalar 필드를 PascalCase 이름의 numpy 배열로 변환
    out: Dict[str, np.ndarray] = {}
    for fname in scalar_fields:
        key = f"{prefix}{_to_pascal(fname)}" if prefix else _to_pascal(fname)
        arr = np.zeros(n, dtype=np.float64)
        for i, msg in enumerate(synced):
            if msg is not None:
                arr[i] = _fval(msg, fname)
        out[key] = arr

    return out
