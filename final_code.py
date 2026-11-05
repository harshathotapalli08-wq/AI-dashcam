#!/usr/bin/env python3
import collections
import shutil
import os
import csv
import queue
from datetime import datetime
 
# ============================================================
# CPU CONTROL
# ============================================================
# Keep BLAS/OpenMP from creating extra worker threads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import cv2

# Keep OpenCV itself from consuming all cores.
cv2.setNumThreads(1)

import numpy as np
import time
import threading
import argparse
from pathlib import Path

import torch

# Ultralytics/NCNN internally uses torch helpers.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# pyrefly: ignore [missing-import]
import onnxruntime as ort
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# pyrefly: ignore [missing-import]
from detectors.eye_yawn_detector import EyeYawnDetector


# ============================================================
# INCIDENT VIDEO RECORDER
# ============================================================

class IncidentRecorder:
    """Save annotated incident clips with pre-roll and post-roll footage.

    Call add_frame() for every annotated main-loop frame. Call
    on_accident_detected() while an accident/collision alert is active.
    A separate worker writes video so disk I/O does not normally block
    the camera loop. The live-frame queue is bounded; dropped frames
    are counted and reported rather than allowed to grow without limit.
    """

    def __init__(
        self,
        output_dir,
        fallback_dir,
        pre_seconds=15.0,
        post_seconds=15.0,
        max_clip_seconds=120.0,
        fps_hint=25.0,
        fourcc="mp4v",
        min_free_gb=1.0,
        queue_size=60,
        on_clip_saved=None,
    ):
        if pre_seconds < 0 or post_seconds < 0:
            raise ValueError("pre_seconds and post_seconds must be >= 0")
        if max_clip_seconds <= 0:
            raise ValueError("max_clip_seconds must be > 0")
        if pre_seconds > max_clip_seconds:
            raise ValueError("pre_seconds cannot exceed max_clip_seconds")
        if fps_hint <= 0:
            raise ValueError("fps_hint must be > 0")
        if min_free_gb < 0 or queue_size < 1:
            raise ValueError("min_free_gb must be >= 0 and queue_size >= 1")
        if len(fourcc) != 4:
            raise ValueError("fourcc must contain exactly four characters")

        self.requested_dir = Path(output_dir).expanduser()
        self.fallback_dir = Path(fallback_dir).expanduser()
        self.pre_seconds = float(pre_seconds)
        self.post_seconds = float(post_seconds)
        self.max_clip_seconds = float(max_clip_seconds)
        self.min_free_gb = float(min_free_gb)
        self.fourcc = fourcc
        self.on_clip_saved = on_clip_saved

        self._lock = threading.Lock()
        self._accepting_frames = True
        self._buffer = collections.deque()  # (monotonic timestamp, owned frame)
        self._frame_times = collections.deque(maxlen=60)
        self._frame_size = None             # (width, height)
        self._clip_fps = float(fps_hint)

        self._recording = False
        self._alert_active = False
        self._ignore_until_clear = False
        self._clip_token = 0
        self._record_started_at = None
        self._record_until = None
        self._clip_deadline = None

        # (incident token, frame timestamp, frame). A bounded queue limits RAM.
        self._frames = queue.Queue(maxsize=int(queue_size))
        # Only one START command is created per incident; active detections
        # update _record_until directly rather than flooding a command queue.
        self._controls = queue.SimpleQueue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._dropped_frames = 0
        self._last_drop_warning = 0.0
        self._last_size_warning = 0.0

        self.output_dir = self._resolve_output_dir(self.requested_dir)
        self._worker = threading.Thread(
            target=self._writer_loop,
            name="IncidentRecorderWriter",
            daemon=False,
        )
        self._worker.start()
        print(
            "[IncidentRecorder] Ready: "
            f"pre={self.pre_seconds:.1f}s, post={self.post_seconds:.1f}s, "
            f"max_clip={self.max_clip_seconds:.1f}s"
        )

    # ------------------------- storage helpers -------------------------

    @staticmethod
    def _probe_directory(path):
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".incident_probe_{threading.get_ident()}_{time.monotonic_ns()}"
        try:
            with probe.open("xb") as handle:
                handle.write(b"ok")
        finally:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass
        return path

    def _resolve_output_dir(self, preferred):
        try:
            path = self._probe_directory(preferred)
            print(f"[IncidentRecorder] Saving incidents to: {path}")
            return path
        except OSError as exc:
            print(
                f"[IncidentRecorder] WARNING: '{preferred}' unavailable "
                f"({exc}); trying fallback '{self.fallback_dir}'."
            )

        try:
            path = self._probe_directory(self.fallback_dir)
            print(f"[IncidentRecorder] Using fallback directory: {path}")
            return path
        except OSError as exc:
            raise OSError(
                "Neither the primary nor fallback incident directory is writable."
            ) from exc

    @staticmethod
    def _free_gb(path):
        try:
            return shutil.disk_usage(path).free / (1024 ** 3)
        except OSError:
            return -1.0

    def current_output_dir(self):
        """Return the currently selected output directory."""
        return self.output_dir

    def is_recording(self):
        with self._lock:
            return self._recording

    def dropped_frame_count(self):
        with self._lock:
            return self._dropped_frames

    # ------------------------- public methods -------------------------

    def add_frame(self, frame):
        """Add each processed frame (preferably after all overlays are drawn)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return

        now = time.monotonic()
        copied = frame.copy()
        height, width = copied.shape[:2]
        size = (width, height)

        with self._lock:
            if not self._accepting_frames:
                return

            if self._frame_size is None:
                self._frame_size = size
            elif size != self._frame_size:
                if now - self._last_size_warning >= 5.0:
                    self._last_size_warning = now
                    print(
                        "[IncidentRecorder] ERROR: Frame size changed from "
                        f"{self._frame_size} to {size}; frame rejected. "
                        "Keep camera resolution fixed."
                    )
                return

            self._buffer.append((now, copied))
            cutoff = now - self.pre_seconds
            while self._buffer and self._buffer[0][0] < cutoff:
                self._buffer.popleft()

            self._frame_times.append(now)
            if len(self._frame_times) >= 2:
                span = self._frame_times[-1] - self._frame_times[0]
                if span > 0:
                    self._clip_fps = max(1.0, (len(self._frame_times) - 1) / span)

            recording = self._recording
            token = self._clip_token if recording else None

        if recording:
            try:
                self._frames.put_nowait((token, now, copied))
            except queue.Full:
                with self._lock:
                    self._dropped_frames += 1
                    dropped = self._dropped_frames
                if now - self._last_drop_warning >= 5.0:
                    self._last_drop_warning = now
                    print(
                        "[IncidentRecorder] WARNING: writer queue full; "
                        f"dropped frames so far: {dropped}. Lower resolution/FPS "
                        "or use faster storage if this repeats."
                    )

        self._wake.set()

    def on_accident_detected(self):
        """Start a clip on the first alert; extend its post-roll on later alerts."""
        now = time.monotonic()

        with self._lock:
            if not self._accepting_frames:
                return

            self._alert_active = True

            # If a continuous alert hit the hard clip limit, do not start
            # another clip until the application reports that the alert cleared.
            if self._ignore_until_clear:
                return

            if self._frame_size is None:
                print("[IncidentRecorder] Trigger ignored: no frames received yet.")
                return

            if not self._recording:
                self._clip_token += 1
                token = self._clip_token
                # Frames already copied into the ring buffer are immutable from
                # the recorder's point of view, so retain references here rather
                # than duplicating up to 15 seconds of high-resolution frames.
                preroll = [
                    (timestamp, frame)
                    for timestamp, frame in self._buffer
                    if timestamp >= now - self.pre_seconds
                ]
                actual_preroll = (
                    max(0.0, now - preroll[0][0]) if preroll else 0.0
                )
                deadline = now + max(0.0, self.max_clip_seconds - actual_preroll)
                self._recording = True
                self._record_started_at = now
                self._clip_deadline = deadline
                self._record_until = min(now + self.post_seconds, deadline)
                fps = self._clip_fps
                self._controls.put(
                    ("start", token, preroll, fps, now, self._record_until)
                )
                print(
                    "[IncidentRecorder] Accident/collision trigger. "
                    f"Pre-roll frames={len(preroll)}."
                )
            else:
                self._record_until = min(
                    now + self.post_seconds, self._clip_deadline
                )

        self._wake.set()

    def on_accident_cleared(self):
        """Notify the recorder that the accident/collision alert is no longer active.

        This does not stop post-roll recording. It only clears the safety latch
        used after max_clip_seconds so a later distinct alert can start a new clip.
        """
        with self._lock:
            self._alert_active = False
            self._ignore_until_clear = False
        self._wake.set()

    def shutdown(self):
        """Stop accepting frames and flush queued frames before finalizing."""
        with self._lock:
            if not self._accepting_frames:
                return
            self._accepting_frames = False

        self._stop.set()
        self._wake.set()
        self._worker.join()  # do not release writer while worker is still writing
        print(
            "[IncidentRecorder] Shutdown complete; "
            f"dropped frames={self.dropped_frame_count()}"
        )

    # ------------------------- writer helpers -------------------------

    def _open_writer(self, fps):
        try:
            self.output_dir = self._resolve_output_dir(self.requested_dir)
        except OSError as exc:
            print(f"[IncidentRecorder] ERROR: no writable output directory: {exc}")
            return None, None, None

        free = self._free_gb(self.output_dir)
        if 0 <= free < self.min_free_gb:
            print(
                f"[IncidentRecorder] WARNING: only {free:.2f} GB free at "
                f"{self.output_dir}; trying the fallback directory."
            )
            try:
                self.output_dir = self._probe_directory(self.fallback_dir)
            except OSError as exc:
                print(f"[IncidentRecorder] ERROR: fallback unavailable: {exc}")
                return None, None, None
            free = self._free_gb(self.output_dir)
            if 0 <= free < self.min_free_gb:
                print(
                    f"[IncidentRecorder] ERROR: fallback has only {free:.2f} GB "
                    "free; refusing to start this clip."
                )
                return None, None, None

        with self._lock:
            frame_size = self._frame_size
        if frame_size is None:
            return None, None, None

        filename = datetime.now().strftime("incident_%Y%m%d_%H%M%S_%f.mp4")
        path = self.output_dir / filename
        writer = None
        try:
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*self.fourcc),
                max(1.0, float(fps)),
                frame_size,
            )
            if not writer.isOpened():
                writer.release()
                print(
                    f"[IncidentRecorder] ERROR: VideoWriter could not open {path}. "
                    "Check OpenCV codec support and storage."
                )
                return None, None, None
        except Exception as exc:
            if writer is not None:
                writer.release()
            print(f"[IncidentRecorder] ERROR: could not create {path}: {exc}")
            return None, None, None

        print(f"[IncidentRecorder] Recording incident to: {path}")
        return writer, path, frame_size

    @staticmethod
    def _write_frame(writer, frame, frame_size):
        if writer is None:
            return False
        height, width = frame.shape[:2]
        if (width, height) != frame_size:
            return False
        try:
            writer.write(frame)
            return True
        except Exception as exc:
            print(f"[IncidentRecorder] Frame write error: {exc}")
            return False

    def _finish_clip(self, writer, path, token, failed=False):
        if writer is not None:
            try:
                writer.release()
            except Exception as exc:
                failed = True
                print(f"[IncidentRecorder] Writer release error: {exc}")

        valid = False
        try:
            valid = path is not None and path.exists() and path.stat().st_size > 0
        except OSError:
            pass

        if valid and not failed:
            print(f"[IncidentRecorder] Clip finalized: {path}")
            if self.on_clip_saved is not None:
                try:
                    self.on_clip_saved(path)
                except Exception as exc:
                    print(f"[IncidentRecorder] on_clip_saved callback error: {exc}")
        else:
            print(
                "[IncidentRecorder] WARNING: clip may be incomplete or empty: "
                f"{path}. Verify it with a video player."
            )

    # ------------------------- background writer -------------------------

    def _writer_loop(self):
        writer = None
        writer_path = None
        writer_token = None
        writer_size = None
        active_deadline = None
        clip_failed = False

        while True:
            self._wake.wait(timeout=0.05)
            self._wake.clear()

            # START is queued once per incident, before subsequent live frames
            # are consumed. Extend requests update shared state under _lock.
            while True:
                try:
                    command = self._controls.get_nowait()
                except queue.Empty:
                    break

                kind = command[0]
                if kind != "start":
                    continue

                _, token, preroll, fps, trigger_time, post_deadline = command

                if writer is not None:
                    self._finish_clip(writer, writer_path, writer_token, clip_failed)

                writer, writer_path, writer_size = self._open_writer(fps)
                writer_token = token
                active_deadline = post_deadline
                clip_failed = writer is None

                if writer is not None:
                    for timestamp, frame in preroll:
                        if timestamp <= trigger_time:
                            if not self._write_frame(writer, frame, writer_size):
                                clip_failed = True
                                print("[IncidentRecorder] Pre-roll write failed.")
                                break

            # Drain live frames in FIFO order. The recording deadline is read
            # for each frame because the detector can extend post-roll at any time.
            while True:
                try:
                    token, timestamp, frame = self._frames.get_nowait()
                except queue.Empty:
                    break

                try:
                    if token != writer_token or writer is None:
                        continue
                    with self._lock:
                        allowed_until = (
                            self._record_until
                            if self._clip_token == token and self._record_until is not None
                            else active_deadline
                        )
                    if allowed_until is not None and timestamp > allowed_until:
                        continue
                    if not self._write_frame(writer, frame, writer_size):
                        clip_failed = True
                        print(
                            "[IncidentRecorder] Live write failed; "
                            "saved clip may be incomplete."
                        )
                finally:
                    self._frames.task_done()

            with self._lock:
                current_token = self._clip_token
                recording = self._recording
                current_deadline = self._record_until

            if current_token == writer_token and recording:
                active_deadline = current_deadline

            expired = (
                active_deadline is not None
                and time.monotonic() >= active_deadline
            )
            stopping = self._stop.is_set()

            if writer is not None and (expired or stopping):
                # Final queue drain before releasing the MP4 writer.
                while True:
                    try:
                        token, timestamp, frame = self._frames.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if token != writer_token:
                            continue
                        with self._lock:
                            allowed_until = (
                                self._record_until
                                if self._clip_token == token and self._record_until is not None
                                else active_deadline
                            )
                        if not stopping and allowed_until is not None and timestamp > allowed_until:
                            continue
                        if not self._write_frame(writer, frame, writer_size):
                            clip_failed = True
                    finally:
                        self._frames.task_done()

                finished_token = writer_token
                self._finish_clip(writer, writer_path, writer_token, clip_failed)
                writer = None
                writer_path = None
                writer_token = None
                writer_size = None
                active_deadline = None
                clip_failed = False

                with self._lock:
                    if self._clip_token == finished_token:
                        hit_cap_while_alert_active = (
                            not stopping
                            and self._alert_active
                            and self._clip_deadline is not None
                            and self._record_until is not None
                            and time.monotonic() >= self._clip_deadline
                            and self._record_until >= self._clip_deadline
                        )
                        if hit_cap_while_alert_active:
                            self._ignore_until_clear = True
                            print(
                                "[IncidentRecorder] Maximum clip duration reached "
                                "while alert is still active; no new clip will start "
                                "until the alert clears."
                            )
                        self._recording = False
                        self._record_started_at = None
                        self._record_until = None
                        self._clip_deadline = None

            elif writer is None and writer_token is not None and expired:
                # Writer failed to open; reset state when that attempted
                # incident expires so a later incident can retry.
                failed_token = writer_token
                with self._lock:
                    if self._clip_token == failed_token:
                        hit_cap_while_alert_active = (
                            self._alert_active
                            and self._clip_deadline is not None
                            and self._record_until is not None
                            and time.monotonic() >= self._clip_deadline
                            and self._record_until >= self._clip_deadline
                        )
                        if hit_cap_while_alert_active:
                            self._ignore_until_clear = True
                            print(
                                "[IncidentRecorder] Maximum clip duration reached "
                                "while alert remains active; suppressing repeated starts."
                            )
                        self._recording = False
                        self._record_started_at = None
                        self._record_until = None
                        self._clip_deadline = None
                writer_token = None
                active_deadline = None

            if stopping and self._controls.empty() and self._frames.empty():
                if writer is not None:
                    self._finish_clip(writer, writer_path, writer_token, clip_failed)
                with self._lock:
                    self._recording = False
                    self._record_started_at = None
                    self._record_until = None
                    self._clip_deadline = None
                break



# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_VIDEO = PROJECT_ROOT / "collision.webm"


DEFAULT_OUTPUT = (
    PROJECT_ROOT /
    "all_models_fps_fixed_output.mp4"
)

DMS_MODEL = (
    PROJECT_ROOT /
    "models/yolo/best.onnx"
)

VEHICLE_MODEL = (
    Path.home() /
    "Downloads/vehicle_person_extract/best_ncnn_model"
)

ACCIDENT_MODEL = (
    PROJECT_ROOT /
    "models/yolo/best_ncnn_model"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_OUTPUT_FPS = 25.0

# Each model remains enabled.
# These are scheduling rates, not disabling switches.

MEDIAPIPE_FPS = 20.0
DMS_FPS = 5.0
VEHICLE_FPS = 8.0
ACCIDENT_FPS = 1.5

MEDIAPIPE_INTERVAL = 1.0 / MEDIAPIPE_FPS
DMS_INTERVAL = 1.0 / DMS_FPS
VEHICLE_INTERVAL = 1.0 / VEHICLE_FPS
ACCIDENT_INTERVAL = 1.0 / ACCIDENT_FPS

DMS_CONF = 0.50
VEHICLE_CONF = 0.20
ACCIDENT_CONF = 0.15

VEHICLE_MAX_DET = 20
ACCIDENT_MAX_DET = 20

# Lower NCNN YOLO input size reduces CPU load on Raspberry Pi 5.
# DMS ONNX remains at its required 640x640 input size.
YOLO_IMGSZ = 416

ACCIDENT_HOLD_SECONDS = 2.0


# ============================================================
# CLASS DEFINITIONS
# ============================================================

DMS_CLASS_NAMES = {
    0: "phone",
    1: "seatbelt",
    2: "cigarette",
    3: "smoking",
}

PERSON_CLASSES = {
    0,
    1,
}

VEHICLE_CLASSES = {
    2,
    3,
    4,
    5,
    6,
}

ACCIDENT_CLASSES = {
    "acc",
    "accident",
    "car-accident",
    "motor-accident",
    "people-accident",
}


# ============================================================
# GLOBAL FRAME STATE
# ============================================================

latest_frame = None
latest_frame_id = -1

frame_lock = threading.Lock()

result_lock = threading.Lock()

stop_event = threading.Event()

# ============================================================
# BMI323 + BME280 SENSOR STATE
# ============================================================

sensor_lock = threading.Lock()
sensor_result = {
    "status": "Starting sensors...",
    "accel": None,
    "gyro": None,
    "bmi_temp": None,
    "temperature": None,
    "pressure": None,
    "humidity": None,
}


# ============================================================
# QUECTEL EC200U-CN GNSS + GSM/LTE STATE
# ============================================================

gps_lock = threading.Lock()
gps_result = {
    "status": "GPS starting...",
    "latitude": None,
    "longitude": None,
    "altitude_m": None,
    "speed_kmh": None,
    "utc": None,
    "satellites": None,
    "gsm_status": "Unknown",
    "signal_dbm": None,
    "last_update": None,
}


# ============================================================
# GLOBAL RESULTS
# ============================================================

mp_result = {}

dms_result = []

vehicle_result = []

accident_result = []


# ============================================================
# GLOBAL TIMING
# ============================================================

mp_time = 0.0
dms_time = 0.0
vehicle_time = 0.0
accident_time = 0.0

mp_fps = 0.0
dms_fps = 0.0
vehicle_fps = 0.0
accident_fps = 0.0


# ============================================================
# ACCIDENT STATE
# ============================================================

last_accident_detection_time = 0.0


# ============================================================
# COLLISION FALLBACK STATE
# ============================================================

previous_vehicle_objects = []

collision_counter = 0

COLLISION_CONFIRM = 3


# ============================================================
# LANE STATE
# ============================================================

previous_left_line = None

previous_right_line = None

lane_departure_counter = 0


# ============================================================
# BMI323 + BME280 SENSOR READER
# ============================================================

def sensor_read_reg(bus, address, register, count, dummy_bytes=0):
    """Read I2C registers; BMI323 needs two dummy bytes on reads."""
    # pyrefly: ignore [missing-import]
    from smbus2 import i2c_msg
    write_msg = i2c_msg.write(address, [register])
    read_msg = i2c_msg.read(address, count + dummy_bytes)
    bus.i2c_rdwr(write_msg, read_msg)
    return list(read_msg)[dummy_bytes:]


def sensor_write_reg(bus, address, register, value):
    # pyrefly: ignore [missing-import]
    from smbus2 import i2c_msg
    bus.i2c_rdwr(i2c_msg.write(address, [register, value & 0xFF]))


def sensor_write_bmi16(bus, register, value):
    # pyrefly: ignore [missing-import]
    from smbus2 import i2c_msg
    bus.i2c_rdwr(i2c_msg.write(
        0x68, [register, value & 0xFF, (value >> 8) & 0xFF]
    ))


def sensor_u16(data, index=0):
    return data[index] | (data[index + 1] << 8)


def sensor_s16(data, index=0):
    value = sensor_u16(data, index)
    return value - 65536 if value >= 32768 else value


def sensor_s12(value):
    return value - 4096 if value & 0x800 else value


def sensor_init_bmi(bus):
    chip_id = sensor_read_reg(bus, 0x68, 0x00, 1, dummy_bytes=2)[0]
    if chip_id != 0x43:
        raise RuntimeError(f"BMI323 ID mismatch: 0x{chip_id:02X}, expected 0x43")

    # Configure accelerometer and gyroscope using 16-bit BMI323 registers.
    sensor_write_bmi16(bus, 0x20, 0x4027)
    sensor_write_bmi16(bus, 0x21, 0x4047)
    time.sleep(0.1)
    print(f"[SENSORS] BMI323 detected (ID 0x{chip_id:02X})")


def sensor_read_bmi(bus):
    # 7 16-bit words: accelerometer XYZ, gyroscope XYZ, temperature.
    data = sensor_read_reg(bus, 0x68, 0x03, 14, dummy_bytes=2)
    ax, ay, az = (sensor_s16(data, i) / 4096.0 for i in (0, 2, 4))
    gx, gy, gz = (sensor_s16(data, i) / 16.384 for i in (6, 8, 10))
    bmi_temp = sensor_s16(data, 12) / 512.0 + 23.0
    return (ax, ay, az), (gx, gy, gz), bmi_temp


def sensor_init_bme(bus):
    chip_id = sensor_read_reg(bus, 0x76, 0xD0, 1)[0]
    if chip_id != 0x60:
        raise RuntimeError(f"BME280 ID mismatch: 0x{chip_id:02X}, expected 0x60")

    a = sensor_read_reg(bus, 0x76, 0x88, 26)
    h = sensor_read_reg(bus, 0x76, 0xE1, 7)
    c = {
        "T1": sensor_u16(a, 0), "T2": sensor_s16(a, 2), "T3": sensor_s16(a, 4),
        "P1": sensor_u16(a, 6), "P2": sensor_s16(a, 8), "P3": sensor_s16(a, 10),
        "P4": sensor_s16(a, 12), "P5": sensor_s16(a, 14), "P6": sensor_s16(a, 16),
        "P7": sensor_s16(a, 18), "P8": sensor_s16(a, 20), "P9": sensor_s16(a, 22),
        "H1": a[25], "H2": sensor_s16(h, 0), "H3": h[2],
        "H4": sensor_s12((h[3] << 4) | (h[4] & 0x0F)),
        "H5": sensor_s12((h[5] << 4) | (h[4] >> 4)),
        "H6": h[6] - 256 if h[6] > 127 else h[6],
    }

    sensor_write_reg(bus, 0x76, 0xF2, 0x01)  # humidity oversampling x1
    sensor_write_reg(bus, 0x76, 0xF4, 0x27)  # temp/pressure x1, normal mode
    sensor_write_reg(bus, 0x76, 0xF5, 0xA0)  # standby/filter configuration
    time.sleep(0.1)
    print(f"[SENSORS] BME280 detected (ID 0x{chip_id:02X})")
    return c


def sensor_read_bme(bus, c):
    d = sensor_read_reg(bus, 0x76, 0xF7, 8)
    rp = (d[0] << 12) | (d[1] << 4) | (d[2] >> 4)
    rt = (d[3] << 12) | (d[4] << 4) | (d[5] >> 4)
    rh = (d[6] << 8) | d[7]

    v1 = (rt / 16384.0 - c["T1"] / 1024.0) * c["T2"]
    v2 = (rt / 131072.0 - c["T1"] / 8192.0) ** 2 * c["T3"]
    fine = v1 + v2
    temperature = fine / 5120.0

    v1 = fine / 2.0 - 64000.0
    v2 = v1 * v1 * c["P6"] / 32768.0
    v2 += v1 * c["P5"] * 2.0
    v2 = v2 / 4.0 + c["P4"] * 65536.0
    v1 = (c["P3"] * v1 * v1 / 524288.0 + c["P2"] * v1) / 524288.0
    v1 = (1.0 + v1 / 32768.0) * c["P1"]

    pressure = None
    if v1 != 0:
        p = 1048576.0 - rp
        p = (p - v2 / 4096.0) * 6250.0 / v1
        v1p = c["P9"] * p * p / 2147483648.0
        v2p = p * c["P8"] / 32768.0
        pressure = (p + (v1p + v2p + c["P7"]) / 16.0) / 100.0

    x = fine - 76800.0
    x = (rh - (c["H4"] * 64.0 + c["H5"] / 16384.0 * x)) * (
        c["H2"] / 65536.0 * (
            1.0 + c["H6"] / 67108864.0 * x *
            (1.0 + c["H3"] / 67108864.0 * x)
        )
    )
    x *= 1.0 - c["H1"] * x / 524288.0
    humidity = max(0.0, min(100.0, x))
    return temperature, pressure, humidity


def sensor_worker():
    """Read sensors on a background thread so AI/video stays responsive."""
    global sensor_result
    try:
        # pyrefly: ignore [missing-import]
        from smbus2 import SMBus
    except Exception as exc:
        with sensor_lock:
            sensor_result["status"] = f"smbus2 unavailable: {exc}"
        print("[SENSORS ERROR] Install smbus2 in the Python environment running this script.")
        return

    try:
        with SMBus(1) as bus:
            bmi_ok = bme_ok = False
            calibration = None
            try:
                sensor_init_bmi(bus)
                bmi_ok = True
            except Exception as exc:
                print(f"[BMI323 INIT ERROR] {exc}")
            try:
                calibration = sensor_init_bme(bus)
                bme_ok = True
            except Exception as exc:
                print(f"[BME280 INIT ERROR] {exc}")

            if not bmi_ok and not bme_ok:
                with sensor_lock:
                    sensor_result["status"] = "Both sensors unavailable"
                return

            while not stop_event.is_set():
                updates = {}
                errors = []
                if bmi_ok:
                    try:
                        accel, gyro, bmi_temp = sensor_read_bmi(bus)
                        updates.update(accel=accel, gyro=gyro, bmi_temp=bmi_temp)
                    except Exception as exc:
                        errors.append(f"BMI323 read: {exc}")
                if bme_ok:
                    try:
                        temp, pressure, humidity = sensor_read_bme(bus, calibration)
                        updates.update(temperature=temp, pressure=pressure, humidity=humidity)
                    except Exception as exc:
                        errors.append(f"BME280 read: {exc}")
                with sensor_lock:
                    sensor_result.update(updates)
                    sensor_result["status"] = "; ".join(errors) if errors else "OK"
                    snapshot = dict(sensor_result)
                if updates:
                    if snapshot.get("accel") is not None:
                        ax, ay, az = snapshot["accel"]
                        gx, gy, gz = snapshot["gyro"]
                        print(f"[BMI323] Accel g=({ax:.2f}, {ay:.2f}, {az:.2f}) | Gyro dps=({gx:.1f}, {gy:.1f}, {gz:.1f})")
                    if snapshot.get("temperature") is not None:
                        print(f"[BME280] {snapshot['temperature']:.2f} C | {snapshot['pressure']:.2f} hPa | {snapshot['humidity']:.1f}% RH")
                time.sleep(1.0)
    except Exception as exc:
        with sensor_lock:
            sensor_result["status"] = f"Sensor bus error: {exc}"
        print(f"[SENSORS ERROR] {exc}")


def draw_sensor_overlay(frame):
    """Show the latest sensor values in the video and saved output."""
    with sensor_lock:
        values = dict(sensor_result)

    h, w = frame.shape[:2]
    x = max(10, w - 410)
    y = 42
    panel_w = min(400, w - x - 10)
    panel_h = 112
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)

    lines = []
    if values.get("accel") is not None:
        ax, ay, az = values["accel"]
        gx, gy, gz = values["gyro"]
        lines.append(f"BMI Accel g: {ax:.2f} {ay:.2f} {az:.2f}")
        lines.append(f"BMI Gyro dps: {gx:.1f} {gy:.1f} {gz:.1f}")
    else:
        lines.append("BMI323: waiting / unavailable")

    if values.get("temperature") is not None:
        lines.append(f"BME: {values['temperature']:.1f} C  {values['pressure']:.0f} hPa")
        lines.append(f"Humidity: {values['humidity']:.1f}% RH")
    else:
        lines.append("BME280: waiting / unavailable")

    for i, line in enumerate(lines[:4]):
        cv2.putText(frame, line, (x + 8, y + 23 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================
# QUECTEL EC200U-CN GNSS + GSM/LTE READER
# ============================================================

def gps_at_command(port, command, timeout=2.0):
    """Send one AT command and return its response without blocking the AI threads."""
    port.reset_input_buffer()
    port.write((command + "\r").encode("ascii"))
    port.flush()
    deadline = time.monotonic() + timeout
    chunks = []
    while time.monotonic() < deadline:
        line = port.readline().decode("utf-8", errors="ignore").strip()
        if line:
            chunks.append(line)
            if line == "OK" or line == "ERROR" or "+CME ERROR" in line:
                break
    return "\n".join(chunks)


def gps_parse_qgpsloc(response):
    """Parse Quectel +QGPSLOC response into decimal-degree coordinates."""
    for line in response.splitlines():
        if "+QGPSLOC:" not in line:
            continue
        payload = line.split(":", 1)[1].strip()
        fields = [item.strip() for item in payload.split(",")]
        # EC200U QGPSLOC: UTC, latitude, longitude, HDOP, altitude,
        # fix, course, speed(km/h), speed(knots), date, satellites
        if len(fields) < 8:
            continue
        try:
            return {
                "utc": fields[0],
                "latitude": float(fields[1]),
                "longitude": float(fields[2]),
                "altitude_m": float(fields[4]),
                "speed_kmh": float(fields[7]),
                "satellites": int(fields[10]) if len(fields) > 10 and fields[10] else None,
            }
        except (ValueError, IndexError):
            continue
    return None


def gps_worker(port_name="/dev/ttyUSB0", baud=115200):
    """
    Poll GNSS and cellular status in the background.
    Coordinates come from the EC200U's GNSS receiver; GSM/LTE reports
    registration and signal strength, not GPS coordinates.
    """
    try:
        import serial
    except Exception as exc:
        with gps_lock:
            gps_result["status"] = f"pyserial unavailable: {exc}"
        print("[GPS ERROR] Install pyserial in the dms_v2 virtual environment: pip install pyserial")
        return

    try:
        with serial.Serial(port_name, baudrate=baud, timeout=0.25, write_timeout=1) as port:
            time.sleep(0.3)
            at_response = gps_at_command(port, "AT")
            if "OK" not in at_response:
                raise RuntimeError(f"No AT response on {port_name}. Check the EC200U AT port.")

            gps_status = gps_at_command(port, "AT+QGPS?")
            if "+QGPS: 1" not in gps_status:
                start_response = gps_at_command(port, "AT+QGPS=1", timeout=3.0)
                if "OK" not in start_response and "already" not in start_response.lower():
                    print(f"[GPS] QGPS start response: {start_response or 'no response'}")
            print(f"[GPS] EC200U AT port ready: {port_name}")
            print("[GPS] Waiting for satellite fix; first fix can take several minutes outdoors.")

            while not stop_event.is_set():
                updates = {}

                loc_response = gps_at_command(port, "AT+QGPSLOC=0", timeout=2.0)
                loc = gps_parse_qgpsloc(loc_response)
                if loc:
                    updates.update(loc)
                    updates["status"] = "FIX"
                    updates["last_update"] = time.time()
                    print(
                        f"[GPS] {loc['latitude']:.6f}, {loc['longitude']:.6f} | "
                        f"{loc['speed_kmh']:.1f} km/h"
                    )
                elif "+CME ERROR" in loc_response or "ERROR" in loc_response:
                    updates["status"] = "Waiting for GNSS fix"
                else:
                    updates["status"] = "Waiting for GNSS fix"

                # Cellular network status and RSSI are separate from GNSS coordinates.
                creg = gps_at_command(port, "AT+CREG?", timeout=1.5)
                gsm_status = "Not registered"
                for line in creg.splitlines():
                    if "+CREG:" in line:
                        try:
                            values = line.split(":", 1)[1].replace(" ", "").split(",")
                            reg = int(values[-1])
                            gsm_status = "Registered" if reg in (1, 5) else "Not registered"
                        except (ValueError, IndexError):
                            pass
                updates["gsm_status"] = gsm_status

                csq = gps_at_command(port, "AT+CSQ", timeout=1.5)
                for line in csq.splitlines():
                    if "+CSQ:" in line:
                        try:
                            raw = int(line.split(":", 1)[1].split(",")[0].strip())
                            updates["signal_dbm"] = None if raw == 99 else -113 + 2 * raw
                        except (ValueError, IndexError):
                            pass

                with gps_lock:
                    # Preserve the last good coordinates when a later poll has no fix.
                    if loc:
                        gps_result.update(loc)
                        gps_result["last_update"] = updates.get("last_update")
                    gps_result["status"] = updates.get("status", gps_result["status"])
                    gps_result["gsm_status"] = updates.get("gsm_status", gps_result["gsm_status"])
                    if "signal_dbm" in updates:
                        gps_result["signal_dbm"] = updates["signal_dbm"]

                time.sleep(1.5)

    except Exception as exc:
        with gps_lock:
            gps_result["status"] = f"Unavailable: {exc}"
        print(f"[GPS/GSM ERROR] {exc}")
        print("[GPS/GSM] Check the EC200U AT port (default /dev/ttyUSB0) and install pyserial.")


def log_incident_event(reason, lane_departed=False):
    """Append one incident transition with the latest available GNSS and sensor data."""
    with gps_lock:
        location = dict(gps_result)
    with sensor_lock:
        sensors = dict(sensor_result)

    path = PROJECT_ROOT / "dms_incident_log.csv"
    new_file = not path.exists()
    row = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": reason,
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "speed_kmh": location.get("speed_kmh"),
        "altitude_m": location.get("altitude_m"),
        "gps_status": location.get("status"),
        "gsm_status": location.get("gsm_status"),
        "signal_dbm": location.get("signal_dbm"),
        "lane_departure": bool(lane_departed),
        "accel_x_g": None,
        "accel_y_g": None,
        "accel_z_g": None,
    }
    accel = sensors.get("accel")
    if accel is not None:
        row["accel_x_g"], row["accel_y_g"], row["accel_z_g"] = accel

    try:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[INCIDENT LOG] {reason} | GPS={row['latitude']},{row['longitude']} "
            f"| saved to {path.name}"
        )
    except Exception as exc:
        print(f"[INCIDENT LOG ERROR] {exc}")


def draw_gps_overlay(frame):
    """Display GNSS coordinates and cellular registration on the live/recorded image."""
    with gps_lock:
        values = dict(gps_result)

    h, w = frame.shape[:2]
    x = max(10, w - 410)
    y = 166
    panel_w = min(400, w - x - 10)
    panel_h = 108
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)

    lat = values.get("latitude")
    lon = values.get("longitude")
    if lat is not None and lon is not None:
        location_line = f"GPS: {lat:.6f}, {lon:.6f}"
    else:
        location_line = "GPS: waiting for satellite fix"

    speed = values.get("speed_kmh")
    speed_line = f"Speed: {speed:.1f} km/h" if speed is not None else "Speed: --"
    alt = values.get("altitude_m")
    alt_line = f"Altitude: {alt:.1f} m" if alt is not None else "Altitude: --"
    signal = values.get("signal_dbm")
    signal_line = f"GSM/LTE: {values.get('gsm_status', 'Unknown')}"
    if signal is not None:
        signal_line += f" | {signal} dBm"

    lines = [
        f"GNSS: {values.get('status', 'Starting')}",
        location_line,
        f"{speed_line} | {alt_line}",
        signal_line,
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x + 8, y + 22 + i * 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )


# ============================================================
# CPU AFFINITY
# ============================================================

def set_thread_affinity(cpus):
    """
    Restrict the current Linux thread to selected CPU cores.
    Main/video thread -> CPU 0
    MediaPipe -> CPU 1
    Serialized YOLO -> CPUs 2 and 3
    """

    try:

        tid = threading.get_native_id()

        os.sched_setaffinity(
            tid,
            set(cpus)
        )

        print(
            f"[CPU] thread {tid} -> "
            f"CPU {sorted(cpus)}"
        )

    except Exception as e:

        print(
            "[CPU] affinity not applied:",
            e
        )


# ============================================================
# FRAME HELPERS
# ============================================================

def publish_frame(
    frame,
    frame_id
):
    global latest_frame
    global latest_frame_id

    with frame_lock:

        latest_frame = frame

        latest_frame_id = frame_id


def get_latest_frame_copy():

    with frame_lock:

        if latest_frame is None:

            return None

        return latest_frame.copy()


# ============================================================
# RESULT STORAGE
# ============================================================

def store_result(
    name,
    result,
    elapsed
):

    global mp_result
    global dms_result
    global vehicle_result
    global accident_result

    global mp_time
    global dms_time
    global vehicle_time
    global accident_time

    global mp_fps
    global dms_fps
    global vehicle_fps
    global accident_fps

    global last_accident_detection_time

    fps = (
        1.0 / elapsed
        if elapsed > 0.0
        else 0.0
    )

    with result_lock:

        if name == "mp":

            mp_result = result

            mp_time = elapsed

            mp_fps = fps


        elif name == "dms":

            dms_result = result

            dms_time = elapsed

            dms_fps = fps


        elif name == "vehicle":

            vehicle_result = result

            vehicle_time = elapsed

            vehicle_fps = fps


        elif name == "accident":

            accident_result = result

            accident_time = elapsed

            accident_fps = fps

            for detection in result:

                class_name = str(
                    detection["class_name"]
                ).lower().strip()

                if (
                    class_name
                    in
                    ACCIDENT_CLASSES
                ):

                    last_accident_detection_time = (
                        time.perf_counter()
                    )


# ============================================================
# DMS ONNX DETECTOR
# ============================================================

class DMSDetector:

    def __init__(
        self,
        model_path,
        confidence
    ):

        self.model_path = Path(
            model_path
        )

        self.confidence = confidence

        if not self.model_path.exists():

            raise FileNotFoundError(
                "DMS model not found:\n"
                f"{self.model_path}"
            )

        options = ort.SessionOptions()

        # Keep ONNX from competing with NCNN.
        options.intra_op_num_threads = 1

        options.inter_op_num_threads = 1

        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = (
            ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        print(
            "[DMS] READY:",
            self.model_path
        )


    @staticmethod
    def preprocess(frame):

        image = cv2.resize(
            frame,
            (640, 640),
            interpolation=cv2.INTER_LINEAR
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = (
            image.astype(np.float32)
            /
            255.0
        )

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        return np.expand_dims(
            image,
            axis=0
        )


    def detect(
        self,
        frame
    ):

        h, w = frame.shape[:2]

        tensor = self.preprocess(
            frame
        )

        outputs = self.session.run(
            None,
            {
                self.input_name:
                tensor
            }
        )

        predictions = outputs[0][0]

        detections = []

        for prediction in predictions.T:

            scores = prediction[4:]

            class_id = int(
                np.argmax(scores)
            )

            confidence = float(
                scores[class_id]
            )

            if (
                confidence
                <
                self.confidence
            ):

                continue

            x_center = float(
                prediction[0]
            )

            y_center = float(
                prediction[1]
            )

            box_w = float(
                prediction[2]
            )

            box_h = float(
                prediction[3]
            )

            x1 = int(
                (
                    x_center
                    -
                    box_w / 2
                )
                *
                w
                /
                640
            )

            y1 = int(
                (
                    y_center
                    -
                    box_h / 2
                )
                *
                h
                /
                640
            )

            x2 = int(
                (
                    x_center
                    +
                    box_w / 2
                )
                *
                w
                /
                640
            )

            y2 = int(
                (
                    y_center
                    +
                    box_h / 2
                )
                *
                h
                /
                640
            )

            x1 = max(
                0,
                min(
                    x1,
                    w - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    h - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    w - 1
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    h - 1
                )
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                continue

            detections.append({

                "class_id":
                    class_id,

                "class_name":
                    DMS_CLASS_NAMES.get(
                        class_id,
                        str(class_id)
                    ),

                "confidence":
                    confidence,

                "bbox":
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
            })

        return detections


# ============================================================
# NCNN MODEL LOADING
# ============================================================

def load_ncnn_model(
    model_path,
    name
):

    print(
        f"[{name}] Loading NCNN:"
    )

    print(
        model_path
    )

    if not Path(
        model_path
    ).exists():

        raise FileNotFoundError(
            f"{name} NCNN model not found:\n"
            f"{model_path}"
        )

    model = YOLO(
        str(model_path)
    )

    # CPU NCNN is the known-working backend
    # on this Pi. No CUDA/Vulkan selection.
    device = "cpu"

    print(
        f"[{name}] READY "
        f"(NCNN CPU)"
    )

    return (
        model,
        device
    )


# ============================================================
# ULTRALYTICS RESULT PARSER
# ============================================================

def parse_result(
    result,
    model,
    max_det
):

    detections = []

    if result.boxes is None:

        return detections

    boxes = list(
        result.boxes
    )

    # Strongest detections first.
    boxes.sort(
        key=lambda box:
        float(box.conf[0]),
        reverse=True
    )

    boxes = boxes[
        :max_det
    ]

    for box in boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        if isinstance(
            model.names,
            dict
        ):

            class_name = (
                model.names.get(
                    class_id,
                    str(class_id)
                )
            )

        else:

            class_name = str(
                class_id
            )

        detections.append({

            "class_id":
                class_id,

            "class_name":
                class_name,

            "confidence":
                confidence,

            "bbox":
                (
                    x1,
                    y1,
                    x2,
                    y2
                ),
        })

    return detections


# ============================================================
# SERIALIZED YOLO SCHEDULER
# ============================================================

def ai_scheduler():

    # Heavy AI gets CPUs 2 and 3.
    set_thread_affinity(
        {2, 3}
    )

    print(
        "[AI] Loading DMS..."
    )

    try:

        dms_model = DMSDetector(
            DMS_MODEL,
            DMS_CONF
        )

    except Exception as e:

        print(
            "[DMS INIT ERROR]",
            e
        )

        dms_model = None


    print(
        "[AI] Loading Vehicle..."
    )

    try:

        vehicle_model, vehicle_device = (
            load_ncnn_model(
                VEHICLE_MODEL,
                "Vehicle"
            )
        )

    except Exception as e:

        print(
            "[Vehicle INIT ERROR]",
            e
        )

        vehicle_model = None

        vehicle_device = "cpu"


    print(
        "[AI] Loading Accident..."
    )

    try:

        accident_model, accident_device = (
            load_ncnn_model(
                ACCIDENT_MODEL,
                "Accident"
            )
        )

    except Exception as e:

        print(
            "[Accident INIT ERROR]",
            e
        )

        accident_model = None

        accident_device = "cpu"


    print(
        "[AI] Scheduler READY"
    )

    # Separate clocks for each model.
    last_run = {

        "dms":
            0.0,

        "vehicle":
            0.0,

        "accident":
            0.0,
    }


    while not stop_event.is_set():

        now = time.perf_counter()

        due = []

        if (
            dms_model is not None
            and
            now
            -
            last_run["dms"]
            >=
            DMS_INTERVAL
        ):

            due.append(
                "dms"
            )


        if (
            vehicle_model is not None
            and
            now
            -
            last_run["vehicle"]
            >=
            VEHICLE_INTERVAL
        ):

            due.append(
                "vehicle"
            )


        if (
            accident_model is not None
            and
            now
            -
            last_run["accident"]
            >=
            ACCIDENT_INTERVAL
        ):

            due.append(
                "accident"
            )


        if not due:

            time.sleep(
                0.001
            )

            continue


        # Most overdue model goes first.
        name = max(
            due,
            key=lambda item:
                now
                -
                last_run[item]
        )


        frame = (
            get_latest_frame_copy()
        )

        if frame is None:

            time.sleep(
                0.002
            )

            continue


        # ====================================================
        # DMS
        # ====================================================

        if name == "dms":

            last_run["dms"] = (
                time.perf_counter()
            )

            try:

                t0 = (
                    time.perf_counter()
                )

                detections = (
                    dms_model.detect(
                        frame
                    )
                )

                elapsed = (
                    time.perf_counter()
                    -
                    t0
                )

                store_result(
                    "dms",
                    detections,
                    elapsed
                )

            except Exception as e:

                print(
                    "[DMS ERROR]",
                    e
                )


        # ====================================================
        # VEHICLE
        # ====================================================

        elif name == "vehicle":

            last_run["vehicle"] = (
                time.perf_counter()
            )

            try:

                t0 = (
                    time.perf_counter()
                )

                results = (
                    vehicle_model.predict(
                        frame,
                        imgsz=YOLO_IMGSZ,
                        conf=VEHICLE_CONF,
                        max_det=VEHICLE_MAX_DET,
                        device=vehicle_device,
                        verbose=False,
                    )
                )

                detections = []

                for result in results:

                    detections.extend(
                        parse_result(
                            result,
                            vehicle_model,
                            VEHICLE_MAX_DET
                        )
                    )

                elapsed = (
                    time.perf_counter()
                    -
                    t0
                )

                store_result(
                    "vehicle",
                    detections,
                    elapsed
                )

            except Exception as e:

                print(
                    "[VEHICLE ERROR]",
                    e
                )


        # ====================================================
        # ACCIDENT
        # ====================================================

        else:

            last_run["accident"] = (
                time.perf_counter()
            )

            try:

                t0 = (
                    time.perf_counter()
                )

                results = (
                    accident_model.predict(
                        frame,
                        imgsz=YOLO_IMGSZ,
                        conf=ACCIDENT_CONF,
                        max_det=ACCIDENT_MAX_DET,
                        device=accident_device,
                        verbose=False,
                    )
                )

                detections = []

                for result in results:

                    detections.extend(
                        parse_result(
                            result,
                            accident_model,
                            ACCIDENT_MAX_DET
                        )
                    )

                elapsed = (
                    time.perf_counter()
                    -
                    t0
                )

                store_result(
                    "accident",
                    detections,
                    elapsed
                )

            except Exception as e:

                print(
                    "[ACCIDENT ERROR]",
                    e
                )


# ============================================================
# MEDIAPIPE WORKER
# ============================================================

def mediapipe_worker():

    set_thread_affinity(
        {1}
    )

    print(
        "[MediaPipe] Loading..."
    )

    try:

        detector = (
            EyeYawnDetector()
        )

    except Exception as e:

        print(
            "[MediaPipe INIT ERROR]",
            e
        )

        return

    print(
        "[MediaPipe] READY"
    )

    last_run = 0.0

    last_id = -1

    while not stop_event.is_set():

        with frame_lock:

            frame = latest_frame

            frame_id = latest_frame_id

        if (
            frame is None
            or
            frame_id == last_id
        ):

            time.sleep(
                0.002
            )

            continue

        now = (
            time.perf_counter()
        )

        if (
            now
            -
            last_run
            <
            MEDIAPIPE_INTERVAL
        ):

            time.sleep(
                0.001
            )

            continue

        frame = frame.copy()

        last_id = frame_id

        last_run = now

        try:

            t0 = (
                time.perf_counter()
            )

            result = (
                detector.process_frame(
                    frame
                )
            )

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            store_result(
                "mp",
                result,
                elapsed
            )

        except Exception as e:

            print(
                "[MediaPipe ERROR]",
                e
            )


# ============================================================
# FAST LANE DETECTION
# ============================================================

def detect_lanes_small(
    frame
):

    original_h, original_w = (
        frame.shape[:2]
    )

    # Lane processing is done at 960 px width
    # instead of full 1920 px width.
    lane_width = 960

    if original_w <= lane_width:

        small = frame

        scale = 1.0

    else:

        scale = (
            lane_width
            /
            float(original_w)
        )

        lane_height = int(
            original_h
            *
            scale
        )

        small = cv2.resize(
            frame,
            (
                lane_width,
                lane_height
            ),
            interpolation=cv2.INTER_AREA
        )

    h, w = small.shape[:2]

    roi_top = int(
        h * 0.48
    )

    roi = small[
        roi_top:h
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    mask = np.zeros_like(
        edges
    )

    polygon = np.array(
        [[
            (
                int(w * 0.05),
                roi.shape[0] - 1
            ),
            (
                int(w * 0.38),
                int(
                    roi.shape[0]
                    * 0.12
                )
            ),
            (
                int(w * 0.62),
                int(
                    roi.shape[0]
                    * 0.12
                )
            ),
            (
                int(w * 0.95),
                roi.shape[0] - 1
            ),
        ]],
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        polygon,
        255
    )

    edges = cv2.bitwise_and(
        edges,
        mask
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=35,
        maxLineGap=80
    )

    left = []

    right = []

    if lines is not None:

        for raw in lines:

            (
                x1,
                y1,
                x2,
                y2
            ) = map(
                int,
                raw.reshape(-1)
            )

            y1 += roi_top

            y2 += roi_top

            dx = x2 - x1

            dy = y2 - y1

            if dx == 0:
                continue

            slope = (
                dy
                /
                float(dx)
            )

            if (
                abs(slope)
                <
                0.35
                or
                abs(slope)
                >
                5.0
            ):

                continue

            mid_x = (
                x1 + x2
            ) / 2.0

            length = (
                dx * dx
                +
                dy * dy
            ) ** 0.5

            if (
                slope < 0
                and
                mid_x < w * 0.60
            ):

                left.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        length
                    )
                )

            elif (
                slope > 0
                and
                mid_x > w * 0.40
            ):

                right.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        length
                    )
                )

    return (
        left,
        right,
        scale,
        original_w,
        original_h
    )


# ============================================================
# LANE FIT
# ============================================================

def fit_lane_line(
    lines,
    top_y,
    bottom_y,
    width
):

    if not lines:

        return None

    xs = []

    ys = []

    for (
        x1,
        y1,
        x2,
        y2,
        _
    ) in lines:

        xs.extend(
            [
                x1,
                x2
            ]
        )

        ys.extend(
            [
                y1,
                y2
            ]
        )

    if len(xs) < 4:

        return None

    try:

        # x = m*y + b
        m, b = np.polyfit(
            np.asarray(ys),
            np.asarray(xs),
            1
        )

        x_bottom = (
            m * bottom_y
            + b
        )

        x_top = (
            m * top_y
            + b
        )

        if not (
            -0.2 * width
            <=
            x_bottom
            <=
            1.2 * width
        ):

            return None

        if not (
            -0.3 * width
            <=
            x_top
            <=
            1.3 * width
        ):

            return None

        return (
            int(x_bottom),
            int(bottom_y),
            int(x_top),
            int(top_y)
        )

    except Exception:

        return None


# ============================================================
# LANE SMOOTHING
# ============================================================

def smooth_line(
    previous,
    current,
    alpha=0.75
):

    if current is None:

        return previous

    if previous is None:

        return current

    return tuple(
        int(
            alpha * old
            +
            (1.0 - alpha)
            * new
        )
        for old, new
        in zip(
            previous,
            current
        )
    )


# ============================================================
# PROCESS LANES
# ============================================================

def process_lanes(
    frame
):

    global previous_left_line
    global previous_right_line

    (
        left_lines,
        right_lines,
        scale,
        original_w,
        original_h
    ) = detect_lanes_small(
        frame
    )

    scaled_w = int(
        original_w
        * scale
    )

    scaled_h = int(
        original_h
        * scale
    )

    top_y_small = int(
        scaled_h * 0.54
    )

    bottom_y_small = (
        scaled_h - 10
    )

    left_line = fit_lane_line(
        left_lines,
        top_y_small,
        bottom_y_small,
        scaled_w
    )

    right_line = fit_lane_line(
        right_lines,
        top_y_small,
        bottom_y_small,
        scaled_w
    )

    previous_left_line = (
        smooth_line(
            previous_left_line,
            left_line
        )
    )

    previous_right_line = (
        smooth_line(
            previous_right_line,
            right_line
        )
    )

    left_line = (
        previous_left_line
    )

    right_line = (
        previous_right_line
    )

    left_x = None

    right_x = None

    top_y = int(
        original_h * 0.54
    )

    bottom_y = (
        original_h - 10
    )

    if left_line is not None:

        x1 = int(
            left_line[0]
            /
            scale
        )

        y1 = int(
            left_line[1]
            /
            scale
        )

        x2 = int(
            left_line[2]
            /
            scale
        )

        y2 = int(
            left_line[3]
            /
            scale
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            4
        )

        left_x = x1

    if right_line is not None:

        x1 = int(
            right_line[0]
            /
            scale
        )

        y1 = int(
            right_line[1]
            /
            scale
        )

        x2 = int(
            right_line[2]
            /
            scale
        )

        y2 = int(
            right_line[3]
            /
            scale
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            4
        )

        right_x = x1

    lane_detected = (
        left_x is not None
        and
        right_x is not None
        and
        right_x > left_x
    )

    lane_center = None

    if lane_detected:

        lane_width = (
            right_x
            -
            left_x
        )

        if not (
            frame.shape[1] * 0.18
            <=
            lane_width
            <=
            frame.shape[1] * 0.95
        ):

            lane_detected = False

        else:

            lane_center = (
                left_x
                +
                right_x
            ) / 2.0

            cv2.line(
                frame,
                (
                    int(lane_center),
                    bottom_y
                ),
                (
                    int(lane_center),
                    top_y
                ),
                (255, 0, 255),
                3
            )

    return (
        lane_detected,
        lane_center
    )


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    a,
    b
):

    ax1, ay1, ax2, ay2 = a

    bx1, by1, bx2, by2 = b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    if (
        ix2 <= ix1
        or
        iy2 <= iy1
    ):

        return 0.0

    intersection = (
        ix2 - ix1
    ) * (
        iy2 - iy1
    )

    area_a = (
        ax2 - ax1
    ) * (
        ay2 - ay1
    )

    area_b = (
        bx2 - bx1
    ) * (
        by2 - by1
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection
        /
        union
    )


# ============================================================
# COLLISION FALLBACK
# ============================================================

def collision_fallback(
    detections
):

    global previous_vehicle_objects

    global collision_counter

    current = []

    for d in detections:

        if (
            d["class_id"]
            not in
            VEHICLE_CLASSES
        ):

            continue

        x1, y1, x2, y2 = (
            d["bbox"]
        )

        area = (
            max(
                1,
                x2 - x1
            )
            *
            max(
                1,
                y2 - y1
            )
        )

        current.append({

            "bbox":
                d["bbox"],

            "cx":
                (x1 + x2)
                / 2.0,

            "cy":
                (y1 + y2)
                / 2.0,

            "area":
                area,
        })

    possible = False

    # Rapid apparent size growth.
    for obj in current:

        closest = None

        best_distance = (
            float("inf")
        )

        for old in (
            previous_vehicle_objects
        ):

            distance = (
                (
                    obj["cx"]
                    -
                    old["cx"]
                ) ** 2
                +
                (
                    obj["cy"]
                    -
                    old["cy"]
                ) ** 2
            ) ** 0.5

            if (
                distance
                <
                best_distance
            ):

                best_distance = (
                    distance
                )

                closest = old

        if (
            closest is not None
            and
            closest["area"] > 0
        ):

            growth = (
                obj["area"]
                /
                closest["area"]
            )

            if growth > 1.60:

                possible = True

    # Limit pairwise comparisons.
    current_for_iou = (
        current[:10]
    )

    for i in range(
        len(current_for_iou)
    ):

        for j in range(
            i + 1,
            len(current_for_iou)
        ):

            if (
                calculate_iou(
                    current_for_iou[i]["bbox"],
                    current_for_iou[j]["bbox"]
                )
                >
                0.65
            ):

                possible = True

    if possible:

        collision_counter += 1

    else:

        collision_counter = max(
            0,
            collision_counter - 1
        )

    previous_vehicle_objects = (
        current
    )

    return (
        collision_counter
        >=
        COLLISION_CONFIRM
    )


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_detection(
    frame,
    detection,
    prefix,
    color
):

    x1, y1, x2, y2 = (
        detection["bbox"]
    )

    label = (
        f"{prefix}: "
        f"{detection['class_name']} "
        f"{detection['confidence']:.2f}"
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                20,
                y1 - 7
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# STATUS PANEL
# ============================================================

def draw_status(
    frame,
    frame_no,
    total_frames,
    video_fps,
    lane_detected,
    lane_departed,
    collision_alert
):

    with result_lock:

        mp = dict(
            mp_result
        )

        dms = list(
            dms_result
        )

        vehicle = list(
            vehicle_result
        )

        accident = list(
            accident_result
        )

        mf = mp_fps
        df = dms_fps
        vf = vehicle_fps
        af = accident_fps

        mt = mp_time
        dt = dms_time
        vt = vehicle_time
        at = accident_time

    accident_name = ""

    accident_confidence = 0.0

    for d in accident:

        name = str(
            d["class_name"]
        ).lower().strip()

        if (
            name
            in
            ACCIDENT_CLASSES
            and
            d["confidence"]
            >
            accident_confidence
        ):

            accident_name = name

            accident_confidence = (
                d["confidence"]
            )

    held_accident = (
        time.perf_counter()
        -
        last_accident_detection_time
        <
        ACCIDENT_HOLD_SECONDS
    )

    accident_active = (
        bool(accident_name)
        or
        held_accident
        or
        collision_alert
    )

    # --------------------------------------------------------
    # TOP STATUS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"VIDEO FPS: {video_fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    face_found = mp.get(
        "face_found",
        False
    )

    cv2.putText(
        frame,
        (
            "FACE: YES"
            if face_found
            else
            "FACE: NO"
        ),
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    if face_found:

        cv2.putText(
            frame,
            f"Eyes: {mp.get('eyes_closed', False)}",
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Yawn: {mp.get('yawning', False)}",
            (10, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # BOTTOM PANEL
    # --------------------------------------------------------

    h, w = frame.shape[:2]

    panel_h = 190

    panel_y = (
        h - panel_h
    )

    cv2.rectangle(
        frame,
        (5, panel_y),
        (540, h - 5),
        (0, 0, 0),
        -1
    )

    rows = [

        (
            f"DMS       : "
            f"{df:.1f} FPS | "
            f"{dt * 1000:.0f} ms"
        ),

        (
            f"Vehicle   : "
            f"{vf:.1f} FPS | "
            f"{vt * 1000:.0f} ms"
        ),

        (
            f"Accident  : "
            f"{af:.1f} FPS | "
            f"{at * 1000:.0f} ms"
        ),

        (
            f"MediaPipe : "
            f"{mf:.1f} FPS | "
            f"{mt * 1000:.1f} ms"
        ),

        (
            f"Vehicles  : "
            f"{len(vehicle)}"
        ),
    ]

    y = (
        panel_y
        +
        20
    )

    for line in rows:

        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 21

    cv2.putText(
        frame,
        (
            "LANES: DETECTED"
            if lane_detected
            else
            "LANES: NOT DETECTED"
        ),
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (
            (0, 255, 0)
            if lane_detected
            else
            (0, 0, 255)
        ),
        2,
        cv2.LINE_AA
    )

    y += 23

    cv2.putText(
        frame,
        (
            "!!! LANE DEPARTURE !!!"
            if lane_departed
            else
            "LANE POSITION: OK"
        ),
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (
            (0, 0, 255)
            if lane_departed
            else
            (0, 255, 0)
        ),
        (
            2
            if lane_departed
            else
            1
        ),
        cv2.LINE_AA
    )

    y += 23

    if accident_active:

        if accident_name:

            text = (
                f"!!! ACCIDENT: "
                f"{accident_name} "
                f"{accident_confidence:.2f} !!!"
            )

        elif collision_alert:

            text = (
                "!!! POSSIBLE COLLISION !!!"
            )

        else:

            text = (
                "!!! ACCIDENT DETECTED !!!"
            )

        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            frame,
            "ACCIDENT: NO",
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    frame_label = (
        f"Frame: {frame_no} | LIVE"
        if total_frames <= 0
        else f"Frame: {frame_no}/{total_frames}"
    )
    cv2.putText(
        frame,
        frame_label,
        (max(10, w - 220), 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# MAIN - LIVE RASPBERRY PI CAMERA
# ============================================================

def main():
    global lane_departure_counter

    parser = argparse.ArgumentParser(
        description="Live DMS test using the Raspberry Pi CSI camera"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save the annotated live camera feed to an MP4 file",
    )
    parser.add_argument(
        "-o", "--output",
        default=str(PROJECT_ROOT / "dms_live_test_output.mp4"),
        help="Output path used with --record",
    )
    parser.add_argument(
        "--no-incident-clips",
        action="store_true",
        help="Disable automatic incident-only video clips (enabled by default)",
    )
    parser.add_argument(
        "--incident-dir",
        default=str(PROJECT_ROOT / "incident_clips"),
        help="Folder for accident/collision clips",
    )
    parser.add_argument(
        "--incident-fallback-dir",
        default=str(PROJECT_ROOT / "incident_clips_local"),
        help="Fallback folder if the primary incident folder is unavailable",
    )
    parser.add_argument(
        "--incident-pre", type=float, default=15.0,
        help="Seconds to save before accident detection (default: 15)",
    )
    parser.add_argument(
        "--incident-post", type=float, default=15.0,
        help="Seconds to save after the last active accident trigger (default: 15)",
    )
    parser.add_argument(
        "--incident-max-clip", type=float, default=120.0,
        help="Maximum incident clip duration including pre-roll (default: 120)",
    )
    parser.add_argument(
        "--incident-min-free-gb", type=float, default=1.0,
        help="Minimum free GB required to start a clip (default: 1)",
    )
    parser.add_argument(
        "--incident-queue-size", type=int, default=60,
        help="Maximum live frames queued for incident writing (default: 60)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Camera preview width (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Camera preview height (default: 720)",
    )
    parser.add_argument(
        "--gps-port", default="/dev/ttyUSB0",
        help="Quectel EC200U AT-command serial port (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--gps-baud", type=int, default=115200,
        help="Quectel EC200U serial baud rate (default: 115200)",
    )
    args = parser.parse_args()

    # Import Picamera2 only for the live-camera version.
    try:
        # pyrefly: ignore [missing-import]
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError(
            "Picamera2 is unavailable in this Python environment. "
            "Use the dms_v2 .venv that can import Picamera2."
        ) from exc

    # Verify model paths before opening the camera.
    required = [
        (DMS_MODEL, "DMS ONNX model"),
        (VEHICLE_MODEL, "Vehicle/person NCNN model"),
        (ACCIDENT_MODEL, "Accident NCNN model"),
    ]
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found:\n{path}")

    print("\n" + "=" * 68)
    print("DMS LIVE PI CAMERA TEST")
    print("=" * 68)
    print(f"Project       : {PROJECT_ROOT}")
    print(f"Camera size   : {args.width}x{args.height}")
    print(f"Vehicle/accident YOLO input: {YOLO_IMGSZ}x{YOLO_IMGSZ} (DMS ONNX stays 640x640)")
    print(f"DMS model     : {DMS_MODEL}")
    print(f"Vehicle model : {VEHICLE_MODEL}")
    print(f"Accident model: {ACCIDENT_MODEL}")
    print(f"Recording     : {'ON -> ' + args.output if args.record else 'OFF'}")
    print(
        f"Incident clips: {'OFF' if args.no_incident_clips else 'ON -> ' + args.incident_dir}"
    )
    if not args.no_incident_clips:
        print(
            f"Incident timing: {args.incident_pre:.1f}s before + "
            f"{args.incident_post:.1f}s after last active accident/collision trigger"
        )
    print(f"EC200U AT port: {args.gps_port} @ {args.gps_baud}")
    print("Press Q in the preview window to stop.\n")

    camera = Picamera2()
    writer = None
    incident_recorder = None
    sensor_thread = None
    gps_thread = None
    camera_started = False

    try:
        if not args.no_incident_clips:
            incident_recorder = IncidentRecorder(
                output_dir=args.incident_dir,
                fallback_dir=args.incident_fallback_dir,
                pre_seconds=args.incident_pre,
                post_seconds=args.incident_post,
                max_clip_seconds=args.incident_max_clip,
                fps_hint=TARGET_OUTPUT_FPS,
                fourcc="mp4v",
                min_free_gb=args.incident_min_free_gb,
                queue_size=args.incident_queue_size,
            )

        config = camera.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            buffer_count=4,
        )
        camera.configure(config)
        camera.start()
        camera_started = True
        time.sleep(1.5)  # allow auto-exposure/white balance to settle

        # The model/detector code expects OpenCV BGR frames.
        print("[CAMERA] Picamera2 started.")
        print("[SYSTEM] Starting AI and sensor workers...")

        stop_event.clear()
        lane_departure_counter = 0

        ai_thread = threading.Thread(target=ai_scheduler, daemon=True)
        mp_thread = threading.Thread(target=mediapipe_worker, daemon=True)
        sensor_thread = threading.Thread(target=sensor_worker, daemon=True)
        gps_thread = threading.Thread(
            target=gps_worker,
            kwargs={"port_name": args.gps_port, "baud": args.gps_baud},
            daemon=True,
        )
        ai_thread.start()
        mp_thread.start()
        sensor_thread.start()
        gps_thread.start()

        frame_no = 0
        fps_start = time.perf_counter()
        fps_frames = 0
        measured_fps = 0.0
        previous_alert_active = False

        while not stop_event.is_set():
            rgb_frame = camera.capture_array("main")
            if rgb_frame is None or rgb_frame.size == 0:
                print("[CAMERA] Empty frame received; retrying.")
                time.sleep(0.05)
                continue

            # Picamera2's RGB888 capture_array data is already in BGR byte order
            # for OpenCV on Raspberry Pi. Rotate the frame 90 degrees clockwise
            # BEFORE publishing it, so all AI models, overlays, preview, and
            # incident recordings use the same orientation.
            frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_no += 1
            height, width = frame.shape[:2]

            # Publish the unannotated frame to the asynchronous AI workers.
            publish_frame(frame.copy(), frame_no)

            lane_detected, lane_center = process_lanes(frame)
            lane_departed = False
            if lane_detected:
                frame_center = width / 2.0
                lane_error = abs(lane_center - frame_center)
                threshold = width * 0.45 * 0.20
                if lane_error > threshold:
                    lane_departure_counter += 1
                else:
                    lane_departure_counter = max(0, lane_departure_counter - 1)
                lane_departed = lane_departure_counter >= 5
            else:
                lane_departure_counter = 0

            with result_lock:
                current_dms = list(dms_result)
                current_vehicle = list(vehicle_result)
                current_accident = list(accident_result)

            for detection in current_vehicle:
                class_id = detection["class_id"]
                if class_id in PERSON_CLASSES:
                    draw_detection(frame, detection, "PERSON", (255, 0, 0))
                elif class_id in VEHICLE_CLASSES:
                    draw_detection(frame, detection, "VEH", (0, 255, 0))
                else:
                    draw_detection(frame, detection, "VEH", (0, 255, 255))

            for detection in current_dms:
                draw_detection(frame, detection, "DMS", (255, 0, 255))

            for detection in current_accident:
                name = str(detection["class_name"]).lower().strip()
                color = (0, 0, 255) if name in ACCIDENT_CLASSES else (0, 165, 255)
                draw_detection(frame, detection, "ACC", color)

            collision_alert = collision_fallback(current_vehicle)
            accident_model_alert = any(
                str(d.get("class_name", "")).lower().strip() in ACCIDENT_CLASSES
                for d in current_accident
            )
            # Preserve the existing combined alert/logging behaviour, including lane departure.
            alert_active = bool(collision_alert or accident_model_alert or lane_departed)

            # Incident video is triggered by accident-model/collision signals,
            # not by a lane-departure warning alone. The short hold matches the
            # accident status panel and helps avoid flickering inference results.
            held_accident = (
                last_accident_detection_time > 0.0
                and time.perf_counter() - last_accident_detection_time
                < ACCIDENT_HOLD_SECONDS
            )
            incident_alert_active = bool(
                accident_model_alert or collision_alert or held_accident
            )

            if alert_active and not previous_alert_active:
                reasons = []
                if accident_model_alert:
                    reasons.append("ACCIDENT_MODEL")
                if collision_alert:
                    reasons.append("POSSIBLE_COLLISION")
                if lane_departed:
                    reasons.append("LANE_DEPARTURE")
                log_incident_event("+".join(reasons), lane_departed=lane_departed)
            previous_alert_active = alert_active

            fps_frames += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                measured_fps = fps_frames / elapsed
                fps_start = time.perf_counter()
                fps_frames = 0

            # total_frames=0 tells draw_status this is an unbounded live stream.
            draw_status(
                frame, frame_no, 0, measured_fps,
                lane_detected, lane_departed, collision_alert
            )
            draw_sensor_overlay(frame)
            draw_gps_overlay(frame)

            # Incident-only recorder: add EVERY annotated frame so the rolling
            # buffer always holds the last --incident-pre seconds. When an
            # accident/collision signal is active, extend recording through the
            # post-roll window. Saved frames include boxes, status, sensors and GPS.
            if incident_recorder is not None:
                incident_recorder.add_frame(frame)
                if incident_alert_active:
                    incident_recorder.on_accident_detected()
                else:
                    incident_recorder.on_accident_cleared()

            # Existing optional continuous recording is retained unchanged.
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(
                        args.output,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        25.0,
                        (width, height),
                    )
                    if not writer.isOpened():
                        writer.release()
                        writer = None
                        raise RuntimeError(f"Could not create output video: {args.output}")
                writer.write(frame)

            cv2.imshow("DMS - LIVE PI CAMERA", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("[SYSTEM] Q pressed; stopping live test.")
                break

    except KeyboardInterrupt:
        print("\n[SYSTEM] Ctrl+C received; stopping live test.")
    finally:
        stop_event.set()
        if sensor_thread is not None:
            sensor_thread.join(timeout=2.0)
        if gps_thread is not None:
            gps_thread.join(timeout=2.0)
        if incident_recorder is not None:
            try:
                incident_recorder.shutdown()
            except Exception as exc:
                print(f"[IncidentRecorder] Shutdown error: {exc}")
        if writer is not None:
            writer.release()
        if camera_started:
            camera.stop()
        camera.close()
        cv2.destroyAllWindows()

    print("\n" + "=" * 68)
    print("LIVE CAMERA TEST STOPPED")
    if args.record:
        print(f"Recording path: {args.output}")
    print("=" * 68)


if __name__ == "__main__":
    main()

