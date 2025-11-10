#!/usr/bin/env python3
"""
DoorKam Security Camera System - Python 3.14 Free-Threaded Edition
Optimized for true parallelism with GIL disabled.

Requirements:
    - Python 3.14+ (free-threaded build recommended)
    - Run with: PYTHON_GIL=0 python3 branch12.py
    - For best performance: use python3.14t if available
"""

from __future__ import annotations  # PEP 649: Deferred annotation evaluation
import sys
import os
import logging

# Setup logging early for proper error reporting
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Python 3.14 version check
if sys.version_info < (3, 14):
    logging.warning(f"[WARN]  Python 3.14+ recommended (using {sys.version_info.major}.{sys.version_info.minor})")
    logging.warning("   Some features may not be available or may have reduced performance")
    logging.warning("   Install Python 3.14 for optimal performance")

# Standard library imports
import cv2
import time
import smtplib
import numpy as np
import tempfile
import threading
import subprocess
import signal
import json
import socket
import atexit
import tkinter as tk
from tkinter import ttk, messagebox
from multiprocessing import Process, Queue, Value
from flask import Flask, Response, render_template_string, request, redirect, url_for, session, flash, send_file
from flask_limiter import Limiter
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email
import bcrypt
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from dotenv import load_dotenv
import shutil
import imageio
import random
import multiprocessing
import queue
import re
from ultralytics import YOLO  # For YOLOv11 with segmentation

# Python 3.14+ specific imports for modern features
from dataclasses import dataclass, field
from typing import (
    Literal, Optional, Tuple, Dict, List, Any, Callable,
    Protocol, TypeAlias, TypedDict
)
from collections import deque
from functools import wraps
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto

# Python 3.14 specific: Check for subinterpreters (PEP 734)
SUBINTERPRETERS_AVAILABLE = False
try:
    from concurrent.interpreters import create, run_in_thread, InterpreterPoolExecutor
    SUBINTERPRETERS_AVAILABLE = True
except ImportError:
    pass  # Not available in this build

# Check if running with GIL disabled (Python 3.14 free-threading)
GIL_DISABLED = False
if hasattr(sys, '_is_gil_enabled'):
    GIL_DISABLED = not sys._is_gil_enabled()
elif os.getenv('PYTHON_GIL') == '0':
    GIL_DISABLED = True

# Log Python 3.14 capabilities
if sys.version_info >= (3, 14):
    if GIL_DISABLED:
        logging.info("deg Running in FREE-THREADED mode (no GIL) - maximum performance enabled")
    else:
        logging.warning("[WARN]  Running with GIL enabled - consider PYTHON_GIL=0 for better performance")

    if SUBINTERPRETERS_AVAILABLE:
        logging.info("[OK] Subinterpreters available (PEP 734)")
    else:
        logging.info("[INFO] Subinterpreters not available - using standard threading")

# Load environment variables
load_dotenv()

# Constants
EMAIL_COOLDOWN = 300  # Seconds
MIN_CONTOUR_AREA = 100  # Pixels (reduced from 500 for low-contrast detection)
MIN_RECORDING_DURATION = 10  # Seconds
ROTATION_COOLDOWN = 0.5  # Seconds
FRAME_QUEUE_SIZE = 30  # Frames - increased from 10 to reduce dropped frames under load
YOLO_ROI_TIMEOUT = 5.0  # Seconds - timeout for ROI YOLO analysis
YOLO_FULL_FRAME_TIMEOUT = 10.0  # Seconds - timeout for full frame YOLO analysis

# ============================================================================
# PYTHON 3.14: Type-Safe Configuration Classes
# ============================================================================

# Type aliases for better readability
CameraName: TypeAlias = Literal['cam1', 'cam2']
DetectionMode: TypeAlias = Literal['cam1', 'cam2', 'both', 'disable']
Rotation: TypeAlias = Literal[0, 90, 180, 270]

@dataclass
class CameraConfig:
    """Configuration for a single camera"""
    resolution: str  # Format: "WIDTHxHEIGHT"
    area_multiplier: float = 1.0
    threshold_multiplier: float = 1.0
    min_contours: int = 0

    def get_resolution_tuple(self) -> Tuple[int, int]:
        """Parse resolution string safely"""
        try:
            w, h = map(int, self.resolution.split('x'))
            if not (100 < w < 4000 and 100 < h < 3000):
                raise ValueError(f"Invalid resolution: {w}x{h}")
            return (w, h)
        except (ValueError, AttributeError) as e:
            logging.error(f"Invalid resolution '{self.resolution}': {e}")
            return (640, 480)  # Safe default

@dataclass
class ROICoordinates:
    """Region of Interest coordinates"""
    x: int
    y: int
    width: int
    height: int

    def is_valid(self) -> bool:
        """Check if ROI is valid"""
        return all([
            self.x >= 0,
            self.y >= 0,
            self.width > 0,
            self.height > 0
        ])

@dataclass
class YOLOConfig:
    """YOLO detection configuration"""
    enabled: bool = True
    model_path: str = "yolov8m.pt"
    confidence_threshold: float = 0.5
    classes_of_interest: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])

    def validate(self) -> List[str]:
        """Validate YOLO configuration"""
        errors = []
        if not 0.1 <= self.confidence_threshold <= 1.0:
            errors.append(f"Invalid confidence: {self.confidence_threshold}")
        if not os.path.exists(self.model_path):
            errors.append(f"Model not found: {self.model_path}")
        return errors

@dataclass
class AppConfig:
    """Complete application configuration with type safety"""
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    detection_camera: DetectionMode = 'cam1'
    cam1_config: CameraConfig = field(default_factory=lambda: CameraConfig("640x480"))
    cam2_config: CameraConfig = field(default_factory=lambda: CameraConfig("1280x720"))
    cam1_roi: ROICoordinates = field(default_factory=lambda: ROICoordinates(0, 0, 640, 480))
    cam2_roi: ROICoordinates = field(default_factory=lambda: ROICoordinates(0, 0, 1280, 720))
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    sender_email: str = ""
    receiver_emails: List[str] = field(default_factory=list)
    email_password: str = ""
    timer_enabled: bool = False
    schedule_arm_time: str = "08:00"
    schedule_disarm_time: str = "18:00"
    background_threshold: int = 25

    def validate(self) -> List[str]:
        """Comprehensive configuration validation with detailed error messages"""
        errors = []

        # Port validation
        if not (1 <= self.flask_port <= 65535):
            errors.append(f"Invalid Flask port: {self.flask_port} (must be 1-65535)")

        # Detection camera mode validation
        valid_modes = ['cam1', 'cam2', 'both', 'disable']
        if self.detection_camera not in valid_modes:
            errors.append(f"Invalid detection_camera: {self.detection_camera} (must be one of {valid_modes})")

        # Camera resolution validation
        def validate_resolution(res_str: str, camera_name: str) -> bool:
            try:
                parts = res_str.split('x')
                if len(parts) != 2:
                    errors.append(f"{camera_name} resolution format invalid: '{res_str}' (must be WIDTHxHEIGHT)")
                    return False
                w, h = int(parts[0]), int(parts[1])
                if w <= 0 or h <= 0:
                    errors.append(f"{camera_name} resolution has invalid dimensions: {w}x{h}")
                    return False
                if w > 4096 or h > 4096:
                    errors.append(f"{camera_name} resolution exceeds maximum: {w}x{h} (max 4096x4096)")
                    return False
                return True
            except (ValueError, AttributeError) as e:
                errors.append(f"{camera_name} resolution parsing failed: {res_str} ({e})")
                return False

        validate_resolution(self.cam1_config.resolution, "Cam1")
        validate_resolution(self.cam2_config.resolution, "Cam2")

        # Camera config multiplier validation
        if self.cam1_config.area_multiplier <= 0:
            errors.append(f"Cam1 area_multiplier must be > 0 (got {self.cam1_config.area_multiplier})")
        if self.cam2_config.area_multiplier <= 0:
            errors.append(f"Cam2 area_multiplier must be > 0 (got {self.cam2_config.area_multiplier})")
        if self.cam1_config.threshold_multiplier <= 0:
            errors.append(f"Cam1 threshold_multiplier must be > 0 (got {self.cam1_config.threshold_multiplier})")
        if self.cam2_config.threshold_multiplier <= 0:
            errors.append(f"Cam2 threshold_multiplier must be > 0 (got {self.cam2_config.threshold_multiplier})")

        # ROI validation
        if not self.cam1_roi.is_valid():
            errors.append(f"Cam1 ROI invalid: x={self.cam1_roi.x}, y={self.cam1_roi.y}, w={self.cam1_roi.width}, h={self.cam1_roi.height}")
        if not self.cam2_roi.is_valid():
            errors.append(f"Cam2 ROI invalid: x={self.cam2_roi.x}, y={self.cam2_roi.y}, w={self.cam2_roi.width}, h={self.cam2_roi.height}")

        # YOLO validation
        errors.extend(self.yolo.validate())

        # Email validation
        if self.sender_email:
            if '@' not in self.sender_email or '.' not in self.sender_email:
                errors.append(f"Invalid sender email format: {self.sender_email}")
            if not self.email_password:
                errors.append("sender_email is set but email_password is empty")

        if self.receiver_emails:
            for idx, email in enumerate(self.receiver_emails):
                if not isinstance(email, str):
                    errors.append(f"Receiver email {idx} is not a string: {type(email)}")
                elif '@' not in email or '.' not in email:
                    errors.append(f"Invalid receiver email format: {email}")
            if not self.sender_email:
                errors.append("receiver_emails is set but sender_email is empty")

        # Timer schedule validation
        if self.timer_enabled:
            def validate_time_format(time_str: str, field_name: str) -> bool:
                try:
                    parts = time_str.split(':')
                    if len(parts) != 2:
                        errors.append(f"{field_name} must be in HH:MM format (got '{time_str}')")
                        return False
                    h, m = int(parts[0]), int(parts[1])
                    if not (0 <= h <= 23):
                        errors.append(f"{field_name} hour must be 0-23 (got {h})")
                        return False
                    if not (0 <= m <= 59):
                        errors.append(f"{field_name} minute must be 0-59 (got {m})")
                        return False
                    return True
                except (ValueError, AttributeError) as e:
                    errors.append(f"{field_name} parsing failed: '{time_str}' ({e})")
                    return False

            validate_time_format(self.schedule_arm_time, "schedule_arm_time")
            validate_time_format(self.schedule_disarm_time, "schedule_disarm_time")

        # Background threshold validation
        if not (1 <= self.background_threshold <= 255):
            errors.append(f"background_threshold must be 1-255 (got {self.background_threshold})")

        return errors

    @classmethod
    def load_from_json(cls, config_file: str = "config.json") -> AppConfig:
        """Load configuration from JSON file with type safety"""
        try:
            with open(config_file, 'r') as f:
                raw_config = json.load(f)

            # Parse camera configs
            cam1_cfg = CameraConfig(
                resolution=raw_config.get('cam1_resolution', '640x480'),
                area_multiplier=raw_config.get('cam1_area_multiplier', 1.0),
                threshold_multiplier=raw_config.get('cam1_threshold_multiplier', 1.0),
                min_contours=raw_config.get('cam1_min_contours', 0)
            )

            cam2_cfg = CameraConfig(
                resolution=raw_config.get('cam2_resolution', '1280x720'),
                area_multiplier=raw_config.get('cam2_area_multiplier', 4.0),
                threshold_multiplier=raw_config.get('cam2_threshold_multiplier', 1.5),
                min_contours=raw_config.get('cam2_min_contours', 0)
            )

            # Parse ROIs
            cam1_roi_data = raw_config.get('roi_coordinates', {})
            if isinstance(cam1_roi_data, list):
                cam1_roi = ROICoordinates(*cam1_roi_data[:4])
            else:
                cam1_roi = ROICoordinates(
                    cam1_roi_data.get('x', 0),
                    cam1_roi_data.get('y', 0),
                    cam1_roi_data.get('width', 640),
                    cam1_roi_data.get('height', 480)
                )

            cam2_roi_data = raw_config.get('roi_coordinates_cam2', {})
            if isinstance(cam2_roi_data, list):
                cam2_roi = ROICoordinates(*cam2_roi_data[:4])
            else:
                cam2_roi = ROICoordinates(
                    cam2_roi_data.get('x', 0),
                    cam2_roi_data.get('y', 0),
                    cam2_roi_data.get('width', 1280),
                    cam2_roi_data.get('height', 720)
                )

            # Parse YOLO config
            yolo_cfg = YOLOConfig(
                enabled=raw_config.get('use_yolo_detection', True),
                model_path=raw_config.get('yolo_model_path', 'yolov8m-seg.pt'),
                confidence_threshold=raw_config.get('yolo_confidence_threshold', 0.5),
                classes_of_interest=raw_config.get('yolo_classes_of_interest', [0, 1, 2, 3, 5, 7])
            )

            # Create application config
            app_config = cls(
                flask_host=raw_config.get('flask_host', '0.0.0.0'),
                flask_port=raw_config.get('flask_port', 5000),
                detection_camera=raw_config.get('detection_camera', 'cam1'),
                cam1_config=cam1_cfg,
                cam2_config=cam2_cfg,
                cam1_roi=cam1_roi,
                cam2_roi=cam2_roi,
                yolo=yolo_cfg,
                sender_email=raw_config.get('sender_email', ''),
                receiver_emails=raw_config.get('receiver_emails', []),
                email_password=raw_config.get('email_password', ''),
                timer_enabled=raw_config.get('timer_enabled', False),
                schedule_arm_time=raw_config.get('schedule_arm_time', '08:00'),
                schedule_disarm_time=raw_config.get('schedule_disarm_time', '18:00'),
                background_threshold=raw_config.get('background_threshold', 25)
            )

            # Validate with comprehensive error reporting
            errors = app_config.validate()
            if errors:
                logging.error("=" * 80)
                logging.error("CONFIGURATION VALIDATION ERRORS DETECTED:")
                logging.error("=" * 80)
                for i, error in enumerate(errors, 1):
                    logging.error(f"  {i}. {error}")
                logging.error("=" * 80)
                logging.error("System will attempt to continue with invalid config, but expect issues!")
                logging.error("Please fix config.json and restart for proper operation.")
                logging.error("=" * 80)

            return app_config

        except FileNotFoundError:
            logging.error(f"Config file not found: {config_file}")
            return cls()  # Return default config
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            return cls()


# ============================================================================
# NEW ARCHITECTURE: State Machine and Manager Classes  
# ============================================================================

class DetectionState(Enum):
    """State machine for detection flow"""
    IDLE = auto()
    MOTION_DETECTED = auto()
    MOTION_CONFIRMED = auto()
    YOLO_ROI = auto()
    YOLO_FULL = auto()
    RECORDING = auto()
    COOLDOWN = auto()

@dataclass
class DetectionStateManager:
    """Manages detection state transitions with proper timing"""
    state: DetectionState = DetectionState.IDLE
    motion_start_time: Optional[float] = None
    motion_confirmed_time: Optional[float] = None
    roi_yolo_start_time: Optional[float] = None
    full_yolo_start_time: Optional[float] = None
    recording_start_time: Optional[float] = None
    cooldown_start_time: Optional[float] = None
    
    motion_confirmation_duration: float = 1.0
    roi_yolo_timeout: float = 5.0
    full_yolo_timeout: float = 10.0
    cooldown_duration: float = 300.0
    
    roi_detected_person: bool = False
    full_frame_detected_person: bool = False
    source_camera: Optional[str] = None
    
    def transition_to(self, new_state: DetectionState) -> bool:
        """Safely transition to new state"""
        current_time = time.time()
        
        valid_transitions = {
            DetectionState.IDLE: [DetectionState.MOTION_DETECTED],
            DetectionState.MOTION_DETECTED: [DetectionState.MOTION_CONFIRMED, DetectionState.IDLE],
            DetectionState.MOTION_CONFIRMED: [DetectionState.YOLO_ROI, DetectionState.IDLE],
            DetectionState.YOLO_ROI: [DetectionState.YOLO_FULL, DetectionState.IDLE],
            DetectionState.YOLO_FULL: [DetectionState.RECORDING, DetectionState.IDLE],
            DetectionState.RECORDING: [DetectionState.COOLDOWN],
            DetectionState.COOLDOWN: [DetectionState.IDLE]
        }
        
        if new_state not in valid_transitions.get(self.state, []):
            logging.warning(f"Invalid state transition: {self.state} -> {new_state}")
            return False
        
        logging.info(f"State: {self.state.name} -> {new_state.name}")
        
        if new_state == DetectionState.MOTION_DETECTED:
            self.motion_start_time = current_time
        elif new_state == DetectionState.MOTION_CONFIRMED:
            self.motion_confirmed_time = current_time
        elif new_state == DetectionState.YOLO_ROI:
            self.roi_yolo_start_time = current_time
            self.roi_detected_person = False
        elif new_state == DetectionState.YOLO_FULL:
            self.full_yolo_start_time = current_time
            self.full_frame_detected_person = False
        elif new_state == DetectionState.RECORDING:
            self.recording_start_time = current_time
        elif new_state == DetectionState.COOLDOWN:
            self.cooldown_start_time = current_time
        elif new_state == DetectionState.IDLE:
            self.reset()
        
        self.state = new_state
        return True
    
    def reset(self):
        """Reset state variables"""
        global yolo_annotated_pi_frame, yolo_annotated_usb_frame

        self.motion_start_time = None
        self.motion_confirmed_time = None
        self.roi_yolo_start_time = None
        self.full_yolo_start_time = None
        self.recording_start_time = None
        self.roi_detected_person = False
        self.full_frame_detected_person = False
        self.source_camera = None

        # BUGFIX: Clear YOLO annotated frames to prevent reuse across detection events
        yolo_annotated_pi_frame = None
        yolo_annotated_usb_frame = None
    
    def check_timeout(self, current_time: float) -> bool:
        """Check timeouts, return True if timeout occurred"""
        if self.state == DetectionState.MOTION_DETECTED:
            if self.motion_start_time and (current_time - self.motion_start_time) >= 3.0:
                self.transition_to(DetectionState.IDLE)
                return True
        elif self.state == DetectionState.YOLO_ROI:
            if self.roi_yolo_start_time and (current_time - self.roi_yolo_start_time) >= self.roi_yolo_timeout:
                logging.warning(f"ROI YOLO timeout ({self.roi_yolo_timeout}s)")
                self.transition_to(DetectionState.IDLE)
                return True
        elif self.state == DetectionState.YOLO_FULL:
            if self.full_yolo_start_time and (current_time - self.full_yolo_start_time) >= self.full_yolo_timeout:
                logging.warning(f"Full frame YOLO timeout ({self.full_yolo_timeout}s)")
                self.transition_to(DetectionState.IDLE)
                return True
        elif self.state == DetectionState.COOLDOWN:
            if self.cooldown_start_time and (current_time - self.cooldown_start_time) >= self.cooldown_duration:
                self.transition_to(DetectionState.IDLE)
                return True
        return False

# ============================================================================

@dataclass
class RotationManager:
    """Unifies all 4 rotation systems into single source of truth"""
    cam1_display: int = 0
    cam2_display: int = 0
    cam1_email_override: Optional[int] = None
    _cam1_mp_value: Optional[any] = None
    _cam2_mp_value: Optional[any] = None
    _email_mp_value: Optional[any] = None
    web_sessions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    last_change_time: float = 0.0
    rotation_cooldown: float = 0.5
    
    def __post_init__(self):
        import multiprocessing
        if self._cam1_mp_value is None:
            self._cam1_mp_value = multiprocessing.Value('i', self.cam1_display)
        if self._cam2_mp_value is None:
            self._cam2_mp_value = multiprocessing.Value('i', self.cam2_display)
        if self._email_mp_value is None:
            self._email_mp_value = multiprocessing.Value('i', self.cam1_email_override or self.cam1_display)
    
    def set_cam1_display(self, rotation: int) -> bool:
        if not self._validate_rotation(rotation) or not self._check_cooldown():
            return False
        self.cam1_display = rotation
        with self._cam1_mp_value.get_lock():
            self._cam1_mp_value.value = rotation
        if self.cam1_email_override is None:
            with self._email_mp_value.get_lock():
                self._email_mp_value.value = rotation
        self.last_change_time = time.time()
        logging.info(f"Camera 1 rotation set to {rotation}deg")
        return True
    
    def set_cam2_display(self, rotation: int) -> bool:
        if not self._validate_rotation(rotation) or not self._check_cooldown():
            return False
        self.cam2_display = rotation
        with self._cam2_mp_value.get_lock():
            self._cam2_mp_value.value = rotation
        self.last_change_time = time.time()
        logging.info(f"Camera 2 rotation set to {rotation}deg")
        return True
    
    def _validate_rotation(self, rotation: int) -> bool:
        if rotation not in [0, 90, 180, 270]:
            logging.error(f"Invalid rotation: {rotation}")
            return False
        return True
    
    def _check_cooldown(self) -> bool:
        return time.time() - self.last_change_time >= self.rotation_cooldown
    
    @property
    def cam1_rotation(self):
        return self._cam1_mp_value
    
    @property
    def cam2_rotation(self):
        return self._cam2_mp_value
    
    @property
    def email_rotation_cam1(self):
        return self._email_mp_value

@dataclass
class FrameBuffer:
    """Frame pipeline for a single camera"""
    raw: Optional[any] = None
    rotated: Optional[any] = None
    processed: Optional[any] = None
    yolo_annotated: Optional[any] = None
    previous: Optional[any] = None
    
    def get_display_frame(self):
        """Get best frame for display (priority: YOLO > processed > rotated > raw)"""
        return self.yolo_annotated or self.processed or self.rotated or self.raw

@dataclass
class FrameManager:
    """Unified frame management for both cameras"""
    cam1: FrameBuffer = field(default_factory=FrameBuffer)
    cam2: FrameBuffer = field(default_factory=FrameBuffer)
    cam1_blank: Optional[any] = None
    cam2_blank: Optional[any] = None
    
    def initialize_blanks(self, cam1_resolution: Tuple[int, int], cam2_resolution: Tuple[int, int]):
        import numpy as np
        self.cam1_blank = np.zeros((cam1_resolution[1], cam1_resolution[0], 3), dtype=np.uint8)
        self.cam2_blank = np.zeros((cam2_resolution[1], cam2_resolution[0], 3), dtype=np.uint8)
    
    def get_camera(self, camera_name: str) -> FrameBuffer:
        return self.cam1 if camera_name == "cam1" else self.cam2
    
    def get_blank(self, camera_name: str):
        return self.cam1_blank if camera_name == "cam1" else self.cam2_blank

@dataclass
class YOLOFuture:
    """Track YOLO future with metadata"""
    future: any  # concurrent.futures.Future
    stage: str  # "roi" or "full_frame"
    camera: str  # "cam1" or "cam2"
    submission_time: float
    frame_shape: Tuple[int, int, int]  # Shape of submitted frame

@dataclass
class YOLOManager:
    """Manages two-stage YOLO with proper future tracking"""
    pending: List[YOLOFuture] = field(default_factory=list)
    processor: Optional[any] = None  # ParallelYOLOProcessor
    last_submission_time: float = 0.0
    submission_interval: float = 1.5
    
    def can_submit(self, current_time: float) -> bool:
        """Check if enough time has passed since last submission"""
        return (current_time - self.last_submission_time) >= self.submission_interval
    
    def submit_roi(self, frame, camera: str, roi_coords: Tuple[int, int, int, int], current_time: float) -> Optional[YOLOFuture]:
        """Submit ROI for YOLO analysis"""
        if not self.can_submit(current_time) or self.processor is None:
            return None
        
        x, y, w, h = roi_coords
        roi_frame = frame[y:y+h, x:x+w].copy()
        
        future = self.processor.process_frame_async(roi_frame, camera, current_time)
        yolo_future = YOLOFuture(
            future=future,
            stage="roi",
            camera=camera,
            submission_time=current_time,
            frame_shape=roi_frame.shape
        )
        
        self.pending.append(yolo_future)
        self.last_submission_time = current_time
        logging.info(f"Submitted ROI frame ({w}x{h}) from {camera} to YOLO")
        return yolo_future
    
    def submit_full_frame(self, frame, camera: str, current_time: float) -> Optional[YOLOFuture]:
        """Submit full frame for YOLO verification"""
        if self.processor is None:
            return None
        
        future = self.processor.process_frame_async(frame, camera, current_time)
        yolo_future = YOLOFuture(
            future=future,
            stage="full_frame",
            camera=camera,
            submission_time=current_time,
            frame_shape=frame.shape
        )
        
        self.pending.append(yolo_future)
        self.last_submission_time = 0  # Allow immediate next submission if needed
        logging.info(f"Submitted FULL frame from {camera} to YOLO")
        return yolo_future
    
    def process_results(self, state_mgr: DetectionStateManager) -> Optional[Tuple[str, bool, any]]:
        """Process completed futures, returns (stage, contains_objects, annotated_frame) if done"""
        completed = []
        
        for yf in self.pending[:]:
            if yf.future.done():
                try:
                    result = yf.future.result(timeout=0.1)
                    completed.append((yf, result))
                    self.pending.remove(yf)
                except Exception as e:
                    logging.error(f"YOLO future error: {e}")
                    self.pending.remove(yf)
        
        # Process in order of submission
        for yf, result in completed:
            contains_objects, annotated_frame, processing_time = result
            
            if yf.stage == "roi":
                if contains_objects:
                    logging.info(f"[OK] Person detected in ROI on {yf.camera.upper()} ({processing_time:.2f}s)")
                    state_mgr.roi_detected_person = True
                    return ("roi", True, annotated_frame)
                else:
                    logging.debug(f"No person in ROI on {yf.camera.upper()}")
            
            elif yf.stage == "full_frame":
                if contains_objects:
                    logging.info(f"[OK] Person CONFIRMED in full frame on {yf.camera.upper()} ({processing_time:.2f}s)")
                    state_mgr.full_frame_detected_person = True
                    return ("full_frame", True, annotated_frame)
                else:
                    logging.warning(f"ROI had person but full frame verification failed on {yf.camera.upper()}")
                    return ("full_frame", False, None)
        
        return None

@dataclass
class RecordingManager:
    """Manages video recording with proper frame synchronization"""
    active: bool = False
    start_time: Optional[float] = None
    video_writer: Optional[any] = None
    temp_video_path: Optional[str] = None
    detected_frame: Optional[any] = None
    source_camera: str = "cam1"
    duration: float = 10.0
    resolution: Tuple[int, int] = (640, 480)
    
    def start(self, frame, camera: str, resolution: Tuple[int, int], current_time: float) -> bool:
        """Start recording with synchronized frame"""
        if self.active:
            logging.warning("Recording already active")
            return False
        
        import tempfile
        
        self.detected_frame = frame.copy()
        self.source_camera = camera
        self.resolution = resolution
        self.start_time = current_time
        
        # Create temp video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        self.temp_video_path = temp_file.name
        temp_file.close()
        
        # Create video writer
        self.video_writer = cv2.VideoWriter(
            self.temp_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            resolution
        )
        
        if not self.video_writer.isOpened():
            logging.error(f"Failed to open video writer: {self.temp_video_path}")
            return False
        
        self.active = True
        logging.info(f"Started recording from {camera} to {self.temp_video_path}")
        return True
    
    def write_frame(self, frame) -> bool:
        """Write frame to video"""
        if not self.active or self.video_writer is None:
            return False
        
        # Resize frame if needed
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)
        
        self.video_writer.write(frame)
        return True
    
    def should_stop(self, current_time: float) -> bool:
        """Check if recording duration has elapsed"""
        if not self.active or self.start_time is None:
            return False
        return (current_time - self.start_time) >= self.duration
    
    def stop(self) -> Optional[str]:
        """Stop recording and return video path"""
        if not self.active:
            return None
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        video_path = self.temp_video_path
        actual_duration = time.time() - self.start_time if self.start_time else 0
        
        logging.info(f"Stopped recording (duration: {actual_duration:.2f}s)")
        
        # Reset state
        self.active = False
        self.start_time = None
        self.temp_video_path = None
        
        return video_path

# PYTHON 3.14: Lock-Free Data Structures
# ============================================================================

class LockFreeQueue:
    """
    Thread-safe queue optimized for Python 3.14 free-threading.
    Uses deque with atomic operations in free-threaded Python,
    or traditional locks when GIL is enabled for safety.
    """
    def __init__(self, maxsize: int = 2):
        self._queue = deque(maxlen=maxsize)
        self._maxsize = maxsize
        self._dropped_frames = 0
        self._last_warning_time = 0.0

        # Use locks if GIL is enabled for thread safety
        self._use_locks = not GIL_DISABLED
        if self._use_locks:
            self._lock = threading.Lock()
            logging.debug(f"LockFreeQueue initialized with locks (GIL enabled)")
        else:
            self._lock = None
            logging.debug(f"LockFreeQueue initialized lock-free (GIL disabled)")

    def put(self, item: Any) -> None:
        """Non-blocking put, drops oldest if full"""
        if self._use_locks:
            with self._lock:
                self._put_impl(item)
        else:
            # In free-threaded Python 3.14, deque operations are atomic
            self._put_impl(item)

    def _put_impl(self, item: Any) -> None:
        """Internal put implementation"""
        was_full = len(self._queue) >= self._maxsize
        if was_full:
            self._dropped_frames += 1
            try:
                self._queue.popleft()
            except IndexError:
                pass
            # Log warning if queue consistently full (rate-limited to once per 60s)
            current_time = time.time()
            if current_time - self._last_warning_time > 60:
                logging.warning(f"Queue full, dropped {self._dropped_frames} frames total. Consider increasing FRAME_QUEUE_SIZE or reducing processing load.")
                self._last_warning_time = current_time
        self._queue.append(item)

        # Warn if queue is >80% full (rate-limited)
        if len(self._queue) > (self._maxsize * 0.8):
            current_time = time.time()
            if current_time - self._last_warning_time > 60:
                logging.warning(f"Queue {len(self._queue)}/{self._maxsize} ({len(self._queue)/self._maxsize*100:.0f}% full) - may start dropping frames soon")
                self._last_warning_time = current_time

    def get(self) -> Optional[Any]:
        """Non-blocking get, returns None if empty"""
        if self._use_locks:
            with self._lock:
                try:
                    return self._queue.popleft()
                except IndexError:
                    return None
        else:
            try:
                return self._queue.popleft()  # Atomic in free-threaded mode
            except IndexError:
                return None

    def get_nowait(self) -> Any:
        """Raises IndexError if empty"""
        if self._use_locks:
            with self._lock:
                return self._queue.popleft()
        else:
            return self._queue.popleft()

    def empty(self) -> bool:
        if self._use_locks:
            with self._lock:
                return len(self._queue) == 0
        else:
            return len(self._queue) == 0

    def full(self) -> bool:
        if self._use_locks:
            with self._lock:
                return len(self._queue) >= self._maxsize
        else:
            return len(self._queue) >= self._maxsize

    def qsize(self) -> int:
        if self._use_locks:
            with self._lock:
                return len(self._queue)
        else:
            return len(self._queue)

    def get_dropped_frames(self) -> int:
        """Return total number of dropped frames"""
        if self._use_locks:
            with self._lock:
                return self._dropped_frames
        else:
            return self._dropped_frames

    def get_fullness_percent(self) -> float:
        """Return queue fullness as percentage (0-100)"""
        if self._use_locks:
            with self._lock:
                return (len(self._queue) / self._maxsize) * 100 if self._maxsize > 0 else 0
        else:
            return (len(self._queue) / self._maxsize) * 100 if self._maxsize > 0 else 0

@dataclass
class AtomicCameraState:
    """Lock-free camera state using atomic operations"""
    rotation: int = 0  # 0, 90, 180, 270
    last_rotation_time: float = 0.0
    available: bool = False

    def rotate_atomic(self, current_time: float, cooldown: float = 0.5) -> bool:
        """Atomic rotation update without locks"""
        # In Python 3.14 free-threaded mode, simple assignments are thread-safe
        if current_time - self.last_rotation_time >= cooldown:
            self.rotation = (self.rotation + 90) % 360
            self.last_rotation_time = current_time
            return True
        return False

    def set_rotation(self, rotation: Rotation) -> None:
        """Set rotation directly"""
        self.rotation = rotation

    def get_rotation(self) -> int:
        """Get current rotation"""
        return self.rotation

# ============================================================================
# Camera Reconnection with Exponential Backoff
# ============================================================================

class CameraReconnector:
    """
    Handles camera reconnection with exponential backoff.
    Provides resilient camera initialization for both Pi and USB cameras.
    """
    def __init__(self, camera_name: str, initial_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.camera_name = camera_name
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.current_delay = initial_delay
        self.attempt_count = 0
        self.total_reconnects = 0
        self.last_attempt_time = 0.0

    def reset(self):
        """Reset backoff after successful connection"""
        self.current_delay = self.initial_delay
        self.attempt_count = 0
        logging.info(f"{self.camera_name}: Reconnection backoff reset after successful connection")

    def should_retry(self, current_time: float) -> bool:
        """Check if enough time has passed to retry connection"""
        if current_time - self.last_attempt_time >= self.current_delay:
            return True
        return False

    def record_attempt(self, current_time: float, success: bool = False):
        """Record connection attempt and update backoff delay"""
        self.last_attempt_time = current_time
        self.attempt_count += 1

        if success:
            self.total_reconnects += 1
            logging.info(f"{self.camera_name}: Reconnected successfully (attempt {self.attempt_count}, total reconnects: {self.total_reconnects})")
            self.reset()
        else:
            # Exponential backoff
            self.current_delay = min(self.current_delay * self.backoff_factor, self.max_delay)
            logging.warning(f"{self.camera_name}: Connection failed (attempt {self.attempt_count}), next retry in {self.current_delay:.1f}s")

    def get_stats(self) -> dict:
        """Get reconnection statistics"""
        return {
            "camera": self.camera_name,
            "attempt_count": self.attempt_count,
            "total_reconnects": self.total_reconnects,
            "current_delay": self.current_delay,
            "max_delay": self.max_delay
        }

# ============================================================================
# PYTHON 3.14: Parallel YOLO Processor
# ============================================================================

class ParallelYOLOProcessor:
    """
    Parallel YOLO processing using Python 3.14 free-threading.
    Processes multiple frames simultaneously without GIL contention.
    """

    def __init__(self, config: YOLOConfig, num_workers: int = 2):
        self.config = config
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="YOLO-Worker"
        )
        self.model_path = config.model_path
        self._models = {}  # Thread-local models

        logging.info(f"Initialized ParallelYOLOProcessor with {num_workers} workers")

    def _get_model(self) -> YOLO:
        """Get thread-local YOLO model instance"""
        thread_id = threading.get_ident()
        if thread_id not in self._models:
            self._models[thread_id] = YOLO(self.model_path)
            logging.info(f"Loaded YOLO model in thread {thread_id}")
        return self._models[thread_id]

    def process_frame_async(
        self,
        frame: np.ndarray,
        camera: CameraName,
        request_time: float
    ) -> Any:  # Returns Future[Dict[str, Any]]
        """Submit frame for async parallel processing"""
        return self.executor.submit(
            self._detect_in_thread,
            frame, camera, request_time
        )

    def _detect_in_thread(
        self,
        frame: np.ndarray,
        camera: CameraName,
        request_time: float
    ) -> Dict[str, Any]:
        """
        Run YOLO detection in worker thread.
        Truly parallel in free-threaded Python - no GIL contention.
        """
        model = self._get_model()

        # Run detection
        results = model(frame, conf=self.config.confidence_threshold, verbose=False)

        # Process results - support both bounding boxes and segmentation masks
        detections = []
        contains_objects_of_interest = False

        for r in results:
            # Check if segmentation masks are available
            has_masks = r.masks is not None

            for i, box in enumerate(r.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detection = {
                    'class': cls,
                    'class_name': cls_name,
                    'confidence': conf,
                    'box': (x1, y1, x2, y2),
                    'has_mask': has_masks
                }

                # Add mask data if available
                if has_masks and i < len(r.masks.data):
                    mask = r.masks.data[i].cpu().numpy()  # Shape: (H, W)
                    detection['mask'] = mask
                    mask_area = int(mask.sum())
                    detection['mask_area'] = mask_area

                    logging.info(
                        f"YOLO segmentation: class={cls} ({cls_name}), "
                        f"conf={conf:.2f}, box=({x1},{y1},{x2},{y2}), "
                        f"mask_area={mask_area}px"
                    )
                else:
                    logging.info(
                        f"YOLO detection: class={cls} ({cls_name}), "
                        f"conf={conf:.2f}, box=({x1},{y1},{x2},{y2})"
                    )

                detections.append(detection)

                # Check if this object is of interest
                is_of_interest = cls in self.config.classes_of_interest
                if is_of_interest:
                    contains_objects_of_interest = True

        # Get annotated frame
        annotated_frame = results[0].plot()

        processing_time = time.time() - request_time
        logging.debug(f"YOLO processing took {processing_time:.2f}s for {camera}")

        return {
            'camera': camera,
            'detections': detections,
            'annotated_frame': annotated_frame,
            'contains_objects': contains_objects_of_interest,
            'processing_time': processing_time
        }

    def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logging.info("YOLO processor shutdown complete")

# ============================================================================
# PYTHON 3.14: Performance Monitoring
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Track performance metrics"""
    frame_processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    yolo_inference_times: deque = field(default_factory=lambda: deque(maxlen=100))
    fps_samples: deque = field(default_factory=lambda: deque(maxlen=30))
    thread_count_samples: deque = field(default_factory=lambda: deque(maxlen=30))

    def record_frame_time(self, duration: float):
        self.frame_processing_times.append(duration)

    def record_yolo_time(self, duration: float):
        self.yolo_inference_times.append(duration)

    def record_fps(self, fps: float):
        self.fps_samples.append(fps)

    def get_average_frame_time(self) -> float:
        if not self.frame_processing_times:
            return 0.0
        return sum(self.frame_processing_times) / len(self.frame_processing_times)

    def get_average_yolo_time(self) -> float:
        if not self.yolo_inference_times:
            return 0.0
        return sum(self.yolo_inference_times) / len(self.yolo_inference_times)

    def get_average_fps(self) -> float:
        if not self.fps_samples:
            return 0.0
        return sum(self.fps_samples) / len(self.fps_samples)

    def log_summary(self):
        """Log performance summary"""
        logging.info("=== Performance Summary ===")
        logging.info(f"Avg Frame Processing: {self.get_average_frame_time():.3f}s")
        logging.info(f"Avg YOLO Inference: {self.get_average_yolo_time():.3f}s")
        logging.info(f"Avg FPS: {self.get_average_fps():.1f}")
        logging.info(f"Active Threads: {threading.active_count()}")
        logging.info(f"GIL Status: {'Disabled' if GIL_DISABLED else 'Enabled'}")

def measure_performance(metric_name: str):
    """Decorator to measure function performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            logging.debug(f"{metric_name}: {duration:.3f}s")
            return result
        return wrapper
    return decorator

# ============================================================================
# PYTHON 3.14: Context Managers for Safe Resource Management
# ============================================================================

@contextmanager
def camera_context(
    device_index: int,
    resolution: Tuple[int, int],
    backend: int = cv2.CAP_V4L2
):
    """Safe camera resource management with automatic cleanup"""
    cap = None
    try:
        cap = cv2.VideoCapture(device_index, backend)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {device_index}. "
                f"Check: sudo usermod -a -G video $USER"
            )

        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)

        logging.info(f"Camera {device_index} opened: {resolution[0]}x{resolution[1]}")

        yield cap

    except OSError as e:
        logging.error(
            f"Camera device error: {e.strerror} (errno: {e.errno}). "
            f"Ensure v4l2loopback is loaded: sudo modprobe v4l2loopback"
        )
        raise
    finally:
        if cap is not None:
            cap.release()
            logging.info(f"Camera {device_index} released")

@contextmanager
def video_writer_context(
    filename: str,
    fourcc: str,
    fps: int,
    resolution: Tuple[int, int]
):
    """Safe video writer resource management"""
    writer = None
    try:
        writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            resolution
        )

        if not writer.isOpened():
            raise IOError(f"Cannot create video writer for {filename}")

        logging.info(f"Video writer created: {filename} ({resolution[0]}x{resolution[1]} @ {fps}fps)")

        yield writer

    finally:
        if writer is not None:
            writer.release()
            logging.info(f"Video writer released: {filename}")

# ============================================================================

# ============================================================================
# PYTHON 3.14: Global Instances (Lock-Free)
# ============================================================================

# Global performance metrics instance
perf_metrics = PerformanceMetrics()

# Global camera states (no locks needed with free-threading)
cam1_state = AtomicCameraState()
cam2_state = AtomicCameraState()

# NEW ARCHITECTURE: Global Manager Instances
state_manager = DetectionStateManager()
rotation_manager = RotationManager()
frame_manager = FrameManager()
yolo_manager = YOLOManager()
recording_manager = RecordingManager()


# ============================================================================

# ============================================================================
# PYTHON 3.14: Refactored Global Variables (Lock-Free)
# ============================================================================

# Application state
live_feed_url = None
streaming_active = False
# Detection control - two separate flags with distinct purposes:
# - detection_active: Controls whether motion detection processing runs (can be temporarily disabled during YOLO/recording)
# - email_armed: User-facing armed/disarmed state (controls whether emails are sent when motion confirmed)
# These serve different purposes and are intentionally separate. detection_active is an internal processing flag,
# while email_armed is the user-controlled arming state.
detection_active = True
detection_camera = "cam1"
pipeline_process = None

# PYTHON 3.14: Lock-free queues (replaced queue.Queue with LockFreeQueue)
# Note: frame_queue for Pi camera pipeline (single process), usb_frame_queue must use multiprocessing.Queue
frame_queue = LockFreeQueue(maxsize=FRAME_QUEUE_SIZE)
usb_frame_queue = multiprocessing.Queue(maxsize=FRAME_QUEUE_SIZE)  # Must use multiprocessing.Queue for subprocess

# Camera rotation state (NOW MANAGED BY rotation_manager for single source of truth)
# These are kept for backwards compatibility - they reference rotation_manager's multiprocessing values
cam1_rotation = rotation_manager.cam1_rotation
cam2_rotation = rotation_manager.cam2_rotation
email_rotation_cam1 = rotation_manager.email_rotation_cam1

cam2_available = multiprocessing.Value('b', False)  # Flag to indicate USB camera availability

# Email control
# Email armed state (see detection_active above for relationship)
email_armed = True
cooldown_active = False
button_visible = False
last_email_time = 0
last_rotation_time = 0

# Timer variables for scheduled arming/disarming
timer_enabled = False
schedule_arm_time = None
schedule_disarm_time = None
schedule_arm_seconds = None
schedule_disarm_seconds = None
last_timer_check = 0

# Screen dimensions
screen_width = 0
screen_height = 0

# PYTHON 3.14: Removed rotation locks - now using AtomicCameraState (cam1_state, cam2_state)
# rotation_lock = threading.Lock()  # REMOVED - lock-free now
# media_lock = threading.Lock()  # REMOVED - will use atomic operations

# Configuration
config = {}  # Will be replaced with AppConfig instance
app_config: Optional[AppConfig] = None  # PYTHON 3.14: Type-safe config
authorized_users = {}

# Background subtractors
bg_subtractor_cam1 = None
bg_subtractor_cam2 = None

# KNN warmup frame counters (for adaptive learning rate during initialization)
knn_frame_counter_cam1 = 0
knn_frame_counter_cam2 = 0

# PYTHON 3.14: YOLO will be replaced with ParallelYOLOProcessor
yolo_processor: Optional[ParallelYOLOProcessor] = None  # New parallel processor
yolo_model = None  # Legacy - will be removed after migration

# YOLO analysis state
yolo_annotated_pi_frame = None
yolo_annotated_usb_frame = None

# Motion detection tracking
motion_source_camera = "cam1"  # Default to cam1
motion_detection_time_cam1 = None  # Time when camera 1 detected motion
motion_detection_time_cam2 = None  # Time when camera 2 detected motion

# ROI selection variables
roi_selection_mode = False
roi_start_point = None
roi_end_point = None
roi_drawing = False
roi_temp = None
settings_dialog = None  # Global reference to settings dialog

# Detect local IP
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        logging.info(f"Detected local IP: {local_ip}")
        return local_ip
    except Exception as e:
        logging.error(f"Error detecting local IP: {e}")
        return "127.0.0.1"

# Load config
CONFIG_FILE = "config.json"
# ============================================================================
# PYTHON 3.14: Modern Config Loading with Type Safety
# ============================================================================

def load_config_modern() -> AppConfig:
    """
    PYTHON 3.14: Load configuration using type-safe AppConfig dataclass.
    This is the preferred method for new code.
    """
    global app_config, yolo_processor

    try:
        # Load type-safe configuration
        app_config = AppConfig.load_from_json(CONFIG_FILE)

        # Initialize ParallelYOLOProcessor if YOLO is enabled
        if app_config.yolo.enabled:
            num_workers = 2 if GIL_DISABLED else 1  # More workers in free-threaded mode
            yolo_processor = ParallelYOLOProcessor(app_config.yolo, num_workers=num_workers)
            logging.info(f"Initialized YOLO processor with {num_workers} workers")

        logging.info("[OK] Loaded type-safe configuration successfully")
        return app_config

    except Exception as e:
        logging.error(f"Failed to load modern config: {e}")
        # Fall back to default
        app_config = AppConfig()
        return app_config

def load_config():
    """Legacy config loading - wraps modern AppConfig loader for backward compatibility"""
    global config, detection_camera, email_rotation_cam1, app_config
    global timer_enabled, schedule_arm_time, schedule_disarm_time, schedule_arm_seconds, schedule_disarm_seconds

    # PYTHON 3.14: Load modern type-safe config first
    app_config = load_config_modern()

    # Populate legacy config dict for backward compatibility
    detection_camera = app_config.detection_camera
    timer_enabled = app_config.timer_enabled
    schedule_arm_time = app_config.schedule_arm_time
    schedule_disarm_time = app_config.schedule_disarm_time

    local_ip = get_local_ip()
    default_config = {
        "live_feed_url": os.getenv('LIVE_FEED_URL', f"http://{local_ip}:5000/login"),
        "flask_host": os.getenv('FLASK_HOST', "0.0.0.0"),
        "flask_port": int(os.getenv('FLASK_PORT', 5000)),
        "users": [
            {"email": os.getenv('USER1_EMAIL', "user1@example.com"), "password_hash": bcrypt.hashpw(os.getenv('USER1_PASSWORD', "password1").encode('utf-8'), bcrypt.gensalt()).decode('utf-8')},
            {"email": os.getenv('USER2_EMAIL', "user2@example.com"), "password_hash": bcrypt.hashpw(os.getenv('USER2_PASSWORD', "password2").encode('utf-8'), bcrypt.gensalt()).decode('utf-8')}
        ],
        "sender_email": os.getenv('SENDER_EMAIL', ""),
        "receiver_emails": [os.getenv('RECEIVER_EMAIL1', ""), os.getenv('RECEIVER_EMAIL2', "")],
        "email_password": os.getenv('EMAIL_PASSWORD', ""),
        "detection_camera": "cam1",
        "email_rotation_cam1": 0,
        "roi_coordinates": {"x": 0, "y": 0, "width": 640, "height": 480},
        "roi_coordinates_cam2": {"x": 0, "y": 0, "width": 320, "height": 240},
        "background_threshold": 25,
        "cam2_area_multiplier": 4,
        "cam2_threshold_multiplier": 1.5,
        "cam2_min_contours": 0,
        "cam1_resolution": "640x480",
        "cam2_resolution": "320x240",
        # Default timer settings
        "timer_enabled": False,
        "schedule_arm_time": "08:00",
        "schedule_disarm_time": "18:00",
        "schedule_arm_seconds": 8 * 3600,  # 8:00 AM in seconds
        "schedule_disarm_seconds": 18 * 3600,  # 6:00 PM in seconds
        # YOLO settings
        "use_yolo_detection": True,  # Enable/disable YOLO
        "yolo_confidence_threshold": 0.5,  # Minimum confidence for detection
        "yolo_classes_of_interest": [0, 1, 2, 3, 5, 7],  # Person, bicycle, car, motorcycle, bus, truck (COCO indices)
        "yolo_model_path": "yolov8m-seg.pt"  # Path to YOLOv8 segmentation model file
    }
    logging.debug(f"Loading config, initial config type: {type(config)}")
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                for key, value in default_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                config.update(loaded_config)
                detection_camera = config.get("detection_camera", "cam1")
                with email_rotation_cam1.get_lock():
                    email_rotation_cam1.value = config.get("email_rotation_cam1", 0)
                
                # Load timer settings
                timer_enabled = config.get("timer_enabled", False)
                schedule_arm_time = config.get("schedule_arm_time", "08:00")
                schedule_disarm_time = config.get("schedule_disarm_time", "18:00")
                schedule_arm_seconds = config.get("schedule_arm_seconds", 8 * 3600)  # Default 8:00 AM
                schedule_disarm_seconds = config.get("schedule_disarm_seconds", 18 * 3600)  # Default 6:00 PM
                
                logging.info("Loaded config merged with defaults")
                logging.debug(f"Config after load: {config}")
                return config
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding config.json: {e}. Using default config.")
            config.update(default_config)
            return config
    logging.info("No config file found, using default config")
    config.update(default_config)
    logging.debug(f"Config after default: {config}")

    # Set timer settings from default config
    timer_enabled = default_config["timer_enabled"]
    schedule_arm_time = default_config["schedule_arm_time"]
    schedule_disarm_time = default_config["schedule_disarm_time"]
    schedule_arm_seconds = default_config["schedule_arm_seconds"]
    schedule_disarm_seconds = default_config["schedule_disarm_seconds"]

    # Validate loaded ROI coordinates for both cameras
    validate_roi_coordinates("roi_coordinates", 640, 480)
    validate_roi_coordinates("roi_coordinates_cam2", 320, 240)
    
    return config

def validate_roi_coordinates(roi_key, default_width, default_height):
    """Helper function to validate ROI coordinates for a specific camera"""
    global config
    roi_coords = config.get(roi_key)
    valid_roi = False
    
    if isinstance(roi_coords, list) and len(roi_coords) == 4:
        w, h = roi_coords[2], roi_coords[3]
        if w > 0 and h > 0:
            valid_roi = True
    elif isinstance(roi_coords, dict):
        w = roi_coords.get("width", 0)
        h = roi_coords.get("height", 0)
        if w > 0 and h > 0:
            valid_roi = True
            # Convert dict to list format internally for consistency
            config[roi_key] = [roi_coords.get("x", 0), roi_coords.get("y", 0), w, h]
            
    if not valid_roi:
        logging.warning(f"Invalid {roi_key} found in config: {roi_coords}. Resetting to default.")
        # Reset to default size for this camera
        config[roi_key] = [0, 0, default_width, default_height]

def load_yolo_model():
    """Load the YOLO model based on config settings"""
    global yolo_model
    
    if not config.get('use_yolo_detection', False):
        yolo_model = None
        logging.info("YOLO detection disabled in config")
        return
    
    try:
        model_path = config.get('yolo_model_path', 'yolov8m-seg.pt')  # Segmentation model for detailed object masks
        logging.info(f"Loading YOLO model from {model_path}...")
        yolo_model = YOLO(model_path)
        logging.info(f"YOLO model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {str(e)}")
        yolo_model = None

# Screen resolution
def get_screen_resolution():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

# Pipeline and cleanup
def start_pipeline():
    global pipeline_process
    
    # Get the configured resolution for Camera 1
    cam1_resolution = config.get("cam1_resolution", "640x480")
    try:
        width, height = map(int, cam1_resolution.split("x"))
    except Exception as e:
        logging.error(f"Error parsing Camera 1 resolution '{cam1_resolution}': {e}")
        width, height = 640, 480  # Default fallback
    
    logging.info(f"Starting video pipeline with resolution {width}x{height}...")
    
    pipeline_process = subprocess.Popen(
        f"libcamera-vid -o - --width {width} --height {height} --framerate 30 --nopreview --codec yuv420 --timeout 0 | "
        f"ffmpeg -f rawvideo -pixel_format yuv420p -video_size {width}x{height} -framerate 30 -i - -f v4l2 -pix_fmt yuv420p /dev/video10",
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

def cleanup(signum=None, frame=None):
    """
    Enhanced cleanup function with guaranteed resource release.
    Called by signal handlers and atexit to ensure proper shutdown.
    """
    global pipeline_process, cap, yolo_processor, usb_process

    # Prevent duplicate cleanup if called multiple times
    if hasattr(cleanup, '_already_called'):
        return
    cleanup._already_called = True

    logging.info("Starting cleanup process...")

    # PYTHON 3.14: Shutdown YOLO processor (replaces old worker thread)
    if 'yolo_processor' in globals() and yolo_processor is not None:
        try:
            logging.debug("Shutting down YOLO processor...")
            yolo_processor.shutdown()
            logging.debug("YOLO processor shut down successfully")
        except Exception as e:
            logging.error(f"Error shutting down YOLO processor: {str(e)}")

    # Safely terminate USB camera process
    if 'usb_process' in globals() and usb_process is not None:
        try:
            logging.debug("Attempting to terminate USB camera process...")
            usb_process.terminate()
            try:
                usb_process.join(timeout=3)
                logging.debug("USB camera process terminated gracefully")
            except Exception:
                logging.debug("USB camera process didn't join, forcing kill...")
                usb_process.kill()
                usb_process.join(timeout=1)
                logging.debug("USB camera process killed")
        except Exception as e:
            logging.error(f"Error during USB camera process cleanup: {str(e)}")

    # Safely terminate pipeline process
    if 'pipeline_process' in globals() and pipeline_process is not None:
        try:
            logging.debug("Attempting to terminate pipeline process...")
            try:
                os.killpg(os.getpgid(pipeline_process.pid), signal.SIGTERM)
                pipeline_process.wait(timeout=2)
                logging.debug("Pipeline process terminated gracefully")
            except (subprocess.TimeoutExpired, ProcessLookupError) as e:
                logging.debug(f"Initial termination failed: {str(e)}, trying SIGKILL...")
                try:
                    os.killpg(os.getpgid(pipeline_process.pid), signal.SIGKILL)
                    logging.debug("Pipeline process terminated with SIGKILL")
                except ProcessLookupError:
                    logging.debug("Process already terminated or doesn't exist")
                except Exception as e:
                    logging.error(f"Failed to kill pipeline process: {str(e)}")
        except Exception as e:
            logging.error(f"Error during pipeline cleanup: {str(e)}")

    # Safely release Pi camera
    if 'cap' in globals() and cap is not None:
        try:
            cap.release()
            logging.debug("Pi camera released successfully")
        except Exception as e:
            logging.error(f"Error releasing Pi camera: {str(e)}")

    # Cleanup OpenCV windows
    try:
        cv2.destroyAllWindows()
        logging.debug("Destroyed all OpenCV windows")
    except Exception as e:
        logging.error(f"Error destroying windows: {str(e)}")

    # Cleanup Tkinter UI if it exists
    if 'root' in globals() and root is not None:
        try:
            root.quit()
            logging.debug("Tkinter event loop stopped")
        except Exception as e:
            logging.debug(f"Error stopping Tkinter: {str(e)}")

    logging.info("Pipeline and application stopped cleanly")

    # Only call sys.exit if called from signal handler
    if signum is not None:
        sys.exit(0)

# Register cleanup handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)
atexit.register(cleanup)  # Guaranteed cleanup on normal exit

# Flask setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'
limiter = Limiter(app, default_limits=["5 per minute"], storage_uri="memory://")

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Log In')

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Modify the USB camera process to make rotation more stable
def usb_camera_process(usb_queue, rotation_value, cam2_available, cam2_resolution="1280x720"):
    """USB camera process with reconnection resilience"""

    # Parse resolution
    try:
        width, height = map(int, cam2_resolution.split("x"))
    except Exception as e:
        logging.error(f"Error parsing Camera 2 resolution '{cam2_resolution}': {e}")
        width, height = 320, 240  # Default fallback

    logging.info(f"USB camera resolution set to {width}x{height}")

    # Initialize reconnection handler
    reconnector = CameraReconnector("USB Camera", initial_delay=2.0, max_delay=60.0)

    def try_init_usb_camera():
        """Attempt to initialize USB camera, return (camera, device_index) or (None, None)"""
        for index in range(20):
            if index == 10:  # Skip Pi camera index
                continue
            temp_cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if temp_cam.isOpened():
                # Set resolution BEFORE reading any frames for better compatibility
                temp_cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                temp_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                temp_cam.set(cv2.CAP_PROP_FPS, 30)

                # Verify what resolution was actually set
                actual_width = int(temp_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(temp_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Read and discard several frames to let camera stabilize at new resolution
                for _ in range(10):
                    temp_cam.read()

                # Now try to read a frame
                ret, frame = temp_cam.read()
                if ret and frame is not None:
                    if actual_width == width and actual_height == height:
                        logging.info(f"USB camera detected at index {index}, resolution successfully set to {actual_width}x{actual_height}")
                    else:
                        logging.warning(f"USB camera at index {index}: requested {width}x{height}, but got {actual_width}x{actual_height}")
                    return temp_cam, index
            temp_cam.release()
        return None, None

    # Initial camera connection
    usb_cam, camera_index = try_init_usb_camera()
    if usb_cam is None:
        logging.error("No USB camera detected on initial attempt, will retry with backoff...")
        with cam2_available.get_lock():
            cam2_available.value = False
    else:
        with cam2_available.get_lock():
            cam2_available.value = True
        reconnector.reset()  # Start with clean state

    frame_interval = 1 / 30  # Target 30 FPS for smoother display
    last_frame_time = time.time()

    # Main loop with reconnection resilience
    while True:
        current_time = time.time()

        # If camera is not available, try to reconnect
        if usb_cam is None or not usb_cam.isOpened():
            if reconnector.should_retry(current_time):
                logging.info(f"Attempting to reconnect USB camera...")
                usb_cam, camera_index = try_init_usb_camera()
                reconnector.record_attempt(current_time, success=(usb_cam is not None))

                if usb_cam is not None:
                    with cam2_available.get_lock():
                        cam2_available.value = True
                else:
                    with cam2_available.get_lock():
                        cam2_available.value = False

            # Sleep and continue if still no camera
            if usb_cam is None:
                time.sleep(0.1)
                continue

        # Frame timing control
        if current_time - last_frame_time < frame_interval:
            time.sleep(0.001)
            continue

        # Try to read frame
        ret, frame = usb_cam.read()
        if ret and frame is not None:
            # Rotation is now handled in main loop, not here
            if not usb_queue.full():
                usb_queue.put(frame)
            last_frame_time = current_time
        else:
            # Frame read failed - prepare for reconnection
            logging.warning(f"Failed to read USB camera frame (index {camera_index}), will attempt reconnection")
            usb_cam.release()
            usb_cam = None
            with cam2_available.get_lock():
                cam2_available.value = False
            # Don't break - let reconnection logic handle it in next loop iteration

# Utility functions
def resize_and_pad(frame, target_width, target_height):
    if frame is None or frame.size == 0:
        logging.warning(f"Frame is None or empty, returning blank {target_width}x{target_height}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        logging.warning(f"Frame has zero dimensions ({w}x{h}), returning blank {target_width}x{target_height}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    try:
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
        return canvas
    except Exception as e:
        logging.error(f"Error in resize_and_pad: {e}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

def apply_rotation(frame, rotation):
    if frame is None or frame.size == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# Fix the gen_frames function to completely stabilize web cam2 rotation
def gen_frames(web_cam1_rotation, web_cam2_rotation):
    global streaming_active, screen_width, screen_height, cam1_rotation, cam2_rotation
    streaming_active = True
    last_picam_frame = None
    last_usb_frame = None
    frame_interval = 1 / 30  # Target 30 FPS for smoother display
    last_frame_time = time.time()
    logging.info("Streaming started")
    
    try:
        while streaming_active:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue

            # PYTHON 3.14: Lock-free queue access - get() returns None if empty
            picam_frame = frame_queue.get() or last_picam_frame
            usb_frame = usb_frame_queue.get() or last_usb_frame

            if picam_frame is not None and picam_frame.size != 0:
                last_picam_frame = picam_frame.copy()
            if usb_frame is not None and usb_frame.size != 0:
                last_usb_frame = usb_frame.copy()
                
            try:
                # Apply web rotation to Pi camera frame - this is for web UI only
                picam_frame_rotated = apply_rotation(last_picam_frame, web_cam1_rotation) if last_picam_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Apply web rotation to USB camera frame - this is for web UI only
                usb_frame_rotated = apply_rotation(last_usb_frame, web_cam2_rotation) if last_usb_frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
                
            except Exception as e:
                logging.error(f"Rotation error in gen_frames: {e}")
                picam_frame_rotated = np.zeros((480, 640, 3), dtype=np.uint8)
                usb_frame_rotated = np.zeros((240, 320, 3), dtype=np.uint8)
            picam_valid = picam_frame_rotated is not None and not np.all(picam_frame_rotated == 0)
            usb_valid = usb_frame_rotated is not None and not np.all(usb_frame_rotated == 0)
            try:
                if picam_valid and usb_valid:
                    target_height_per_frame = screen_height // 2
                    picam_resized = resize_and_pad(picam_frame_rotated, screen_width, target_height_per_frame)
                    usb_resized = resize_and_pad(usb_frame_rotated, screen_width, target_height_per_frame)
                    combined_frame = np.vstack((picam_resized, usb_resized))
                elif picam_valid:
                    combined_frame = resize_and_pad(picam_frame_rotated, screen_width, screen_height)
                elif usb_valid:
                    combined_frame = resize_and_pad(usb_frame_rotated, screen_width, screen_height)
                else:
                    combined_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            except Exception as e:
                logging.error(f"Stacking error in gen_frames: {e}")
                combined_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', combined_frame)
            if not ret:
                logging.warning("Failed to encode frame")
                continue
            frame_data = (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            yield frame_data
            last_frame_time = current_time
    except GeneratorExit:
        pass
    finally:
        streaming_active = False
        logging.info("Streaming stopped, client disconnected")

# Stored media functions
def generate_thumbnail(src_path, dest_path, is_video=False):
    try:
        if not os.path.exists(src_path):
            logging.error(f"Source file does not exist: {src_path}")
            return
        if is_video:
            cap = cv2.VideoCapture(src_path)
            ret, frame = cap.read()
            if ret:
                thumbnail = cv2.resize(frame, (150, 150), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dest_path, thumbnail)
                logging.debug(f"Thumbnail generated for video: {dest_path}")
            else:
                logging.error(f"Failed to read video frame: {src_path}")
            cap.release()
        else:
            img = cv2.imread(src_path)
            if img is not None:
                thumbnail = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dest_path, thumbnail)
                logging.debug(f"Thumbnail generated for image: {dest_path}")
            else:
                logging.error(f"Failed to read image: {src_path}")
    except Exception as e:
        logging.error(f"Error generating thumbnail for {src_path}: {e}")

def save_media_to_storage(image, video_path, camera):
    """Save media to storage with rotation based on camera source"""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs("stored_media", exist_ok=True)
    image_path = f"stored_media/{timestamp}_image.jpg"
    video_dest_path = f"stored_media/{timestamp}_video.mp4"
    thumbnail_image_path = f"stored_media/{timestamp}_image_thumb.jpg"
    thumbnail_video_path = f"stored_media/{timestamp}_video_thumb.jpg"

    # Use appropriate rotation based on camera source
    if camera == "cam2":
        with cam2_rotation.get_lock():
            rotated_image = apply_rotation(image, cam2_rotation.value)
    else:
        with email_rotation_cam1.get_lock():
            rotated_image = apply_rotation(image, email_rotation_cam1.value)

    if cv2.imwrite(image_path, rotated_image):
        logging.debug(f"Image saved: {image_path}")
    else:
        logging.error(f"Failed to save image: {image_path}")

    shutil.copy(video_path, video_dest_path)
    generate_thumbnail(image_path, thumbnail_image_path, is_video=False)
    generate_thumbnail(video_dest_path, thumbnail_video_path, is_video=True)
    logging.info(f"Media saved to storage: {image_path}, {video_dest_path}")

def purge_old_media():
    """PYTHON 3.14: Lock-free media purging - filesystem operations are atomic"""
    while True:
        # PYTHON 3.14: No media_lock needed - filesystem operations are atomic
        now = time.time()
        if os.path.exists("stored_media"):
            for filename in os.listdir("stored_media"):
                file_path = os.path.join("stored_media", filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if now - file_time > 24 * 60 * 60:
                        os.remove(file_path)
                        logging.debug(f"Purged old file: {file_path}")
        time.sleep(60)

# Flask Routes
@app.route('/video_feed', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
@login_required
def video_feed():
    with cam2_available.get_lock():
        cam2_present = cam2_available.value
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Live Feed</title>
            <style>
                * {margin: 0; padding: 0; box-sizing: border-box;}
                body {font-family: Arial, sans-serif; background-color: #000; height: 100vh; width: 100vw; overflow: hidden; position: relative;}
                .video-container {width: 100vw; height: 100vh; overflow: hidden; cursor: pointer;}
                img {width: 100%; height: 100%; object-fit: cover; display: block;}
                .button-container {position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); display: flex; justify-content: center; gap: 15px; z-index: 10;}
                button {background-color: rgba(24, 119, 242, 0.8); color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; transition: background-color 0.3s;}
                button:hover {background-color: rgba(22, 111, 229, 1);}
                @media (max-width: 600px) {button {padding: 8px 16px; font-size: 14px;}}
            </style>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const videoContainer = document.querySelector('.video-container');
                    videoContainer.addEventListener('click', toggleFullscreen);
                    videoContainer.addEventListener('touchstart', toggleFullscreen);
                    function toggleFullscreen(event) {
                        if (event.target.tagName === 'BUTTON') return;
                        if (!document.fullscreenElement) {
                            document.documentElement.requestFullscreen().catch(err => console.log(`Error: ${err.message}`));
                        } else {
                            document.exitFullscreen().catch(err => console.log(`Error: ${err.message}`));
                        }
                    }
                });
            </script>
        </head>
        <body>
            <div class="video-container">
                <img src="{{ url_for('video_stream') }}" alt="Video Feed">
            </div>
            <div class="button-container">
                <form action="{{ url_for('rotate_web_cam1') }}" method="POST" style="display:inline;">
                    <button type="submit">Rotate Camera 1</button>
                </form>
                {% if cam2_present %}
                <form action="{{ url_for('rotate_web_cam2') }}" method="POST" style="display:inline;">
                    <button type="submit">Rotate Camera 2</button>
                </form>
                {% endif %}
            </div>
        </body>
        </html>
    ''', cam2_present=cam2_present)

@app.route('/video_stream')
@login_required
def video_stream():
    # Get web UI rotation values from session
    web_cam1_rotation = session.get('web_cam1_rotation', 0)
    web_cam2_rotation = session.get('web_cam2_rotation', 0)
    
    # Log the web rotation settings
    logging.debug(f"Web stream started with cam1_web_rotation={web_cam1_rotation}deg, cam2_web_rotation={web_cam2_rotation}deg")
    
    # Pass both web rotations to gen_frames - these are independent from local display rotations
    return Response(gen_frames(web_cam1_rotation, web_cam2_rotation), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rotate_web_cam1', methods=['POST'])
@login_required
def rotate_web_cam1():
    """PYTHON 3.14: Lock-free rotation using atomic operations"""
    global last_rotation_time
    current_time = time.time()

    # PYTHON 3.14: No lock needed - atomic operation
    if current_time - last_rotation_time >= ROTATION_COOLDOWN:
        session['web_cam1_rotation'] = (session.get('web_cam1_rotation', 0) + 90) % 360
        last_rotation_time = current_time
        logging.info(f"Web: Camera 1 rotation set to {session['web_cam1_rotation']}deg (web UI only)")
    return redirect(url_for('video_feed'))

@app.route('/rotate_web_cam2', methods=['POST'])
@login_required
def rotate_web_cam2():
    """PYTHON 3.14: Lock-free rotation using atomic operations"""
    global last_rotation_time
    current_time = time.time()

    # PYTHON 3.14: No lock needed - atomic operation
    if current_time - last_rotation_time >= ROTATION_COOLDOWN:
        # Update only the web rotation in session - don't modify the shared cam2_rotation value
        session['web_cam2_rotation'] = (session.get('web_cam2_rotation', 0) + 90) % 360
        last_rotation_time = current_time
        logging.info(f"Web: Camera 2 rotation set to {session['web_cam2_rotation']}deg (web UI only)")
    return redirect(url_for('video_feed'))

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data.encode('utf-8')
        if email in authorized_users and bcrypt.checkpw(password, authorized_users[email]):
            session['user'] = email
            session['web_cam1_rotation'] = 0
            session['web_cam2_rotation'] = 0
            logging.debug(f"Session set: user={email}, cam1_rotation={session['web_cam1_rotation']}, cam2_rotation_adjustment={session['web_cam2_rotation']}")
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.')
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Login</title>
            <style>
                body {font-family: 'Arial', sans-serif; background-color: #f0f2f5; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh;}
                .login-container {background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%; max-width: 400px;}
                h1 {color: #1877f2; text-align: center;}
                form {display: flex; flex-direction: column;}
                input[type="email"], input[type="password"] {margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px;}
                input[type="submit"] {background-color: #1877f2; color: white; border: none; padding: 10px; border-radius: 4px; cursor: pointer;}
                input[type="submit"]:hover {background-color: #166fe5;}
                .flash {background-color: #f8d7da; color: #721c24; padding: 10px; margin-bottom: 15px; border-radius: 4px;}
            </style>
        </head>
        <body>
            <div class="login-container">
                <h1>Login</h1>
                {% for message in get_flashed_messages() %}
                    <div class="flash">{{ message }}</div>
                {% endfor %}
                <form method="POST">
                    {{ form.hidden_tag() }}
                    {{ form.email.label }} {{ form.email(size=32) }}
                    {{ form.password.label }} {{ form.password(size=32) }}
                    {{ form.submit() }}
                </form>
            </div>
        </body>
        </html>
    ''', form=form)

@app.route('/logout')
@login_required
def logout():
    session.pop('user', None)
    session.pop('web_cam1_rotation', None)
    session.pop('web_cam2_rotation', None)
    return redirect(url_for('login'))

@app.route('/health')
def health():
    """
    System health endpoint with comprehensive metrics.
    Returns JSON with camera status, queue metrics, and system state.
    Does not require authentication for monitoring purposes.
    """
    import json as json_module
    from flask import jsonify

    try:
        health_data = {
            "status": "running",
            "timestamp": time.time(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "gil_disabled": GIL_DISABLED,
            "cameras": {
                "pi_camera": {
                    "available": cap is not None and (cap.isOpened() if hasattr(cap, 'isOpened') else False),
                    "device": "/dev/video10"
                },
                "usb_camera": {
                    "available": cam2_available.value if 'cam2_available' in globals() else False,
                    "device": "auto-detected"
                }
            },
            "queues": {
                "frame_queue": {
                    "size": frame_queue.qsize() if 'frame_queue' in globals() and hasattr(frame_queue, 'qsize') else 0,
                    "max_size": FRAME_QUEUE_SIZE,
                    "fullness_percent": frame_queue.get_fullness_percent() if 'frame_queue' in globals() and hasattr(frame_queue, 'get_fullness_percent') else 0,
                    "dropped_frames": frame_queue.get_dropped_frames() if 'frame_queue' in globals() and hasattr(frame_queue, 'get_dropped_frames') else 0
                },
                "usb_frame_queue": {
                    "size": usb_frame_queue.qsize() if 'usb_frame_queue' in globals() else 0,
                    "max_size": FRAME_QUEUE_SIZE,
                    "fullness_percent": (usb_frame_queue.qsize() / FRAME_QUEUE_SIZE * 100) if 'usb_frame_queue' in globals() else 0
                }
            },
            "detection": {
                "active": detection_active if 'detection_active' in globals() else False,
                "camera": detection_camera if 'detection_camera' in globals() else "unknown",
                "email_armed": email_armed if 'email_armed' in globals() else False,
                "cooldown_active": cooldown_active if 'cooldown_active' in globals() else False
            },
            "yolo": {
                "enabled": yolo_processor is not None if 'yolo_processor' in globals() else False,
                "model": config.get('yolo_model_path', 'unknown') if 'config' in globals() else 'unknown'
            },
            "reconnections": {
                "pi_camera": pi_camera_reconnector.get_stats() if 'pi_camera_reconnector' in globals() else {"status": "not_initialized"},
            },
            "system": {
                "uptime_seconds": time.time() - fps_start_time if 'fps_start_time' in globals() else 0,
                "fps": fps_frame_count / (time.time() - fps_start_time) if 'fps_start_time' in globals() and (time.time() - fps_start_time) > 0 else 0
            }
        }

        return jsonify(health_data), 200

    except Exception as e:
        logging.error(f"Health endpoint error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dashboard</title>
            <style>
                body {font-family: 'Arial', sans-serif; background-color: #f0f2f5; text-align: center; padding-top: 50px;}
                h1 {color: #1877f2; margin-bottom: 20px;}
                .button-container {display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;}
                a {background-color: #1877f2; color: white; padding: 10px 20px; border-radius: 4px; text-decoration: none;}
                a:hover {background-color: #166fe5;}
            </style>
        </head>
        <body>
            <h1>Welcome to the Dashboard</h1>
            <div class="button-container">
                <a href="{{ url_for('video_feed') }}">Live Feed</a>
                <a href="{{ url_for('stored_media') }}">Stored Media</a>
            </div>
        </body>
        </html>
    ''')

@app.route('/document', methods=['GET', 'POST'])
@login_required
def document():
    doc_file = "user_document.txt"
    if request.method == 'POST' and 'content' in request.form:
        content = request.form['content']
        try:
            with open(doc_file, 'w') as f:
                f.write(content)
            flash('Document saved successfully!')
            logging.info("User document saved")
        except Exception as e:
            flash(f"Error saving document: {e}")
            logging.error(f"Error saving document: {e}")
    content = ""
    if os.path.exists(doc_file):
        try:
            with open(doc_file, 'r') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Error reading document: {e}")
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document</title>
            <style>
                body {font-family: 'Arial', sans-serif; background-color: #f0f2f5; padding: 50px; text-align: center;}
                h1 {color: #1877f2; margin-bottom: 20px;}
                textarea {width: 80%; height: 400px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; resize: vertical;}
                button {background-color: #1877f2; color: white; padding: 10px 20px; border-radius: 4px; border: none; cursor: pointer; margin-top: 10px;}
                button:hover {background-color: #166fe5;}
                .flash {background-color: #d4edda; color: #155724; padding: 10px; margin: 10px 0; border-radius: 4px;}
            </style>
        </head>
        <body>
            <h1>Edit Document</h1>
            {% for message in get_flashed_messages() %}
                <div class="flash">{{ message }}</div>
            {% endfor %}
            <form method="POST">
                <textarea name="content">{{ content }}</textarea>
                <br>
                <button type="submit">Save</button>
            </form>
        </body>
        </html>
    ''', content=content)

@app.route('/stored_media')
@login_required
def stored_media():
    """PYTHON 3.14: Lock-free media listing - filesystem operations are atomic"""
    media_files = []

    # PYTHON 3.14: No media_lock needed - filesystem operations are atomic
    if os.path.exists("stored_media"):
        for filename in os.listdir("stored_media"):
            if filename.endswith(("_image.jpg", "_video.mp4")):
                timestamp = filename.split("_")[0] + "_" + filename.split("_")[1]
                is_video = filename.endswith("_video.mp4")
                thumb_filename = f"{timestamp}_video_thumb.jpg" if is_video else f"{timestamp}_image_thumb.jpg"
                media_files.append((filename, timestamp, thumb_filename, is_video))

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stored Media</title>
            <style>
                body {font-family: 'Arial', sans-serif; background-color: #f0f2f5; padding: 20px;}
                h1 {color: #1877f2; text-align: center;}
                .media-container {display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;}
                .media-item {text-align: center; position: relative;}
                img {width: 150px; height: 150px; object-fit: cover; border: 1px solid #ddd; border-radius: 4px;}
                .video-indicator {position: absolute; top: 5px; right: 5px; background-color: rgba(0,0,0,0.6); color: white; 
                                 padding: 2px 5px; border-radius: 3px; font-size: 12px;}
                p {margin: 5px 0;}
                button {background-color: #1877f2; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-top: 10px; margin-bottom: 20px;}
                button:hover {background-color: #166fe5;}
                .back-btn {display: inline-block; margin-top: 20px; color: #1877f2; text-decoration: none;}
                .back-btn:hover {text-decoration: underline;}
            </style>
        </head>
        <body>
            <h1>Stored Media</h1>
            <div style="text-align: center;">
                <form action="{{ url_for('purge_media') }}" method="POST" style="display: inline-block;">
                    <button type="submit">Purge All Media</button>
                </form>
                <br>
                <a href="{{ url_for('dashboard') }}" class="back-btn">Back to Dashboard</a>
            </div>
            <div class="media-container">
                {% if media_files %}
                    {% for filename, timestamp, thumb_filename, is_video in media_files %}
                        <div class="media-item">
                            <a href="{{ url_for('serve_media', filename=filename) }}" target="_blank">
                                <img src="{{ url_for('serve_media', filename=thumb_filename) }}" alt="{{ filename }}">
                                {% if is_video %}
                                <span class="video-indicator">VIDEO</span>
                                {% endif %}
                            </a>
                            <p>{{ timestamp }}</p>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No media stored yet.</p>
                {% endif %}
            </div>
        </body>
        </html>
    ''', media_files=media_files)

@app.route('/media/<path:filename>')
@login_required
def serve_media(filename):
    file_path = os.path.join("stored_media", filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    if filename.endswith('.mp4'):
        return send_file(file_path, mimetype='video/mp4', as_attachment=False)
    return send_file(file_path, as_attachment=False)

@app.route('/purge_media', methods=['POST'])
@login_required
def purge_media():
    """PYTHON 3.14: Lock-free media purging - filesystem operations are atomic"""
    # PYTHON 3.14: No media_lock needed - filesystem operations are atomic
    if os.path.exists("stored_media"):
        shutil.rmtree("stored_media")
        os.makedirs("stored_media", exist_ok=True)
        logging.info("All stored media purged")
    return redirect(url_for('stored_media'))

# Motion Detection and Email
def detect_objects_with_yolo(frame):
    """
    Detect objects in frame using YOLO model
    Returns: (detections_list, annotated_frame, contains_objects_of_interest)
    """
    if yolo_model is None or not config.get('use_yolo_detection', False):
        # Return defaults indicating no YOLO processing occurred
        return [], frame, False
    
    try:
        # Run YOLO detection on frame with a lower confidence threshold
        confidence_threshold = config.get('yolo_confidence_threshold', 0.25)  # Lowered default threshold
        classes_of_interest = config.get('yolo_classes_of_interest', list(range(80)))

        # DIAGNOSTIC: Log YOLO configuration
        logging.debug(f"YOLO running with conf_threshold={confidence_threshold}, classes_of_interest={classes_of_interest}")

        results = yolo_model(frame, conf=confidence_threshold, verbose=False)

        # Process results
        detections = []
        contains_objects_of_interest = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = yolo_model.names[cls]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'class': cls,
                    'class_name': cls_name,
                    'confidence': conf,
                    'box': (x1, y1, x2, y2)
                })

                # DIAGNOSTIC: Log every detection to understand what YOLO sees
                is_of_interest = cls in classes_of_interest
                logging.info(f"YOLO detected: class={cls} ({cls_name}), conf={conf:.2f}, box=({x1},{y1},{x2},{y2}), of_interest={is_of_interest}")

                if cls in classes_of_interest:
                    contains_objects_of_interest = True
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        return detections, annotated_frame, contains_objects_of_interest
    
    except Exception as e:
        logging.error(f"Error in YOLO detection: {str(e)}")
        return [], frame, False

# PYTHON 3.14: OLD YOLO WORKER - REPLACED WITH ParallelYOLOProcessor
# This function is no longer used - kept for reference only
"""
def yolo_worker():
    '''
    Async YOLO worker thread that processes frames from the request queue
    and puts results into the response queue without blocking the main loop.

    DEPRECATED: Replaced with ParallelYOLOProcessor using ThreadPoolExecutor
    '''
    global yolo_worker_running, yolo_model

    logging.info("YOLO worker thread started")

    while yolo_worker_running:
        try:
            # Try to get a frame request with timeout to allow checking the running flag
            try:
                request = yolo_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Unpack request
            frame, camera, request_time = request

            # Skip if frame is invalid
            if frame is None or frame.size == 0:
                logging.warning("YOLO worker received invalid frame")
                continue

            # Process with YOLO
            start_time = time.time()
            detections, annotated_frame, contains_objects = detect_objects_with_yolo(frame)
            processing_time = time.time() - start_time

            logging.debug(f"YOLO processing took {processing_time:.2f}s for {camera}")

            # Put result in response queue (non-blocking, drop if full)
            try:
                yolo_response_queue.put_nowait({
                    'camera': camera,
                    'annotated_frame': annotated_frame,
                    'contains_objects': contains_objects,
                    'request_time': request_time,
                    'processing_time': processing_time
                })
            except queue.Full:
                logging.warning("YOLO response queue full, dropping result")

        except Exception as e:
            logging.error(f"Error in YOLO worker thread: {e}")
            time.sleep(0.1)  # Prevent tight error loop

    logging.info("YOLO worker thread stopped")
"""

def process_frame(frame, bg_subtractor, previous_frame, camera, use_bg_subtraction=True):
    global detection_active, detection_camera, state_manager, recording_manager
    global knn_frame_counter_cam1, knn_frame_counter_cam2

    # Log the current state occasionally for debugging
    if random.random() < 0.001:  # Log approx. 0.1% of frames to avoid log spam
        logging.debug(f"Processing frame for {camera}, detection_camera={detection_camera}, method={'BG Subtraction' if use_bg_subtraction else 'Frame Diff'}")
    
    # Skip processing if not active, streaming, or in YOLO/recording states
    yolo_or_recording = state_manager.state in [DetectionState.YOLO_ROI, DetectionState.YOLO_FULL, DetectionState.RECORDING]
    if not detection_active or streaming_active or yolo_or_recording:
        return_prev = previous_frame if previous_frame is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, False, return_prev
        
    # Check if the detection is enabled for this camera
    is_detection_enabled = False
    if detection_camera == "both":
        is_detection_enabled = True
    elif detection_camera == camera:
        is_detection_enabled = True
    elif detection_camera == "disable":
        is_detection_enabled = False
    
    frame_with_text = frame.copy()
    cam_text = "Camera 2 (USB)" if camera == "cam2" else "Camera 1 (Pi)"
    
    if detection_camera == "both":
        camera_status = "ACTIVE (BOTH)"
        detection_mode = "Mode: both cameras"
    else:
        camera_status = "ACTIVE" if is_detection_enabled else "INACTIVE"
        detection_mode = f"Mode: {detection_camera}"
        
    if not is_detection_enabled:
        # Return previous_frame for frame diff, or None if using BG subtraction
        return_prev = previous_frame if previous_frame is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame_with_text, False, return_prev

    try:
        # Use camera-specific ROI coordinates
        roi_key = "roi_coordinates_cam2" if camera == "cam2" else "roi_coordinates"
        
        # Apply ROI using the correct coordinates for this camera
        roi = config.get(roi_key, [0, 0, frame.shape[1], frame.shape[0]])
        if isinstance(roi, list) and len(roi) == 4:
            x, y, w, h = roi
        else:
            x = roi.get("x", 0); y = roi.get("y", 0)
            w = roi.get("width", frame.shape[1]); h = roi.get("height", frame.shape[0])
        
        # Ensure ROI coordinates are within frame bounds
        x = max(0, min(x, frame.shape[1] - 1)); y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x); h = min(h, frame.shape[0] - y)
        
        # Check if ROI is valid
        if w <= 0 or h <= 0:
            logging.warning(f"Invalid ROI dimensions for {camera}, using full frame")
            roi_frame = frame
        else:
            roi_frame = frame[y:y+h, x:x+w]
        
        if roi_frame.size == 0:
            logging.warning(f"Empty ROI frame for {camera}, using full frame")
            roi_frame = frame
            
        # Create a visible indication of the ROI on the original frame
        frame_with_roi = frame_with_text.copy()
        cv2.rectangle(frame_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw ROI outline in green
        
        # Process only within ROI
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        thresh = None # Initialize thresh variable
        updated_previous_frame = previous_frame # Keep track of previous frame if not using BG subtraction

        if use_bg_subtraction:
            # Apply background subtractor with adaptive learning rate for warmup period
            if bg_subtractor is not None:
                # Determine which frame counter to use and increment it
                if camera == "cam1":
                    knn_frame_counter_cam1 += 1
                    frame_count = knn_frame_counter_cam1
                else:  # cam2
                    knn_frame_counter_cam2 += 1
                    frame_count = knn_frame_counter_cam2

                # KNN reset strategy for cam2: Reset the model after camera resolution stabilizes
                # This prevents the false positive caused by KNN training on transitional frames
                CAMERA_STABILIZATION_FRAME = 200  # Frame where camera resolution stabilizes
                WARMUP_FRAMES = 150
                SKIP_DETECTION_FRAMES = 370  # Skip detection until second warmup completes (200 + 150 + 20 buffer)

                # For cam2, reset KNN at frame 200 to retrain on stable resolution frames
                if camera == "cam2" and frame_count == CAMERA_STABILIZATION_FRAME:
                    bg_subtractor.clear()  # Clear the KNN background model
                    logging.info(f"Cam2: Resetting KNN background model at frame {CAMERA_STABILIZATION_FRAME} (camera stabilized)")

                # Determine learning rate based on frame count
                # For cam2: frames 1-199 (initial warmup on unstable frames, will be discarded)
                #           frame 200 (reset)
                #           frames 201-350 (second warmup on stable frames)
                #           frames 351+ (stable mode)
                if camera == "cam2":
                    if frame_count < CAMERA_STABILIZATION_FRAME:
                        learning_rate = 0.005  # Initial warmup (will be reset)
                        if frame_count == 1:
                            logging.info(f"KNN initial warmup for cam2 (will reset at frame {CAMERA_STABILIZATION_FRAME})")
                    elif frame_count < CAMERA_STABILIZATION_FRAME + WARMUP_FRAMES:
                        learning_rate = 0.01  # Second warmup after reset - faster learning on stable frames
                        if frame_count == CAMERA_STABILIZATION_FRAME + 1:
                            logging.info(f"KNN second warmup started for cam2 (learning_rate=0.01 for {WARMUP_FRAMES} frames)")
                        elif frame_count == CAMERA_STABILIZATION_FRAME + WARMUP_FRAMES:
                            logging.info(f"KNN second warmup completed for cam2, switching to stable mode")
                    else:
                        learning_rate = 0.001  # Stable mode
                else:
                    # Cam1 uses original logic
                    if frame_count <= WARMUP_FRAMES:
                        learning_rate = 0.005
                        if frame_count == 1:
                            logging.info(f"KNN warmup started for {camera}")
                        elif frame_count == WARMUP_FRAMES:
                            logging.info(f"KNN warmup completed for {camera}")
                    else:
                        learning_rate = 0.001

                fgMask = bg_subtractor.apply(gray, learningRate=learning_rate)

                # Apply larger morphological operations to filter rain and noise
                kernel = np.ones((5,5), np.uint8)  # Larger kernel for better noise removal
                fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)  # Remove small noise
                fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)  # Fill holes in motion regions
                thresh = fgMask # Use the mask directly as the thresholded image

                # Skip motion detection during warmup frames
                if frame_count <= SKIP_DETECTION_FRAMES:
                    # Return no motion detected during warmup to let background model stabilize
                    return frame_with_roi, False, None
            else:
                logging.warning(f"Background subtractor for {camera} is None, cannot process.")
                return frame_with_roi, False, None # Return None for previous_frame

        else:
            # Use original frame differencing method
            # Better handling of previous frame for consistent comparisons
            if previous_frame is None or previous_frame.shape != gray.shape:
                # If first frame or shape mismatch, just initialize previous frame
                updated_previous_frame = gray.copy()
                # For USB camera, log when creating new reference frame
                if camera == "cam2":
                    logging.debug(f"{camera}: Initializing new previous frame reference (Frame Diff)")
                return frame_with_roi, False, updated_previous_frame
            else:
                # Consistent frame comparison logic
                # Add stabilization for Camera 2 to reduce false positives
                if camera == "cam2":
                    # Apply lighter blur for USB camera to preserve edge details (reduced from 25x25 to 7x7)
                    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                    prev_blurred = cv2.GaussianBlur(previous_frame, (7, 7), 0)

                    # No brightness adjustment for low-contrast detection (changed alpha from 0.9 to 1.0)
                    gray_adjusted = cv2.convertScaleAbs(gray_blurred, alpha=1.0, beta=0)
                    prev_adjusted = cv2.convertScaleAbs(prev_blurred, alpha=1.0, beta=0)

                    frameDelta = cv2.absdiff(prev_adjusted, gray_adjusted)
                else:
                    frameDelta = cv2.absdiff(previous_frame, gray) # Original logic for cam1

                # Apply camera-specific threshold settings
                background_threshold = config.get("background_threshold", 25)
                threshold_multiplier = 1.0 # Default
                if camera == "cam2":
                    # Use the configurable multiplier from settings for Camera 2
                    threshold_multiplier = config.get("cam2_threshold_multiplier", 1.5)
                else:
                    # Use the configurable multiplier from settings for Camera 1
                    threshold_multiplier = config.get("cam1_threshold_multiplier", 1.0) # Use cam1 setting

                effective_threshold = background_threshold * threshold_multiplier
                # Apply thresholding
                _, thresh = cv2.threshold(frameDelta, effective_threshold, 255, cv2.THRESH_BINARY)

                # Disabled morphological opening for low-contrast detection
                # if camera == "cam2":
                #     # Apply morphological operations to remove small noise
                #     kernel = np.ones((3, 3), np.uint8)
                #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                thresh = cv2.dilate(thresh, None, iterations=1)  # Reduced from 2 to 1 for better edge preservation
                
                # Update previous frame for next iteration ONLY when using frame differencing
                updated_previous_frame = gray.copy()

        # --- Common Contour Detection Logic (using 'thresh' from either method) ---
        if thresh is None:
             logging.warning(f"Threshold image ('thresh') is None for camera {camera}. Skipping contour detection.")
             return frame_with_roi, False, updated_previous_frame # Return potentially updated previous_frame

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, getattr(cv2, 'CHAIN_APPROX_SIMPLE', 1))
        
        # Apply camera-specific area thresholds
        min_area_threshold = MIN_CONTOUR_AREA
        area_multiplier = 1.0 # Default

        if camera == "cam2":
            # Use the configurable area multiplier from settings for Camera 2
            area_multiplier = config.get("cam2_area_multiplier", 4.0)
        else:
            # Use the configurable area multiplier from settings for Camera 1
            area_multiplier = config.get("cam1_area_multiplier", 1.0)

        min_area_threshold *= area_multiplier
            
        # Get configurable minimum contours settings for both cameras
        cam1_min_contours = config.get("cam1_min_contours", 0) if camera != "cam2" else 0
        cam2_min_contours = config.get("cam2_min_contours", 0) if camera == "cam2" else 0
            
        motion_detected = False
        contour_areas = []  # Track areas for debugging
        significant_contour_count = 0

        for c in contours:
            area = cv2.contourArea(c)
            contour_areas.append(area)
            if area > min_area_threshold:
                significant_contour_count += 1
                motion_detected = True
                
        # Apply minimum contour check if needed
        if motion_detected:
             min_contours_required = cam2_min_contours if camera == "cam2" else cam1_min_contours
             if min_contours_required > 0 and significant_contour_count <= min_contours_required:
                 motion_detected = False # Override: not enough significant contours
                 # Reduce log spam for this message
                 log_level = logging.DEBUG if random.random() < 0.1 else logging.INFO # Log less often
                 logging.log(log_level, f"{camera}: Motion rejected - only {significant_contour_count} significant contours (area > {min_area_threshold:.1f}), minimum required: {min_contours_required}")

        # Apply motion segmentation overlay if motion is confirmed and YOLO is not active
        yolo_active = state_manager.state in [DetectionState.YOLO_ROI, DetectionState.YOLO_FULL]
        if motion_detected and not yolo_active and significant_contour_count > 0:
            # Use the threshold mask directly for segmentation overlay
            # Convert grayscale thresh to BGR for overlay
            thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            # Create red mask where motion was detected
            red_overlay = np.zeros_like(frame_with_roi)
            red_overlay[y:y+h, x:x+w][thresh > 0] = [0, 0, 255]  # Red color for motion pixels
            # Blend with original frame (30% opacity for motion overlay)
            cv2.addWeighted(red_overlay, 0.3, frame_with_roi, 1.0, 0, frame_with_roi)

        # Log detailed info about motion detection on occasion
        if motion_detected or random.random() < (0.01 if camera == "cam2" else 0.005): # Log less frequently
            areas_str = ", ".join([f"{a:.1f}" for a in sorted(contour_areas, reverse=True)[:5]]) if contour_areas else "none"
            log_level = logging.DEBUG if random.random() < 0.1 else logging.INFO # Reduce log spam
            logging.log(log_level, f"{camera}: Motion {'DETECTED' if motion_detected else 'not detected'}, contours: {len(contours)}, sig_contours: {significant_contour_count}, top areas: {areas_str}, area_thresh: {min_area_threshold:.1f}, method: {'BG Sub' if use_bg_subtraction else 'Frame Diff'}")
            
        # YOLO processing removed from here - now handled asynchronously in main loop
        # This eliminates duplicate YOLO calls and prevents freezing

        # Return the frame with motion boxes and detection status
        # YOLO detection will be handled separately in the main loop
        return frame_with_roi, motion_detected, updated_previous_frame
        
    except Exception as e:
        logging.error(f"Error in process_frame for {camera}: {e}")
        # Return potentially updated previous_frame
        return frame_with_text, False, updated_previous_frame if 'updated_previous_frame' in locals() else previous_frame

def create_gif_from_video(video_path, output_gif_path, camera, max_frames=60, fps=10):
    """Create GIF from video with rotation based on camera source"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        logging.error(f"Video {video_path} has no frames")
        cap.release()
        raise ValueError("No frames in video")
    
    step = max(1, total_frames // max_frames) if total_frames > max_frames else 1
    target_frames = min(total_frames, max_frames)

    logging.debug(f"Creating GIF: total_frames={total_frames}, step={step}, target_frames={target_frames}, source={camera}")

    # Use the appropriate rotation value based on camera source
    if camera == "cam2":
        with cam2_rotation.get_lock():
            rotation = cam2_rotation.value
        logging.debug(f"Using Camera 2 rotation for GIF: {rotation}deg")
    else:
        with email_rotation_cam1.get_lock():
            rotation = email_rotation_cam1.value
            logging.debug(f"Using Camera 1 rotation for GIF: {rotation}deg")

    while cap.isOpened() and len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            frame_rotated = apply_rotation(frame, rotation)
            frame_rgb = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (320, 240))
            frames.append(frame_resized)
        frame_count += 1
    
    cap.release()

    if frames:
        imageio.mimsave(output_gif_path, frames, fps=fps)
        logging.debug(f"GIF created: {output_gif_path}, {len(frames)} frames at {fps} FPS with rotation {rotation}deg")
    else:
        logging.error(f"No frames extracted from {video_path}")
        raise ValueError("Failed to create GIF")

def send_media_via_gmail(image, video_path, sender_email, receiver_emails, password, camera):
    """Send email with media (camera passed as parameter for thread safety)"""
    global last_email_time, cooldown_active
    live_feed_url = config['live_feed_url']
    current_time = time.time()
    if current_time - last_email_time < EMAIL_COOLDOWN:
        logging.debug("Email sending skipped due to cooldown")
        return

    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return

    # Log which camera triggered this email
    camera_source = "Camera 2" if camera == "cam2" else "Camera 1"
    logging.info(f"Creating email media with source: {camera_source}")

    gif_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    try:
        create_gif_from_video(video_path, gif_temp.name, camera)
    except Exception as e:
        logging.error(f"Failed to create GIF: {type(e).__name__}: {e}")
        os.unlink(gif_temp.name)
        return

    msg = MIMEMultipart('alternative')
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)
    msg['Subject'] = f"Motion Detected ({camera_source})"

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    html_body = f"""
    <html>
    <body>
        <p>Motion detected by {camera_source} at {timestamp}</p>
        <p>View the live feed here: <a href="{live_feed_url}">Live Feed</a></p>
        <p>Preview:</p>
        <img src="cid:gif_preview" alt="Motion GIF" style="width: 320px; height: 240px;">
        <p>Still Image:</p>
        <img src="cid:image_preview" alt="Motion Still" style="width: 320px; height: 240px;">
    </body>
    </html>
    """
    msg.attach(MIMEText(html_body, 'html'))

    with open(gif_temp.name, 'rb') as f:
        gif_part = MIMEImage(f.read(), _subtype='gif')
        gif_part.add_header('Content-ID', '<gif_preview>')
        gif_part.add_header('Content-Disposition', 'inline', filename="motion.gif")
        msg.attach(gif_part)

    # Apply appropriate rotation based on camera source
    if camera == "cam2":
        with cam2_rotation.get_lock():
            rotation = cam2_rotation.value
        logging.debug(f"Using Camera 2 rotation for email image: {rotation}deg")
    else:
        with email_rotation_cam1.get_lock():
            rotation = email_rotation_cam1.value
            logging.debug(f"Using Camera 1 rotation for email image: {rotation}deg")
    
    rotated_image = apply_rotation(image, rotation)
    resized_image = cv2.resize(rotated_image, (320, 240), interpolation=cv2.INTER_AREA)
    _, img_encoded = cv2.imencode('.jpg', resized_image)
    img_part = MIMEImage(img_encoded.tobytes(), _subtype='jpeg')
    img_part.add_header('Content-ID', '<image_preview>')
    img_part.add_header('Content-Disposition', 'inline', filename="motion.jpg")
    msg.attach(img_part)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
        last_email_time = current_time
        cooldown_active = True
        threading.Timer(EMAIL_COOLDOWN, lambda: globals().update(cooldown_active=False)).start()
        logging.info("Email sent with embedded GIF, still image, and live feed link")
    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication failed. Check sender_email and password.")
    except smtplib.SMTPException as e:
        logging.error(f"SMTP error: {e}")
    except Exception as e:
        logging.error(f"Failed to send email: {type(e).__name__}: {e}")
    finally:
        os.unlink(gif_temp.name)

def send_email_in_thread(image, video_path, sender, receivers, password, camera):
    """Send email with media in a separate thread (camera passed for thread safety)"""
    global last_email_time, cooldown_active
    try:
        # Additional logging at start of email thread
        camera_source = "Camera 2" if camera == "cam2" else "Camera 1"
        logging.info(f"Email thread starting for {camera_source}, image type: {type(image)}")
        
        # Extra verification for Camera 2 emails
        if camera == "cam2":
            logging.info(f"Camera 2 email verification: video exists={os.path.exists(video_path)}, " +
                        f"size={os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")
        
        if image is None:
            logging.error("Cannot send email: image is None")
            return
        
        if not os.path.exists(video_path):
            logging.error(f"Cannot send email: video path does not exist: {video_path}")
            return
            
        # Save media to storage first
        save_media_to_storage(image, video_path, camera)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        stored_video_path = f"stored_media/{timestamp}_video.mp4"
        
        # Add a delay to ensure video file is fully written
        logging.info("Waiting briefly to ensure video file is fully written...")
        time.sleep(1)
            
        # Verify stored video file exists and has content
        if os.path.exists(stored_video_path) and os.path.getsize(stored_video_path) > 100:
            send_media_via_gmail(image, stored_video_path, sender, receivers, password, camera)
        else:
            logging.error(f"Stored video file {stored_video_path} not found or too small, cannot send email")
    except Exception as e:
        logging.error(f"Error in email thread: {type(e).__name__}: {e}")

def toggle_visibility(event, x, y, flags, param):
    """Mouse callback that shows/hides the toggle button interface"""
    global button_visible
    toggle_dialog = param
    
    if toggle_dialog is None:
        logging.error("toggle_visibility called with None dialog parameter!")
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        logging.info(f"Mouse clicked at ({x}, {y})")
        if button_visible:
            logging.info("Button was visible - hiding toggle dialog")
            toggle_dialog.hide()
        else:
            logging.info("Button was not visible - showing toggle dialog")
            toggle_dialog.show()
        toggle_dialog.last_click_time = time.time()

# ToggleDialog
class ToggleDialog(tk.Toplevel):
    def __init__(self, root):
        super().__init__(root)
        self.root = root

        # Remove all window decorations
        self.overrideredirect(True)
        self.geometry("120x40")
        self.attributes('-topmost', True)
        self.withdraw()

        # Just a single button
        self.toggle_button = ttk.Button(self, text="Armed" if email_armed else "Disarmed", command=self.toggle_state)
        self.toggle_button.pack(fill="both", expand=True)

        self.last_click_time = time.time()
        self.timer_thread = threading.Thread(target=self.auto_hide, daemon=True)
        self.timer_thread.start()
        logging.info("ToggleDialog initialized")

    def toggle_state(self):
        global email_armed, cooldown_active, detection_active
        email_armed = not email_armed
        detection_active = email_armed
        if email_armed and cooldown_active:
            cooldown_active = False
            logging.info("Cooldown overridden by manual arming")
        self.toggle_button.config(text="Armed" if email_armed else "Disarmed")
        self.last_click_time = time.time()
        logging.info(f"Detection emails {'armed' if email_armed else 'disarmed'}")
        logging.info(f"Detection active also set to: {detection_active}")

    def show(self):
        if self.winfo_exists():
            self.deiconify()
            self.lift()
            global button_visible
            button_visible = True
            self.last_click_time = time.time()

    def hide(self):
        if self.winfo_exists():
            self.withdraw()
            global button_visible
            button_visible = False

    def auto_hide(self):
        while True:
            try:
                if button_visible and time.time() - self.last_click_time >= 10:
                    # Use after() to schedule hide in the main thread
                    self.root.after(0, self.hide)
            except Exception as e:
                logging.error(f"Error in auto_hide: {e}")
            time.sleep(1)

    def on_close(self):
        if self.winfo_exists():
            self.hide()

    def update_status(self):
        """Update the toggle dialog to reflect current camera and detection status"""
        global email_armed, cooldown_active, detection_camera

        # Check if Camera 2 is available
        with cam2_available.get_lock():
            cam2_is_available = cam2_available.value

        # If Camera 2 is not available and detection is set to cam2 or both,
        # reset to cam1
        if not cam2_is_available and detection_camera in ['cam2', 'both']:
            detection_camera = 'cam1'
            logging.info("Reset detection camera to 'cam1' in ToggleDialog since Camera 2 is unavailable")

            # Update global config to match
            config['detection_camera'] = 'cam1'
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=4)
                logging.info("Updated config file with new detection camera setting")
            except Exception as e:
                logging.error(f"Failed to update config file: {e}")

        # Update button text
        self.toggle_button.config(text="Armed" if email_armed else "Disarmed")

# SettingsDialog
class SettingsDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Update Settings")
        self.geometry("500x900")  # Increased width and height to fit sliders and ensure scrolling works
        self.resizable(True, True)  # Make it resizable so users can adjust if needed
        self.attributes('-topmost', True)
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        logging.info("SettingsDialog initialized")
        logging.debug(f"Config in SettingsDialog init: {type(config)}, {config}")

        # Add tracking variables for ROI selection sequence
        self.roi_sequence_both = False
        self.original_detection_camera = None
        
        self.local_ip = get_local_ip()
        self.live_feed_url = tk.StringVar(value=config.get('live_feed_url', ''))
        self.flask_host = tk.StringVar(value=config.get('flask_host', ''))
        self.flask_port = tk.StringVar(value=str(config.get('flask_port', 5000)))
        self.sender_email = tk.StringVar(value=config.get('sender_email', ''))
        self.receiver_emails = [tk.StringVar(value=email) for email in config.get('receiver_emails', [])]
        self.email_password = tk.StringVar(value=config.get('email_password', ''))
        self.users = [(tk.StringVar(value=user['email']), tk.StringVar(value="********")) for user in config.get('users', [])]
        self.detection_camera = tk.StringVar(value=config.get('detection_camera', 'cam1'))
        self.email_rotation_label = tk.StringVar(value=f"Email Rotation: {email_rotation_cam1.value}deg")
        
        # Timer schedule settings
        self.timer_enabled = tk.BooleanVar(value=config.get('timer_enabled', False))
        self.schedule_arm_time = tk.StringVar(value=config.get('schedule_arm_time', '08:00'))
        self.schedule_disarm_time = tk.StringVar(value=config.get('schedule_disarm_time', '18:00'))
        
        # Background threshold and value display variable
        self.background_threshold = tk.IntVar(value=config.get("background_threshold", 25))
        self.background_threshold_value = tk.StringVar(value=str(self.background_threshold.get()))
        
        # Camera resolution settings
        self.cam1_resolution = tk.StringVar(value=config.get("cam1_resolution", "640x480"))
        self.cam2_resolution = tk.StringVar(value=config.get("cam2_resolution", "320x240"))
        
        # Camera 1 settings with value display variables
        self.cam1_area_multiplier = tk.DoubleVar(value=config.get("cam1_area_multiplier", 1.0))
        self.cam1_area_multiplier_value = tk.StringVar(value=f"{self.cam1_area_multiplier.get():.1f}x")
        
        self.cam1_threshold_multiplier = tk.DoubleVar(value=config.get("cam1_threshold_multiplier", 1.0))
        self.cam1_threshold_multiplier_value = tk.StringVar(value=f"{self.cam1_threshold_multiplier.get():.1f}x")
        
        self.cam1_min_contours = tk.IntVar(value=config.get("cam1_min_contours", 0))
        self.cam1_min_contours_value = tk.StringVar(value=str(self.cam1_min_contours.get()))
        
        # Camera 2 settings with value display variables
        self.cam2_area_multiplier = tk.DoubleVar(value=config.get("cam2_area_multiplier", 4.0))
        self.cam2_area_multiplier_value = tk.StringVar(value=f"{self.cam2_area_multiplier.get():.1f}x")
        
        self.cam2_threshold_multiplier = tk.DoubleVar(value=config.get("cam2_threshold_multiplier", 1.5))
        self.cam2_threshold_multiplier_value = tk.StringVar(value=f"{self.cam2_threshold_multiplier.get():.1f}x")
        
        self.cam2_min_contours = tk.IntVar(value=config.get("cam2_min_contours", 0))
        self.cam2_min_contours_value = tk.StringVar(value=str(self.cam2_min_contours.get()))

        # Store parent for accessing global variables
        self.parent = parent

        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add canvas and scrollbar to the main frame
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(scrollable_frame, text="Live Feed URL (for email):").pack(pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.live_feed_url, width=40).pack()
        ttk.Button(scrollable_frame, text=f"Set to Local IP ({self.local_ip})", command=self.set_live_feed_to_local).pack(pady=2)

        ttk.Label(scrollable_frame, text="Web UI Host (IP Address):").pack(pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.flask_host, width=40).pack()
        ttk.Button(scrollable_frame, text=f"Set to Local IP ({self.local_ip})", command=self.set_flask_host_to_local).pack(pady=2)

        ttk.Label(scrollable_frame, text="Web UI Port:").pack(pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.flask_port, width=40).pack()

        ttk.Label(scrollable_frame, text="Sender Email:").pack(pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.sender_email, width=40).pack()

        ttk.Label(scrollable_frame, text="Receiver Emails:").pack(pady=5)
        self.receiver_frame = ttk.Frame(scrollable_frame)
        self.receiver_frame.pack(fill="both", expand=True)
        self.update_receiver_fields()

        ttk.Button(scrollable_frame, text="Add Receiver Email", command=self.add_receiver).pack(pady=5)

        ttk.Label(scrollable_frame, text="Email App Password:").pack(pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.email_password, width=40, show="*").pack()

        ttk.Label(scrollable_frame, text="Users (leave password as '********' to keep existing):").pack(pady=5)
        self.user_frame = ttk.Frame(scrollable_frame)
        self.user_frame.pack(fill="both", expand=True)
        self.update_user_fields()

        ttk.Button(scrollable_frame, text="Add User", command=self.add_user).pack(pady=5)

        ttk.Label(scrollable_frame, text="Run Detection On:").pack(pady=5)
        detection_combo = ttk.Combobox(scrollable_frame, textvariable=self.detection_camera, values=["cam1", "cam2", "both", "disable"], state="readonly")
        detection_combo.pack()
        # Add a trace to immediately update the global variable when the dropdown changes
        self.detection_camera.trace_add("write", self.on_detection_camera_change)
        
        # Timer settings section
        ttk.Label(scrollable_frame, text="Detection Timer Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        timer_frame = ttk.Frame(scrollable_frame)
        timer_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Checkbutton(timer_frame, text="Enable Scheduled Arm/Disarm", variable=self.timer_enabled).grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        
        ttk.Label(timer_frame, text="Auto Arm Time (HH:MM):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(timer_frame, textvariable=self.schedule_arm_time, width=10).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(timer_frame, text="(e.g., 08:00 for 8:00 AM)").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(timer_frame, text="Auto Disarm Time (HH:MM):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(timer_frame, textvariable=self.schedule_disarm_time, width=10).grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(timer_frame, text="(e.g., 18:00 for 6:00 PM)").grid(row=2, column=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(timer_frame, text="Note: The Arm/Disarm button will override the timer schedule.").grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        # Common Motion Detection Settings
        ttk.Label(scrollable_frame, text="Motion Detection Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Background threshold slider (common setting)
        motion_frame = ttk.Frame(scrollable_frame)
        motion_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(motion_frame, text="Base Background Threshold:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(motion_frame, textvariable=self.background_threshold_value).grid(row=0, column=2, padx=5)
        bg_scale = ttk.Scale(motion_frame, from_=5, to=50, orient="horizontal", 
                              variable=self.background_threshold, length=200,
                              command=lambda v: self.background_threshold_value.set(str(int(float(v)))))
        bg_scale.grid(row=0, column=1, padx=5)
        ttk.Label(motion_frame, text="(Higher values require more change to trigger motion detection)").grid(row=1, column=0, columnspan=3, padx=5)

        # Camera 1 specific settings
        ttk.Label(scrollable_frame, text="Camera 1 Settings (Pi Camera)", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Camera 1 resolution dropdown
        cam1_res_frame = ttk.Frame(scrollable_frame)
        cam1_res_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cam1_res_frame, text="Resolution:").grid(row=0, column=0, sticky="w", padx=5)
        cam1_res_combo = ttk.Combobox(cam1_res_frame, textvariable=self.cam1_resolution, 
                                     values=["320x240", "640x480", "800x600", "1280x720"], 
                                     state="readonly", width=10)
        cam1_res_combo.grid(row=0, column=1, padx=5)
        ttk.Label(cam1_res_frame, text="(Requires restart to take effect)").grid(row=0, column=2, padx=5)
        
        # Camera 1 area multiplier slider
        cam1_area_frame = ttk.Frame(scrollable_frame)
        cam1_area_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cam1_area_frame, text="Area Threshold Multiplier:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(cam1_area_frame, textvariable=self.cam1_area_multiplier_value).grid(row=0, column=2, padx=5)
        cam1_area_scale = ttk.Scale(cam1_area_frame, from_=0.1, to=5.0, orient="horizontal", 
                             variable=self.cam1_area_multiplier, length=200,
                             command=lambda v: self.cam1_area_multiplier_value.set(f"{float(v):.1f}x"))
        cam1_area_scale.grid(row=0, column=1, padx=5)
        ttk.Label(cam1_area_frame, text="(Higher values reduce false positives, 1x = default)").grid(row=1, column=0, columnspan=3, padx=5)
        
        # Camera 1 threshold multiplier slider
        cam1_thresh_frame = ttk.Frame(scrollable_frame)
        cam1_thresh_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cam1_thresh_frame, text="Threshold Multiplier:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(cam1_thresh_frame, textvariable=self.cam1_threshold_multiplier_value).grid(row=0, column=2, padx=5)
        cam1_thresh_scale = ttk.Scale(cam1_thresh_frame, from_=0.5, to=3.0, orient="horizontal", 
                               variable=self.cam1_threshold_multiplier, length=200,
                               command=lambda v: self.cam1_threshold_multiplier_value.set(f"{float(v):.1f}x"))
        cam1_thresh_scale.grid(row=0, column=1, padx=5)
        ttk.Label(cam1_thresh_frame, text="(Higher values reduce sensitivity, 1x = default)").grid(row=1, column=0, columnspan=3, padx=5)
        
        # Camera 1 minimum contours slider
        cam1_contours_frame = ttk.Frame(scrollable_frame)
        cam1_contours_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cam1_contours_frame, text="Minimum Contours Required:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(cam1_contours_frame, textvariable=self.cam1_min_contours_value).grid(row=0, column=2, padx=5)
        cam1_contours_scale = ttk.Scale(cam1_contours_frame, from_=0, to=10, orient="horizontal", 
                                variable=self.cam1_min_contours, length=200,
                                command=lambda v: self.cam1_min_contours_value.set(str(int(float(v)))))
        cam1_contours_scale.grid(row=0, column=1, padx=5)
        ttk.Label(cam1_contours_frame, text="(0 = any movement, higher values require multiple motion areas)").grid(row=1, column=0, columnspan=3, padx=5)

        # Camera 2 specific settings section
        ttk.Label(scrollable_frame, text="Camera 2 Settings (USB Camera)", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Camera 2 resolution dropdown
        cam2_res_frame = ttk.Frame(scrollable_frame)
        cam2_res_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cam2_res_frame, text="Resolution:").grid(row=0, column=0, sticky="w", padx=5)
        cam2_res_combo = ttk.Combobox(cam2_res_frame, textvariable=self.cam2_resolution, 
                                     values=["320x240", "640x480", "800x600", "1280x720"], 
                                     state="readonly", width=10)
        cam2_res_combo.grid(row=0, column=1, padx=5)
        ttk.Label(cam2_res_frame, text="(Requires restart to take effect)").grid(row=0, column=2, padx=5)
        
        # Camera 2 area multiplier slider
        cam2_area_frame = ttk.Frame(scrollable_frame)
        cam2_area_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cam2_area_frame, text="Area Threshold Multiplier:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(cam2_area_frame, textvariable=self.cam2_area_multiplier_value).grid(row=0, column=2, padx=5)
        cam2_area_scale = ttk.Scale(cam2_area_frame, from_=1.0, to=10.0, orient="horizontal", 
                             variable=self.cam2_area_multiplier, length=200,
                             command=lambda v: self.cam2_area_multiplier_value.set(f"{float(v):.1f}x"))
        cam2_area_scale.grid(row=0, column=1, padx=5)
        ttk.Label(cam2_area_frame, text="(Higher values reduce false positives, 4x recommended for white areas)").grid(row=1, column=0, columnspan=3, padx=5)
        
        # Camera 2 threshold multiplier slider
        cam2_thresh_frame = ttk.Frame(scrollable_frame)
        cam2_thresh_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cam2_thresh_frame, text="Threshold Multiplier:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(cam2_thresh_frame, textvariable=self.cam2_threshold_multiplier_value).grid(row=0, column=2, padx=5)
        cam2_thresh_scale = ttk.Scale(cam2_thresh_frame, from_=0.5, to=3.0, orient="horizontal", 
                               variable=self.cam2_threshold_multiplier, length=200,
                               command=lambda v: self.cam2_threshold_multiplier_value.set(f"{float(v):.1f}x"))
        cam2_thresh_scale.grid(row=0, column=1, padx=5)
        ttk.Label(cam2_thresh_frame, text="(Higher values require more change to detect motion, 1.5x recommended)").grid(row=1, column=0, columnspan=3, padx=5)
        
        # Camera 2 minimum contours slider
        cam2_contours_frame = ttk.Frame(scrollable_frame)
        cam2_contours_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cam2_contours_frame, text="Minimum Contours Required:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(cam2_contours_frame, textvariable=self.cam2_min_contours_value).grid(row=0, column=2, padx=5)
        cam2_contours_scale = ttk.Scale(cam2_contours_frame, from_=0, to=10, orient="horizontal", 
                                variable=self.cam2_min_contours, length=200,
                                command=lambda v: self.cam2_min_contours_value.set(str(int(float(v)))))
        cam2_contours_scale.grid(row=0, column=1, padx=5)
        ttk.Label(cam2_contours_frame, text="(0 = any movement, higher values require multiple motion areas)").grid(row=1, column=0, columnspan=3, padx=5)

        # YOLO Object Detection settings
        ttk.Label(scrollable_frame, text="YOLO Object Detection Settings", font=("Arial", 10, "bold")).pack(pady=10)

        # YOLO toggle frame
        yolo_frame = ttk.Frame(scrollable_frame)
        yolo_frame.pack(fill="x", padx=10, pady=5)
        
        # Add YOLO toggle checkbox
        self.use_yolo_var = tk.BooleanVar(value=config.get('use_yolo_detection', True))
        self.yolo_checkbox = ttk.Checkbutton(
            yolo_frame, 
            text="Use YOLO for Detection Confirmation", 
            variable=self.use_yolo_var
        )
        self.yolo_checkbox.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(yolo_frame, text="(Requires motion first, then YOLO must detect objects to trigger alert)").grid(
            row=1, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        
        # Add YOLO confidence threshold slider
        yolo_conf_frame = ttk.Frame(scrollable_frame)
        yolo_conf_frame.pack(fill="x", padx=10, pady=5)
        
        self.yolo_confidence_threshold_value = tk.StringVar(value=f"{config.get('yolo_confidence_threshold', 0.5):.2f}")
        
        ttk.Label(yolo_conf_frame, text="Detection Confidence Threshold:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(yolo_conf_frame, textvariable=self.yolo_confidence_threshold_value).grid(row=0, column=2, padx=5)
        
        self.yolo_conf_threshold_var = tk.DoubleVar(value=config.get('yolo_confidence_threshold', 0.5))
        yolo_conf_scale = ttk.Scale(yolo_conf_frame, from_=0.1, to=0.9, orient="horizontal", 
                                variable=self.yolo_conf_threshold_var, length=200,
                                command=lambda v: self.yolo_confidence_threshold_value.set(f"{float(v):.2f}"))
        yolo_conf_scale.grid(row=0, column=1, padx=5)
        ttk.Label(yolo_conf_frame, text="(Higher values require more certainty, 0.5 = default)").grid(
            row=1, column=0, columnspan=3, padx=5)

        # Modified ROI Section
        ttk.Label(scrollable_frame, text="ROI Selection:", font=("Arial", 10, "bold")).pack(pady=10)
        self.roi_label = ttk.Label(scrollable_frame, text="")
        self.roi_label.pack(pady=5)
        self.update_roi_camera_label()  # Update the ROI label to show current camera
        
        self.roi_button = ttk.Button(scrollable_frame, text="Select ROI", command=self.toggle_roi_selection)
        self.roi_button.pack(pady=5)
        self.roi_status_label = ttk.Label(scrollable_frame, text="Click and drag on camera feed to select region")
        self.roi_status_label.pack(pady=5)
        self.roi_status_label.pack_forget()  # Hide initially

        ttk.Button(scrollable_frame, text="Rotate Camera 1", command=self.rotate_cam1).pack(pady=5)
        ttk.Button(scrollable_frame, text="Rotate Camera 2", command=self.rotate_cam2).pack(pady=5)

        ttk.Label(scrollable_frame, textvariable=self.email_rotation_label).pack(pady=5)
        ttk.Button(scrollable_frame, text="Rotate Email", command=self.rotate_email).pack(pady=5)

        # Add debug button to print current settings to logs
        ttk.Button(scrollable_frame, text="Debug: Print Config", command=self.debug_print_config).pack(pady=5)

        ttk.Button(scrollable_frame, text="Save", command=self.save_settings).pack(pady=10)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add trace to update ROI camera label when detection camera changes
        self.detection_camera.trace_add("write", lambda *args: self.update_roi_camera_label())

    def update_roi_camera_label(self):
        """Update the ROI label to show which camera's ROI is being configured"""
        current_camera = self.detection_camera.get()
        
        if current_camera == "both":
            # When in "both" mode, show ROI for both cameras
            # First get Camera 1 ROI
            cam1_roi = config.get("roi_coordinates", [0, 0, 640, 480])
            if isinstance(cam1_roi, list) and len(cam1_roi) == 4:
                x1, y1, w1, h1 = cam1_roi
            else:
                x1 = cam1_roi.get("x", 0)
                y1 = cam1_roi.get("y", 0)
                w1 = cam1_roi.get("width", 640)
                h1 = cam1_roi.get("height", 480)
                
            # Then get Camera 2 ROI
            cam2_roi = config.get("roi_coordinates_cam2", [0, 0, 320, 240])
            if isinstance(cam2_roi, list) and len(cam2_roi) == 4:
                x2, y2, w2, h2 = cam2_roi
            else:
                x2 = cam2_roi.get("x", 0)
                y2 = cam2_roi.get("y", 0)
                w2 = cam2_roi.get("width", 320)
                h2 = cam2_roi.get("height", 240)
                
            # Update the label with both ROIs
            self.roi_label.config(
                text=f"Camera 1 ROI: [{x1}, {y1}, {w1}, {h1}]\nCamera 2 ROI: [{x2}, {y2}, {w2}, {h2}]"
            )
        else:
            # Single camera mode - just show the selected camera
            cam_text = "Camera 2 (USB)" if current_camera == "cam2" else "Camera 1 (Pi)"
            roi_key = "roi_coordinates_cam2" if current_camera == "cam2" else "roi_coordinates"
            
            # Get the current ROI settings
            roi = config.get(roi_key, [0, 0, 640, 480])
            if isinstance(roi, list) and len(roi) == 4:
                x, y, w, h = roi
            else:
                x = roi.get("x", 0)
                y = roi.get("y", 0)
                w = roi.get("width", 640)
                h = roi.get("height", 480)
                
            # Update the label text
            self.roi_label.config(text=f"ROI for {cam_text}: [{x}, {y}, {w}, {h}]")

    def set_live_feed_to_local(self):
        self.live_feed_url.set(f"http://{self.local_ip}:{self.flask_port.get()}/login")

    def set_flask_host_to_local(self):
        self.flask_host.set(self.local_ip)

    def update_receiver_fields(self):
        try:
            for widget in self.receiver_frame.winfo_children():
                widget.destroy()
            for i, email_var in enumerate(self.receiver_emails):
                ttk.Label(self.receiver_frame, text=f"Receiver Email {i+1}:").grid(row=i, column=0, padx=5, pady=2)
                ttk.Entry(self.receiver_frame, textvariable=email_var, width=30).grid(row=i, column=1, padx=5, pady=2)
        except Exception as e:
            logging.error(f"Error in update_receiver_fields: {e}")

    def add_receiver(self):
        try:
            self.receiver_emails.append(tk.StringVar())
            self.update_receiver_fields()
        except Exception as e:
            logging.error(f"Error in add_receiver: {e}")

    def update_user_fields(self):
        try:
            for widget in self.user_frame.winfo_children():
                widget.destroy()
            for i, (email_var, pass_var) in enumerate(self.users):
                ttk.Label(self.user_frame, text=f"User {i+1} Email:").grid(row=i*2, column=0, padx=5, pady=2)
                ttk.Entry(self.user_frame, textvariable=email_var, width=30).grid(row=i*2, column=1, padx=5, pady=2)
                ttk.Label(self.user_frame, text=f"User {i+1} Password:").grid(row=i*2+1, column=0, padx=5, pady=2)
                ttk.Entry(self.user_frame, textvariable=pass_var, width=30, show="*").grid(row=i*2+1, column=1, padx=5, pady=2)
        except Exception as e:
            logging.error(f"Error in update_user_fields: {e}")

    def add_user(self):
        try:
            self.users.append((tk.StringVar(), tk.StringVar()))
            self.update_user_fields()
        except Exception as e:
            logging.error(f"Error in add_user: {e}")

    def rotate_cam1(self):
        """PYTHON 3.14: Lock-free rotation using AtomicCameraState"""
        global last_rotation_time
        current_time = time.time()

        # PYTHON 3.14: Use atomic camera state
        if cam1_state.rotate_atomic(current_time, ROTATION_COOLDOWN):
            logging.info(f"Settings: Camera 1 rotation set to {cam1_state.get_rotation()} degrees")

    def rotate_cam2(self):
        """PYTHON 3.14: Rotation with multiprocessing.Value lock (for cross-process safety)"""
        global cam2_rotation, last_rotation_time
        current_time = time.time()

        # Keep multiprocessing.Value lock for cross-process safety
        if current_time - last_rotation_time >= ROTATION_COOLDOWN:
            with cam2_rotation.get_lock():
                cam2_rotation.value = (cam2_rotation.value + 90) % 360
            last_rotation_time = current_time
            logging.info(f"Settings: Camera 2 rotation set to {cam2_rotation.value} degrees")

    def rotate_email(self):
        global email_rotation_cam1
        with email_rotation_cam1.get_lock():
            email_rotation_cam1.value = (email_rotation_cam1.value + 90) % 360
            self.email_rotation_label.set(f"Email Rotation: {email_rotation_cam1.value}deg")
            logging.info(f"Settings: Email rotation set to {email_rotation_cam1.value} degrees")

    def toggle_roi_selection(self):
        """Toggle ROI selection mode using the global functions"""
        global roi_selection_mode, detection_camera
        
        # Update detection_camera from the current setting
        detection_camera = self.detection_camera.get()
        
        if roi_selection_mode:
            # If already in selection mode, deactivate it
            deactivate_roi_selection(save_roi=False)
            # Manually update UI here as well for redundancy
            self.roi_button.config(text="Select ROI")
            self.roi_status_label.pack_forget()
            logging.debug("ROI button text manually reset to 'Select ROI' in toggle function")
        else:
            # Handle "both" cameras special case
            if detection_camera == "both":
                # We'll start with Camera 1, then do Camera 2 after
                # Temporarily set detection_camera to cam1
                detection_camera = "cam1"
                is_both_cameras = True
                # Mark that we're in a sequence for both cameras
                self.roi_sequence_both = True
                self.original_detection_camera = "both"
                logging.info("Starting ROI selection sequence for both cameras")
            else:
                is_both_cameras = False
                self.roi_sequence_both = False
                
            # Activate selection mode for first camera
            if activate_roi_selection(is_both_cameras=is_both_cameras):
                # Only update UI if activation was successful
                self.roi_button.config(text="Cancel ROI Selection")
                self.roi_status_label.pack(pady=5)
                self.roi_status_label.config(text="Click and drag on Camera 1 feed to select region")
                logging.debug("ROI button text set to 'Cancel ROI Selection'")
            else:
                # If activation failed, ensure UI reflects inactive state
                self.roi_button.config(text="Select ROI")
                self.roi_status_label.pack_forget()
                logging.debug("ROI activation failed, button text reset")

    def on_detection_camera_change(self, *args):
        """Immediately update the global detection_camera variable when dropdown changes"""
        global detection_camera
        new_value = self.detection_camera.get()
        if new_value in ["cam1", "cam2", "both", "disable"]:
            detection_camera = new_value
            self.update_roi_camera_label()  # Update the ROI display with the new camera
            logging.info(f"Detection camera immediately updated to: {detection_camera}")
            
            # Update config file immediately to ensure persistence
            config["detection_camera"] = new_value
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=4)
                logging.info(f"Detection camera setting saved to config: {new_value}")
            except Exception as e:
                logging.error(f"Failed to save detection camera setting: {e}")
                
            # Force update the toggle dialog if it exists
            if toggle_dialog and toggle_dialog.winfo_exists():
                toggle_dialog.update_status()

    def save_settings(self):
        """Save settings to config file"""
        try:
            port = 5000
            try:
                port = int(self.flask_port.get())
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid port value: {e}")
                messagebox.showerror("Invalid Port", "Port must be a number. Using default 5000.")
                port = 5000

            # Format validation for timer settings
            arm_time = self.schedule_arm_time.get()
            disarm_time = self.schedule_disarm_time.get()
            
            # Validate time format (HH:MM)
            time_format_valid = True
            try:
                # Check arm time format
                if not re.match(r'^([01]\d|2[0-3]):([0-5]\d)$', arm_time):
                    messagebox.showerror("Invalid Time Format", "Auto Arm time must be in HH:MM format (24-hour).")
                    time_format_valid = False

                # Check disarm time format
                if not re.match(r'^([01]\d|2[0-3]):([0-5]\d)$', disarm_time):
                    messagebox.showerror("Invalid Time Format", "Auto Disarm time must be in HH:MM format (24-hour).")
                    time_format_valid = False
            except (AttributeError, TypeError) as e:
                logging.error(f"Time format validation error: {e}")
                messagebox.showerror("Time Format Error", "Please enter valid times in HH:MM format (24-hour).")
                time_format_valid = False
                
            if not time_format_valid:
                return
                
            # Filter out empty receiver emails
            receiver_emails = [email.get() for email in self.receiver_emails if email.get().strip()]

            # Prepare user data, handling password updates
            users = []
            for email_var, pwd_var in self.users:
                email = email_var.get().strip()
                if not email:
                    continue
                password = pwd_var.get()
                # Check if this is an existing user
                existing_user = next((user for user in config.get('users', []) if user.get('email') == email), None)
                if existing_user and password == "********":
                    # Keep existing password hash for this user
                    users.append({"email": email, "password_hash": existing_user['password_hash']})
                else:
                    # Create new password hash for new user or updated password
                    pwd_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    users.append({"email": email, "password_hash": pwd_hash})

            # Calculate timer settings in seconds for convenience in the main loop
            timer_arm_seconds = None
            timer_disarm_seconds = None
            
            if self.timer_enabled.get():
                try:
                    # Parse the HH:MM strings and convert to seconds since midnight
                    arm_hour, arm_minute = map(int, arm_time.split(':'))
                    disarm_hour, disarm_minute = map(int, disarm_time.split(':'))
                    
                    # Calculate seconds since midnight
                    timer_arm_seconds = arm_hour * 3600 + arm_minute * 60
                    timer_disarm_seconds = disarm_hour * 3600 + disarm_minute * 60
                    
                    logging.info(f"Timer settings calculated: Arm={arm_time} ({timer_arm_seconds} secs), " +
                                f"Disarm={disarm_time} ({timer_disarm_seconds} secs)")
                except Exception as e:
                    logging.error(f"Error converting time to seconds: {e}")
                    timer_arm_seconds = None
                    timer_disarm_seconds = None
                    messagebox.showerror("Timer Error", f"Could not process timer settings: {e}")
            else:
                logging.info("Timer disabled - not calculating seconds values")
            
            # Create a new config dictionary
            new_config = config.copy()
            new_config.update({
                "live_feed_url": self.live_feed_url.get().strip(),
                "flask_host": self.flask_host.get().strip(),
                "flask_port": port,
                "users": users,
                "sender_email": self.sender_email.get().strip(),
                "receiver_emails": receiver_emails,
                "email_password": self.email_password.get().strip(),
                "detection_camera": self.detection_camera.get().strip(),
                "email_rotation_cam1": email_rotation_cam1.value,
                "background_threshold": self.background_threshold.get(),
                
                # Timer settings
                "timer_enabled": self.timer_enabled.get(),
                "schedule_arm_time": arm_time,
                "schedule_disarm_time": disarm_time,
                "schedule_arm_seconds": timer_arm_seconds,
                "schedule_disarm_seconds": timer_disarm_seconds,
                
                # Camera 1 settings
                "cam1_area_multiplier": round(self.cam1_area_multiplier.get(), 1),
                "cam1_threshold_multiplier": round(self.cam1_threshold_multiplier.get(), 1),
                "cam1_min_contours": self.cam1_min_contours.get(),
                
                # Camera 2 settings
                "cam2_area_multiplier": round(self.cam2_area_multiplier.get(), 1),
                "cam2_threshold_multiplier": round(self.cam2_threshold_multiplier.get(), 1),
                "cam2_min_contours": self.cam2_min_contours.get(),
                
                # YOLO settings
                "use_yolo_detection": self.use_yolo_var.get(),
                "yolo_confidence_threshold": round(self.yolo_conf_threshold_var.get(), 2),
                
                # Keep existing ROI coordinates for both cameras
                "roi_coordinates": config.get("roi_coordinates", [0, 0, 640, 480]),
                "roi_coordinates_cam2": config.get("roi_coordinates_cam2", [0, 0, 320, 240]),
                "cam1_resolution": self.cam1_resolution.get(),
                "cam2_resolution": self.cam2_resolution.get()
            })
            
            # Save in a non-blocking way
            def save_config_task():
                try:
                    # Declare all globals at the beginning of the function
                    global timer_enabled, schedule_arm_time, schedule_disarm_time
                    global schedule_arm_seconds, schedule_disarm_seconds
                    global detection_active, email_armed, detection_camera, root
                    
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump(new_config, f, indent=4)
                    
                    # Update the global config and detection camera
                    config.update(new_config)
                    
                    # Store original setting to check for YOLO changes
                    original_yolo_setting = config.get('use_yolo_detection', True)
                    
                    # Store previous timer state to detect changes
                    previous_timer_enabled = timer_enabled
                    
                    timer_enabled = config.get("timer_enabled", False)
                    schedule_arm_time = config.get("schedule_arm_time")
                    schedule_disarm_time = config.get("schedule_disarm_time")
                    schedule_arm_seconds = config.get("schedule_arm_seconds")
                    schedule_disarm_seconds = config.get("schedule_disarm_seconds")
                    
                    # Handle state based on timer being enabled or disabled
                    if previous_timer_enabled and not timer_enabled:
                        # Timer was enabled and is now being disabled
                        if detection_camera != "disable":
                            # Restore detection to active state
                            detection_active = True
                            email_armed = True
                            logging.info("Timer disabled - restoring detection active and email armed to True")
                            
                            # Set flag to show "TIMER DISABLED" message for a few seconds
                            root.timer_recently_disabled = time.time()
                            
                            # Update the toggle button to reflect the restored state
                            if toggle_dialog and toggle_dialog.winfo_exists():
                                toggle_dialog.toggle_button.config(text="Armed")
                    
                    elif not previous_timer_enabled and timer_enabled:
                        # Timer was disabled and is now enabled - this should clear any manual override
                        logging.info("Timer enabled - resetting any manual override")
                        
                        # Get current time and determine if we should be armed based on schedule
                        current_time = time.time()
                        current_time_struct = time.localtime(current_time)
                        current_seconds = current_time_struct.tm_hour * 3600 + current_time_struct.tm_min * 60 + current_time_struct.tm_sec
                        
                        # Calculate if system should be armed now based on schedule
                        should_be_armed = False
                        if schedule_arm_seconds is not None and schedule_disarm_seconds is not None:
                            if schedule_arm_seconds < schedule_disarm_seconds:
                                # Time window doesn't cross midnight
                                should_be_armed = (current_seconds >= schedule_arm_seconds and 
                                                  current_seconds < schedule_disarm_seconds)
                            else:
                                # Time window crosses midnight
                                should_be_armed = (current_seconds >= schedule_arm_seconds or 
                                                  current_seconds < schedule_disarm_seconds)
                        
                        # Update armed state to match schedule immediately
                        email_armed = should_be_armed
                        if detection_camera != "disable":
                            detection_active = email_armed
                        
                        # Update toggle button if available
                        if toggle_dialog and toggle_dialog.winfo_exists():
                            toggle_dialog.toggle_button.config(text="Armed" if email_armed else "Disarmed")
                            logging.info(f"Reset armed state to {email_armed} per schedule")
                    
                    logging.info(f"Timer settings updated: enabled={timer_enabled}, arm={schedule_arm_time}, disarm={schedule_disarm_time}")
                    
                    # Update the detection_camera variable
                    detection_camera = new_config["detection_camera"]
                    logging.info(f"Detection camera updated to: {detection_camera}")
                    
                    # PYTHON 3.14: Reload YOLO processor if setting changed
                    if config.get('use_yolo_detection', False) != original_yolo_setting:
                        global yolo_processor
                        logging.info("YOLO detection setting changed. Reloading...")
                        load_yolo_model()  # Legacy model reload

                        # Reinitialize ParallelYOLOProcessor via load_config_modern()
                        load_config_modern()

                        if yolo_processor is not None:
                            logging.info("YOLO ParallelYOLOProcessor reinitialized")
                        else:
                            logging.info("YOLO detection disabled")
                    
                    # Update the ROI camera label to reflect any changes
                    self.update_roi_camera_label()
                    
                    # Show success message and hide dialog
                    messagebox.showinfo("Settings Saved", "Settings saved to config.json. Changes applied.")
                    logging.info(f"Settings saved: flask_host={new_config['flask_host']}, flask_port={port}")
                    self.withdraw()
                except Exception as save_e:
                    messagebox.showerror("Save Error", f"Failed to save settings: {save_e}")
                    logging.error(f"Error saving settings: {save_e}")
            
            # Use after() to schedule the save task
            self.after(10, save_config_task)
            
        except Exception as e:
            logging.error(f"Error in save_settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def show(self):
        if self.winfo_exists():
            self.deiconify()
            self.lift()
            self.grab_set()
            # Update the ROI label with the current camera
            self.update_roi_camera_label()
            logging.info("SettingsDialog shown")

    def on_close(self):
        if self.winfo_exists():
            self.withdraw()
            logging.info("SettingsDialog hidden via close button")

    def debug_print_config(self):
        try:
            # Print current config to logs
            logging.info(f"Current config: {json.dumps(config, indent=2)}")
            
            # Print Camera 1 specific settings
            cam1_settings = {
                "resolution": config.get("cam1_resolution", "640x480"),
                "cam1_area_multiplier": config.get("cam1_area_multiplier", 1.0),
                "cam1_threshold_multiplier": config.get("cam1_threshold_multiplier", 1.0),
                "cam1_min_contours": config.get("cam1_min_contours", 0)
            }
            logging.info(f"Camera 1 settings: {json.dumps(cam1_settings, indent=2)}")
            
            # Print Camera 2 specific settings
            cam2_settings = {
                "resolution": config.get("cam2_resolution", "320x240"),
                "cam2_area_multiplier": config.get("cam2_area_multiplier", 4.0),
                "cam2_threshold_multiplier": config.get("cam2_threshold_multiplier", 1.5),
                "cam2_min_contours": config.get("cam2_min_contours", 0)
            }
            logging.info(f"Camera 2 settings: {json.dumps(cam2_settings, indent=2)}")
            
            messagebox.showinfo("Debug", "Current config printed to logs")
        except Exception as e:
            logging.error(f"Error printing debug config: {e}")

def update_settings(settings_dialog):
    settings_dialog.show()

# Fix the ROI selection to use the correct camera frame
def activate_roi_selection(is_both_cameras=False):
    """Direct function to activate ROI selection mode"""
    global roi_selection_mode, cap, roi_temp, roi_start_point, roi_end_point, roi_drawing, detection_camera
    
    try:
        # Reset ROI variables
        roi_selection_mode = True
        roi_drawing = False
        roi_start_point = None
        roi_end_point = None
        
        # Store whether we need to handle both cameras
        if is_both_cameras:
            # Store flag for later handling in deactivate function
            roi_selection_mode = "both_cameras_cam1"
        
        # Choose the appropriate camera frame based on detection_camera setting
        if detection_camera == "cam2":
            logging.info("ROI selection for Camera 2 (USB camera)")
            
            # Wait for frame from USB camera
            for attempt in range(10):  # Try up to 10 times to get a valid frame
                # multiprocessing.Queue: use get() with timeout
                try:
                    frame = usb_frame_queue.get(timeout=0.5)
                    if frame is not None and frame.size > 0:
                        # USB camera frames already have rotation applied in the process
                        roi_temp = frame.copy()
                        break
                except queue.Empty:
                    pass  # Timeout waiting for frame, will retry
                logging.debug(f"Waiting for USB frame, attempt {attempt+1}")
            
            # If we still don't have a frame, show an error
            if roi_temp is None:
                logging.error("Failed to get valid frame from USB camera for ROI selection")
                messagebox.showerror("Error", "Could not get a valid frame from Camera 2. Please try again.")
                roi_selection_mode = False
                return False
                
        else:  # Default to Camera 1 (Pi camera)
            logging.info("ROI selection for Camera 1 (Pi camera)")
            
            # Get a frame from Pi camera
            if cap is None or not cap.isOpened():
                logging.error("Primary camera capture object is not valid")
                messagebox.showerror("Error", "Camera 1 is not available.")
                roi_selection_mode = False
                return False
                
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logging.error("Failed to capture frame from primary camera")
                messagebox.showerror("Error", "Failed to capture frame from Camera 1.")
                roi_selection_mode = False
                return False
            
            # Apply the current rotation to the Pi camera frame
            frame = apply_rotation(frame, cam1_rotation)
            roi_temp = frame.copy()
        
        # Create a separate window for ROI selection with the correct camera name
        camera_text = "Camera 2 (USB)" if detection_camera == "cam2" else "Camera 1 (Pi)"
        window_name = f"ROI Selection for {camera_text} - Click and drag to select a region"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.setMouseCallback(window_name, roi_selection_callback)
        
        # Show the frame in the ROI selection window
        cv2.imshow(window_name, roi_temp)
        
        # Update UI if possible
        if settings_dialog and settings_dialog.winfo_exists():
            settings_dialog.roi_button.config(text="Cancel ROI Selection")
            settings_dialog.roi_status_label.pack(pady=5)
            
        logging.info(f"ROI selection mode activated for {camera_text} - click and drag to select region in the separate window")
        return True
    except Exception as e:
        logging.error(f"Error activating ROI selection: {e}")
        roi_selection_mode = False
        try:
            cv2.destroyWindow(window_name)
        except cv2.error as cv_err:
            logging.warning(f"Could not destroy window {window_name}: {cv_err}")
            pass
        
        # Show error to user
        messagebox.showerror("Error", f"Error selecting ROI: {e}")
        return False

def deactivate_roi_selection(save_roi=False):
    """Direct function to deactivate ROI selection mode"""
    global roi_selection_mode
    
    # Handle possible cancellation during "both" cameras mode
    if isinstance(roi_selection_mode, str) and roi_selection_mode.startswith("both_cameras_"):
        # If user cancels during "both" sequence, we need to restore detection_camera setting
        logging.info("ROI selection for both cameras cancelled midway")
        # Use a timer to safely restore the detection_camera value
        threading.Timer(0.1, lambda: safe_restore_detection_camera()).start()
    
    # Use the safer thread-based approach
    threading.Timer(0.1, lambda: safe_deactivate_roi(roi_start_point, roi_end_point, save_roi)).start()
    return True

def safe_restore_detection_camera():
    """Helper function to restore detection_camera to 'both' if selection was cancelled"""
    global detection_camera
    detection_camera = "both"
    logging.info("Restored detection_camera to 'both' after cancellation")

def safe_deactivate_roi(start_point, end_point, save_roi=True):
    """Safely deactivate ROI selection mode from a separate thread"""
    global roi_selection_mode, config, toggle_dialog, settings_dialog, detection_camera
    
    try:
        # Only proceed if we're still in ROI selection mode
        if not roi_selection_mode:
            return
            
        # Check if we're in the "both cameras" sequence
        is_both_cameras_mode = isinstance(roi_selection_mode, str) and roi_selection_mode.startswith("both_cameras_")
        current_mode = roi_selection_mode  # Store current mode before resetting it
        
        # Get window name based on detection camera for closing
        camera_text = "Camera 2 (USB)" if detection_camera == "cam2" else "Camera 1 (Pi)"
        window_name = f"ROI Selection for {camera_text} - Click and drag to select a region"
            
        # Calculate and save ROI if requested
        if save_roi and start_point and end_point:
            x1, y1 = start_point
            x2, y2 = end_point
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            width = x2 - x1
            height = y2 - y1
            
            # Validate minimum size
            if width >= 5 and height >= 5:
                # Use camera-specific ROI coordinates
                roi_key = "roi_coordinates_cam2" if detection_camera == "cam2" else "roi_coordinates"
                
                # Save ROI coordinates to config with the appropriate key
                config[roi_key] = [x1, y1, width, height]
                try:
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump(config, f, indent=4)
                    logging.info(f"ROI saved for {camera_text}: x={x1}, y={y1}, width={width}, height={height}")
                except Exception as e:
                    logging.error(f"Failed to save ROI to config: {e}")
            else:
                logging.warning(f"ROI too small ({width}x{height}), not saving")
        
        # Close the current ROI window
        try:
            cv2.destroyWindow(window_name)
        except Exception as window_e:
            logging.error(f"Error closing ROI window: {window_e}")
            
        # Check if we need to continue with Camera 2 for "both" mode
        if is_both_cameras_mode and current_mode == "both_cameras_cam1":
            # We just finished Camera 1, now do Camera 2
            logging.info("First camera ROI done, now showing Camera 2 ROI selection")
            
            # Store current state for second camera
            next_camera = "cam2"
            
            # Reset the ROI mode first to avoid any issues
            roi_selection_mode = False
            
            # Schedule the second camera ROI selection on the main thread
            # This avoids thread safety issues with Tkinter
            if settings_dialog:
                try:
                    # Store the callback in a global variable to run from main thread
                    global pending_roi_activation
                    pending_roi_activation = {
                        "camera": next_camera,
                        "is_second_camera": True
                    }
                    logging.info("Scheduled Camera 2 ROI selection from main thread")
                except Exception as e:
                    logging.error(f"Error scheduling second camera ROI: {e}")
            return
            
        # If we're here, we're completely done with ROI selection
        # Reset the mode
        roi_selection_mode = False
        
        # Reset callbacks on main window
        try:
            cv2.setMouseCallback("Camera Feeds", toggle_visibility, toggle_dialog)
        except Exception as callback_e:
            logging.error(f"Error resetting mouse callback: {callback_e}")
        
        # Update UI if possible - ENSURE THIS RUNS ON MAIN THREAD
        if settings_dialog is not None:
            try:
                # Store UI update request in global variable for main thread
                global pending_ui_update
                pending_ui_update = {
                    "action": "reset_roi_ui",
                    "camera": detection_camera
                }
                logging.debug("Scheduled UI update for ROI button via main thread")
            except Exception as ui_e:
                logging.error(f"Error scheduling UI update: {ui_e}")
        else:
            logging.warning("Settings dialog not available for UI update")
        
        logging.info(f"ROI selection mode safely deactivated for {camera_text}")
    except Exception as e:
        logging.error(f"Error in safe_deactivate_roi: {e}")
        roi_selection_mode = False
        
        # Try to clean up any open windows
        try:
            for name in ["ROI Selection for Camera 1 (Pi)", "ROI Selection for Camera 2 (USB)"]:
                try:
                    cv2.destroyWindow(name)
                except cv2.error:
                    pass  # Window may not exist
        except Exception as e:
            logging.debug(f"Error cleaning up ROI windows: {e}")
            pass

# Add these global variables to track pending actions for the main thread
pending_roi_activation = None
pending_ui_update = None

def roi_selection_callback(event, x, y, flags, param):
    """Simplified mouse callback for ROI selection"""
    global roi_selection_mode, roi_start_point, roi_end_point, roi_drawing, roi_temp, detection_camera, settings_dialog
    
    # Handle possible string value of roi_selection_mode for "both" cameras
    if not roi_selection_mode or (isinstance(roi_selection_mode, str) and not roi_selection_mode.startswith("both_cameras_")):
        return
        
    # Get window name based on detection camera
    camera_text = "Camera 2 (USB)" if detection_camera == "cam2" else "Camera 1 (Pi)"
    window_name = f"ROI Selection for {camera_text} - Click and drag to select a region"
    
    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_drawing = True
            roi_start_point = (x, y)
            logging.debug(f"ROI start point: {roi_start_point}")
            
        elif event == cv2.EVENT_MOUSEMOVE and roi_drawing:
            if roi_temp is not None:
                # Draw temporary rectangle
                current_frame = roi_temp.copy()
                cv2.rectangle(current_frame, roi_start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow(window_name, current_frame)
                
        elif event == cv2.EVENT_LBUTTONUP:
            roi_drawing = False
            roi_end_point = (x, y)
            logging.debug(f"ROI end point: {roi_end_point}")
            
            # Draw final rectangle
            if roi_temp is not None:
                final_frame = roi_temp.copy()
                cv2.rectangle(final_frame, roi_start_point, roi_end_point, (0, 255, 0), 2)
                
                # Show final selection
                cv2.imshow(window_name, final_frame)
                
                # Process the ROI in a non-blocking way - no UI updates here
                # Set a timer to call safe_deactivate_roi after a short delay
                # All UI updates will be handled in the main thread
                threading.Timer(0.5, lambda: safe_deactivate_roi(roi_start_point, roi_end_point)).start()
                logging.debug("ROI selection completed, processing scheduled")
            
    except Exception as e:
        logging.error(f"Error in ROI selection callback: {e}")
        # Use the timer approach for cleanup too
        threading.Timer(0.1, lambda: safe_deactivate_roi(None, None, False)).start()

# Update main to handle pending actions
def main():
    global streaming_active, recording, detection_active, pipeline_process, cam1_rotation, cam2_rotation
    global cam2_available, email_rotation_cam1, screen_width, screen_height, last_cam1_rotation, last_cam2_rotation
    global last_rotation_time, email_armed, button_visible, config, authorized_users, last_email_time, cooldown_active
    global usb_frame_queue
    global cap, toggle_dialog, settings_dialog, detection_camera, motion_source_camera
    # PYTHON 3.14: Removed old YOLO worker globals (yolo_worker_thread, yolo_worker_running, yolo_request_queue, yolo_response_queue)
    global yolo_annotated_pi_frame, yolo_annotated_usb_frame
    global roi_selection_mode, roi_temp, pending_roi_activation, pending_ui_update

    root = tk.Tk()
    root.withdraw()

    video_writer = temp_video_file = None
    pi_previous_frame = None  # Keep for frame differencing method
    usb_previous_frame = None  # Keep for frame differencing method
    detection_check_start = None  # Track detection timing
    # Initialize last_timer_check
    last_timer_check = time.time()

    # Load config at start of main
    load_config()
    logging.debug(f"Config after initial load in main: {type(config)}, {config}")
    
    
    # PYTHON 3.14: YOLO processor is now initialized in load_config_modern()
    # Old worker thread initialization removed - using ParallelYOLOProcessor instead
    load_yolo_model()  # Legacy - still needed for backward compatibility
    if config.get('use_yolo_detection', False):
        logging.info("YOLO object detection enabled (using ParallelYOLOProcessor)")
    else:
        logging.info("YOLO object detection disabled in config")

    # NEW ARCHITECTURE: Initialize managers
    # Initialize frame manager with blank frames
    cam1_res_str = config.get('cam1_resolution', '640x480')
    cam2_res_str = config.get('cam2_resolution', '1280x720')
    cam1_w, cam1_h = map(int, cam1_res_str.split('x'))
    cam2_w, cam2_h = map(int, cam2_res_str.split('x'))
    frame_manager.initialize_blanks(
        cam1_resolution=(cam1_w, cam1_h),
        cam2_resolution=(cam2_w, cam2_h)
    )
    
    # Wire YOLO processor to yolo_manager
    yolo_manager.processor = yolo_processor
    
    # Set state_manager timeouts from config
    state_manager.roi_yolo_timeout = YOLO_ROI_TIMEOUT
    state_manager.full_yolo_timeout = YOLO_FULL_FRAME_TIMEOUT
    state_manager.cooldown_duration = EMAIL_COOLDOWN
    
    logging.info("NEW ARCHITECTURE: All managers initialized successfully")


    # Initialize rotation tracking variables
    last_cam1_rotation = 0
    last_cam2_rotation = 0

    # Debug timer settings
    if timer_enabled:
        logging.info(f"TIMER SETTINGS AT STARTUP: Enabled={timer_enabled}, " +
                   f"Arm time={schedule_arm_time} ({schedule_arm_seconds} secs), " +
                   f"Disarm time={schedule_disarm_time} ({schedule_disarm_seconds} secs)")
    else:
        logging.info("Timer is disabled at startup")
        
    # Reset motion detection tracking variables
    motion_detection_time_cam1 = None
    motion_detection_time_cam2 = None
    
    # Initialize background subtractors (KNN for better dynamic lighting/weather handling)
    bg_subtractor_cam1 = cv2.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=400.0, history=500)
    bg_subtractor_cam2 = cv2.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=400.0, history=500)

    # Reset frame counters for warmup period
    knn_frame_counter_cam1 = 0
    knn_frame_counter_cam2 = 0

    logging.info("Initialized KNN background subtractors for both cameras.")

    # FORCE RESET ROI COORDINATES TO VALID VALUES
    # This ensures that regardless of what was in the config file, we start with valid ROI
    # Set ROI to full frame for each camera type
    # config["roi_coordinates"] = [0, 0, 640, 480]  # Pi camera (typically 640x480)
    # config["roi_coordinates_cam2"] = [0, 0, 320, 240]  # USB camera (typically 320x240)
    
    # Save the updated config immediately
    # try:
    #     with open(CONFIG_FILE, 'w') as f:
    #         json.dump(config, f, indent=4)
    #     logging.info("ROI coordinates reset to full frame for both cameras")
    # except Exception as e:
    #     logging.error(f"Failed to save reset ROI coordinates: {e}")
    
    authorized_users.update({user["email"]: user["password_hash"].encode('utf-8') for user in config["users"]})
    sender_email = config['sender_email']
    receiver_emails = config['receiver_emails']
    email_password = config['email_password']
    flask_host = config['flask_host']
    flask_port = config['flask_port']

    # Start Pipeline and USB process
    try:
        subprocess.run("sudo pkill -9 -f 'libcamera-vid'", shell=True, check=True)
    except subprocess.CalledProcessError:
        logging.warning("pkill libcamera-vid failed, maybe it wasn't running.")
    try:
        subprocess.run("sudo modprobe v4l2loopback video_nr=10", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to load v4l2loopback module: {e}. Check permissions and module availability.")
        # Optionally exit or try to continue without Pi camera
    start_pipeline()

    # Get cam2 resolution from config and pass to subprocess
    cam2_res = config.get("cam2_resolution", "1280x720")
    usb_process = Process(target=usb_camera_process, args=(usb_frame_queue, cam2_rotation, cam2_available, cam2_res), daemon=True)
    usb_process.start()

    time.sleep(2)
    with cam2_available.get_lock():
        logging.info(f"Camera 2 available: {cam2_available.value}")

    # Start Flask Server and Purge Thread
    server_thread = threading.Thread(target=lambda: app.run(host=flask_host, port=flask_port, threaded=True))
    server_thread.start()
    logging.info(f"Flask server started on {flask_host}:{flask_port}")
    purge_thread = threading.Thread(target=purge_old_media, daemon=True)
    purge_thread.start()
    logging.info("Purge thread started")

    # Initialize UI Dialogs
    logging.debug(f"Config type before dialogs: {type(config)}, contents: {config}")
    toggle_dialog = ToggleDialog(root)
    settings_dialog = SettingsDialog(root)  # Now stored in global variable

    # --- Pi Camera Initialization with Fault Tolerance ---
    cap = None
    pi_camera_available = False  # Track if Pi camera is actually available
    # Initialize with default blank frame based on config or default resolution
    try:
        cam1_res_w, cam1_res_h = map(int, config.get("cam1_resolution", "640x480").split('x'))
    except ValueError:
        cam1_res_w, cam1_res_h = 640, 480 # Fallback resolution
    last_pi_frame = np.zeros((cam1_res_h, cam1_res_w, 3), dtype=np.uint8) # Default blank frame

    try:
        cap = cv2.VideoCapture(10, cv2.CAP_V4L2)
        if not cap or not cap.isOpened():
            logging.error("Failed to open Pi Camera (/dev/video10) on initial attempt.")
            cap = None # Ensure cap is None if opening failed
        else:
            # Try to grab an initial frame if camera opened successfully
            ret, frame = cap.read()
            if ret and frame is not None:
                logging.info("Initial Pi camera frame captured successfully.")
                pi_camera_available = True  # Camera is working
                # Resize frame if it doesn't match expected dimensions (less likely needed here but safe)
                if frame.shape[1] != cam1_res_w or frame.shape[0] != cam1_res_h:
                     frame = cv2.resize(frame, (cam1_res_w, cam1_res_h))
                # PYTHON 3.14: LockFreeQueue.put() is non-blocking, drops oldest if full
                frame_queue.put(frame)
                last_pi_frame = frame.copy() # Use real frame if available
            else:
                 logging.warning("Pi Camera opened but failed to capture initial frame.")
                 cap.release() # Release if initial frame fails
                 cap = None
    except Exception as e:
        logging.error(f"Exception during Pi Camera initialization: {e}")
        cap = None

    if cap is None:
        logging.warning("Continuing without Pi Camera.")
        # last_pi_frame is already initialized to a blank frame of the correct size

    # --- End Pi Camera Initialization ---

    screen_width, screen_height = get_screen_resolution()
    logging.info(f"Detected screen resolution: {screen_width}x{screen_height}")

    cv2.namedWindow("Camera Feeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Feeds", screen_width, screen_height)
    cv2.setWindowProperty("Camera Feeds", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Camera Feeds", toggle_visibility, toggle_dialog)
    logging.info(f"Window size set to screen resolution: {screen_width}x{screen_height}")

    # Main loop variables
    # last_pi_frame already initialized
    # Initialize blank USB frame based on config or default resolution
    try:
        cam2_res_w, cam2_res_h = map(int, config.get("cam2_resolution", "320x240").split('x'))
    except ValueError:
        cam2_res_w, cam2_res_h = 320, 240 # Fallback resolution
    last_usb_frame = np.zeros((cam2_res_h, cam2_res_w, 3), dtype=np.uint8) # Initialize blank USB frame
    last_usb_frame_raw = last_usb_frame.copy()  # Initialize raw frame buffer
    last_pi_frame_raw = last_pi_frame.copy()  # Initialize raw pi frame buffer
    processed_pi_frame = last_pi_frame.copy()  # Initialize processed frames
    processed_usb_frame = last_usb_frame.copy() # Initialize processed frames

    frame_interval = 1 / 30
    # Initialize Pi camera reconnection handler with exponential backoff
    pi_camera_reconnector = CameraReconnector("Pi Camera", initial_delay=2.0, max_delay=60.0)
    use_background_subtraction = True # <<< Set this to True to use BG subtraction, False for Frame Differencing

    # FPS tracking for performance monitoring
    fps_frame_count = 0
    fps_start_time = time.time()
    fps_update_interval = 2.0  # Log FPS every 2 seconds

    while True:
        current_time = time.time()

        # Track FPS - moved to after cv2.imshow() to measure actual display updates
        # (counter will be incremented after frame is actually displayed)

        if cooldown_active and (current_time - last_email_time >= EMAIL_COOLDOWN):
            cooldown_active = False
            email_armed = True
            logging.info("Email cooldown ended, system re-armed")
            
        # Check scheduled timer for automatic arming/disarming
        if timer_enabled and (current_time - last_timer_check >= 5):  # Check every 5 seconds
            last_timer_check = current_time
            
            # Don't apply scheduled changes if there's a manual override in effect
            manual_override = False
            if toggle_dialog and toggle_dialog.winfo_exists():
                button_text = toggle_dialog.toggle_button['text']
                manual_override = "(Manual)" in button_text
            
            if not manual_override:
                # Get current time in seconds since midnight
                current_time_struct = time.localtime(current_time)
                current_seconds = current_time_struct.tm_hour * 3600 + current_time_struct.tm_min * 60 + current_time_struct.tm_sec
                
                # Log for debugging
                current_time_str = time.strftime("%H:%M:%S", current_time_struct)
                logging.debug(f"Timer check at {current_time_str}, arm={schedule_arm_time}, disarm={schedule_disarm_time}")
                
                # Calculate if system should be armed now
                should_be_armed = False
                
                if schedule_arm_seconds is not None and schedule_disarm_seconds is not None:
                    if schedule_arm_seconds < schedule_disarm_seconds:
                        # Time window doesn't cross midnight (e.g., 8:00 to 18:00)
                        should_be_armed = (current_seconds >= schedule_arm_seconds and 
                                          current_seconds < schedule_disarm_seconds)
                        logging.debug(f"Regular schedule: {should_be_armed}")
                    else:
                        # Time window crosses midnight (e.g., 18:00 to 8:00)
                        should_be_armed = (current_seconds >= schedule_arm_seconds or 
                                          current_seconds < schedule_disarm_seconds)
                        logging.debug(f"Overnight schedule: {should_be_armed}")
                
                    # Only update if the state needs to change
                    if email_armed != should_be_armed:
                        prev_state = email_armed
                        email_armed = should_be_armed
                        
                        # CRITICAL: Also update detection_active state to match arm state
                        # This is necessary to actually stop motion detection
                        if detection_camera != "disable":  # If detection isn't manually disabled
                            detection_active = email_armed
                        
                        logging.info(f"Timer changed state: {prev_state}->{email_armed} at {current_time_str}")
                        logging.info(f"Detection active also set to: {detection_active}")
                        
                        # Update toggle button if visible
                        if toggle_dialog and toggle_dialog.winfo_exists():
                            toggle_dialog.toggle_button.config(text="Armed" if email_armed else "Disarmed")

        # --- Read Pi Camera Frame (with checks) ---
        pi_frame = None
        ret = False
        if cap and cap.isOpened():
            ret, pi_frame = cap.read()
            # Store raw frame to avoid compound rotation
            if ret and pi_frame is not None:
                last_pi_frame_raw = pi_frame.copy()
            if not ret:
                 logging.warning("Failed to grab Pi camera frame")

                 # Release failed camera
                 if cap:
                     cap.release()
                 cap = None

                 # Pipeline restart logic with exponential backoff
                 if pi_camera_reconnector.should_retry(current_time):
                     logging.info(f"Attempting to reconnect Pi camera...")

                     # Kill and restart pipeline process if it exists
                     if pipeline_process:
                         try:
                             os.killpg(os.getpgid(pipeline_process.pid), signal.SIGTERM)
                             pipeline_process.wait(timeout=2)
                         except (subprocess.TimeoutExpired, ProcessLookupError):
                             try:
                                os.killpg(os.getpgid(pipeline_process.pid), signal.SIGKILL)
                             except ProcessLookupError:
                                 pass  # Process already gone
                         pipeline_process = None

                     # Restart pipeline
                     start_pipeline()
                     time.sleep(2)  # Give pipeline time to start

                     # Try to reopen camera
                     try:
                        cap = cv2.VideoCapture(10, cv2.CAP_V4L2)
                        if cap and cap.isOpened():
                             # Try to read a frame to verify it's working
                             ret_restart, frame_restart = cap.read()
                             if ret_restart and frame_restart is not None:
                                 pi_camera_reconnector.record_attempt(current_time, success=True)
                                 pi_frame = frame_restart
                                 last_pi_frame_raw = pi_frame.copy()
                                 logging.info("Pi camera reconnected and frame captured successfully")
                                 continue  # Process this frame
                             else:
                                 logging.warning("Pi camera opened but failed to capture frame")
                                 cap.release()
                                 cap = None
                                 pi_camera_reconnector.record_attempt(current_time, success=False)
                        else:
                             logging.error("Failed to reopen Pi camera at /dev/video10")
                             cap = None
                             pi_camera_reconnector.record_attempt(current_time, success=False)
                     except Exception as e:
                        logging.error(f"Exception reopening Pi Camera: {e}")
                        cap = None
                        pi_camera_reconnector.record_attempt(current_time, success=False)

                 # Use blank frame if camera still unavailable
                 if cap is None:
                    # Ensure blank frame matches expected dimensions
                    try:
                       h, w, _ = last_pi_frame.shape
                    except (AttributeError, ValueError) as e:
                       logging.debug(f"Could not get shape from last_pi_frame: {e}")
                       h, w = cam1_res_h, cam1_res_w
                    blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    last_pi_frame = blank_frame
                    last_pi_frame_raw = blank_frame.copy()
                    # Continue loop - will retry based on backoff timer

        # Use last valid frame if current read failed but cap exists
        if pi_frame is None:
            pi_frame_to_process = last_pi_frame_raw # Use raw frame to avoid compound rotation
        else:
            # Ensure frame has correct dimensions (might be needed if pipeline restarts with different res?)
            # BUGFIX: Compare to RAW frame, not processed frame (which may be rotated with swapped dimensions)
            if pi_frame.shape[0] != last_pi_frame_raw.shape[0] or pi_frame.shape[1] != last_pi_frame_raw.shape[1]:
                 try:
                     pi_frame = cv2.resize(pi_frame, (last_pi_frame_raw.shape[1], last_pi_frame_raw.shape[0]))
                     logging.warning("Resized incoming Pi frame to match expected dimensions.")
                 except Exception as resize_err:
                     logging.error(f"Failed to resize Pi frame: {resize_err}. Using last good frame.")
                     pi_frame = last_pi_frame # Fallback

            pi_frame_to_process = pi_frame
            # Note: last_pi_frame will be updated after process_frame() with processed version

        # --- Read USB Camera Frame ---
        # multiprocessing.Queue: use get_nowait() which raises queue.Empty if empty
        try:
            usb_frame = usb_frame_queue.get_nowait()
            # Store raw frame to avoid compound rotation when queue is empty
            last_usb_frame_raw = usb_frame.copy() if usb_frame is not None else last_usb_frame_raw
        except queue.Empty:
            # Use last raw frame (not processed/rotated) to avoid compound rotation
            usb_frame = last_usb_frame_raw if last_usb_frame_raw is not None else last_usb_frame

        if usb_frame is not None:
            # Ensure frame has correct dimensions
            # BUGFIX: Compare to RAW frame, not processed frame (which may be rotated with swapped dimensions)
            if last_usb_frame_raw is not None and (usb_frame.shape[0] != last_usb_frame_raw.shape[0] or usb_frame.shape[1] != last_usb_frame_raw.shape[1]):
                 try:
                    usb_frame = cv2.resize(usb_frame, (last_usb_frame_raw.shape[1], last_usb_frame_raw.shape[0]))
                    logging.warning("Resized incoming USB frame to match expected dimensions.")
                 except Exception as resize_err:
                     logging.error(f"Failed to resize USB frame: {resize_err}. Using last good frame.")
                     usb_frame = last_usb_frame_raw if last_usb_frame_raw is not None else last_usb_frame

            usb_frame_to_process = usb_frame
            # Note: last_usb_frame will be updated after process_frame() with processed version
        else:
            usb_frame_to_process = last_usb_frame # Process the last known frame (could be blank initially)

        # --- Handle Rotations ---
        # PYTHON 3.14: Lock-free rotation handling using atomic state
        current_cam1_rotation = cam1_state.get_rotation()
        if current_cam1_rotation != last_cam1_rotation:
            pi_previous_frame = None  # Reset for frame differencing
            bg_subtractor_cam1 = cv2.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=400.0, history=500)  # Reset background subtractor
            knn_frame_counter_cam1 = 0  # Reset frame counter for new warmup period
            last_cam1_rotation = current_cam1_rotation
            time.sleep(0.05)
            # PYTHON 3.14: Clear Pi frame queue for web feed
            while frame_queue.get() is not None:
                pass  # Drain the queue
            # Read a fresh frame directly from camera to update raw buffer
            if cap and cap.isOpened():
                ret_fresh, fresh_pi_frame = cap.read()
                if ret_fresh and fresh_pi_frame is not None:
                    last_pi_frame_raw = fresh_pi_frame.copy()
                    logging.debug(f"Main: Got fresh Pi frame after rotation change")
            logging.debug(f"Main: Reset Pi cam state, rotation now {current_cam1_rotation}")

        # Handle cam2 rotation - drain buffered frames immediately
        current_cam2_rotation = 0
        with cam2_rotation.get_lock():
            current_cam2_rotation = cam2_rotation.value
        if current_cam2_rotation != last_cam2_rotation:
            usb_previous_frame = None  # Reset for frame differencing
            bg_subtractor_cam2 = cv2.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=400.0, history=500)  # Reset background subtractor
            knn_frame_counter_cam2 = 0  # Reset frame counter for new warmup period
            last_cam2_rotation = current_cam2_rotation
            # Drain the USB queue to clear old pre-rotation frames
            drained_count = 0
            try:
                while True:
                    usb_frame_queue.get_nowait()
                    drained_count += 1
            except queue.Empty:
                pass
            # Wait for a fresh frame from the queue (timeout after 1 second)
            try:
                fresh_frame = usb_frame_queue.get(timeout=1.0)
                if fresh_frame is not None:
                    last_usb_frame_raw = fresh_frame.copy()
                    logging.debug(f"Main: Got fresh frame after rotation change")
            except queue.Empty:
                logging.warning(f"Main: Timeout waiting for fresh USB frame after rotation")
            logging.debug(f"Main: Reset USB cam state, rotation now {current_cam2_rotation}, drained {drained_count} old frames from queue")

        # Apply rotation BEFORE processing for Pi Camera
        rotated_pi_frame = apply_rotation(pi_frame_to_process, current_cam1_rotation)
        # Apply rotation for USB Camera
        rotated_usb_frame = apply_rotation(usb_frame_to_process, current_cam2_rotation)

        # --- Store Original Frames for YOLO ---
        # IMPORTANT: Store the original full frames BEFORE processing
        # YOLO needs to analyze the complete frame, not just the ROI
        original_pi_frame = rotated_pi_frame.copy() if rotated_pi_frame is not None else None
        original_usb_frame = rotated_usb_frame.copy() if rotated_usb_frame is not None else None

        # --- Process Frames ---
        # Process Camera 1 (Pi Camera) if available
        processed_pi_frame, pi_detected, pi_previous_frame = process_frame(
            rotated_pi_frame,
            bg_subtractor_cam1,
            pi_previous_frame,
            "cam1",
            use_bg_subtraction=use_background_subtraction
        )

        # Process Camera 2 (USB) if available
        processed_usb_frame, usb_detected, usb_previous_frame = process_frame(
            rotated_usb_frame,
            bg_subtractor_cam2,
            usb_previous_frame,
            "cam2",
            use_bg_subtraction=use_background_subtraction
        )
        
        # Determine if motion is detected by either camera
        raw_motion_detected = pi_detected or usb_detected

        # With async YOLO, we just need to detect motion to trigger the confirmation phase
        # YOLO analysis will happen asynchronously after motion confirmation
        # Note: pi_yolo_objects and usb_yolo_objects will always be False now since
        # we removed YOLO from process_frame() - YOLO runs asynchronously instead
        
        if raw_motion_detected and not detection_active:
            logging.debug("Motion detected but system is disarmed - ignoring")
        
        # --- Update last frames for display/queues ---
        # Use processed frames which include ROI/motion boxes
        # Note: processed frames are already copies from process_frame(), no need to copy again
        # OPTIMIZATION FIX: Always update display frames unconditionally to prevent freeze
        # YOLO annotated frames will be preferred at display time (see display logic below)
        if processed_pi_frame is not None:
            last_pi_frame = processed_pi_frame
        if processed_usb_frame is not None:
            last_usb_frame = processed_usb_frame

        # Logging for Camera 2 motion
        if usb_detected and detection_camera in ["cam2", "both"]:
             # Reduce log frequency
             if random.random() < 0.1: logging.info(f"Camera 2 detected motion in '{detection_camera}' mode")
             if detection_camera == "both" and not pi_detected:
                 motion_source_camera = "cam2"
                 if random.random() < 0.1: logging.info("Setting motion_source_camera to 'cam2' (only Cam2 detected)")

        # --- Combine Frames for Display ---
        window_width, window_height = screen_width, screen_height

        # Show YOLO annotated frames during YOLO analysis AND recording
        # Only revert to live frames after recording completes
        yolo_or_rec = state_manager.state in [DetectionState.YOLO_ROI, DetectionState.YOLO_FULL, DetectionState.RECORDING]
        if yolo_or_rec and yolo_annotated_pi_frame is not None:
            display_pi_frame = yolo_annotated_pi_frame
        else:
            display_pi_frame = last_pi_frame

        yolo_or_rec = state_manager.state in [DetectionState.YOLO_ROI, DetectionState.YOLO_FULL, DetectionState.RECORDING]
        if yolo_or_rec and yolo_annotated_usb_frame is not None:
            display_usb_frame = yolo_annotated_usb_frame
        else:
            display_usb_frame = last_usb_frame

        # Check actual camera hardware availability, not frame content
        picam_valid = cap is not None and (cap.isOpened() if hasattr(cap, 'isOpened') else False)
        usb_valid = cam2_available.value if 'cam2_available' in globals() else False

        try:
            # Handle cases where one camera might be invalid (e.g., Pi cam failed)
            if picam_valid and not usb_valid:
                combined_frame = resize_and_pad(display_pi_frame, window_width, window_height)
            elif not picam_valid and usb_valid:
                combined_frame = resize_and_pad(display_usb_frame, window_width, window_height)
            elif picam_valid and usb_valid: # Both valid
                target_width_per_frame = window_width // 2
                processed_pi_frame_resized = resize_and_pad(display_pi_frame, target_width_per_frame, window_height)
                usb_frame_resized = resize_and_pad(display_usb_frame, target_width_per_frame, window_height)
                combined_frame = np.hstack((processed_pi_frame_resized, usb_frame_resized))
            else: # Neither valid (both potentially blank)
                 combined_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
                 # Optional: Add text indicating no camera feed
                 # cv2.putText(combined_frame, "No Camera Feed Available", (50, window_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        except Exception as e:
            logging.error(f"Local stacking error: {e}")
            combined_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # DEBUG: Log frame info
        if random.random() < 0.1:  # Log 10% of the time to avoid spam
            logging.debug(f"Display: picam_valid={picam_valid}, usb_valid={usb_valid}, combined_frame shape={combined_frame.shape if combined_frame is not None else 'None'}, all_zero={np.all(combined_frame == 0) if combined_frame is not None else 'N/A'}")

        # Add timer information to the display
        # Either when timer is enabled or when it was recently disabled (to show notification)
        add_timer_info = timer_enabled
        
        # If timer is not enabled but was recently disabled, show a temporary notification
        if not timer_enabled and hasattr(root, 'timer_recently_disabled'):
            current_time = time.time()
            # Show "TIMER DISABLED" for 10 seconds after disabling timer
            if current_time - root.timer_recently_disabled < 10:
                add_timer_info = True  # Show the timer info for a few seconds
            else:
                # Clear the timer_recently_disabled attribute after 10 seconds
                delattr(root, 'timer_recently_disabled')
        
        if add_timer_info:
            combined_frame = add_timer_info_to_frame(combined_frame)

        cv2.imshow("Camera Feeds", combined_frame)

        # Track FPS after actual display update
        fps_frame_count += 1
        if current_time - fps_start_time >= fps_update_interval:
            fps = fps_frame_count / (current_time - fps_start_time)
            logging.debug(f"Display FPS: {fps:.1f}")
            fps_frame_count = 0
            fps_start_time = current_time

        # --- Update Web Frame Queue ---
        # PYTHON 3.14: LockFreeQueue.put() is non-blocking, drops oldest if full
        if processed_pi_frame is not None and processed_pi_frame.size > 0:
            frame_queue.put(processed_pi_frame)

        # --- Motion Detection Logic with NEW ARCHITECTURE State Machine ---
        current_time = time.time()
        
        if not streaming_active and detection_active:
            # Handle motion detection with proper state machine
            if raw_motion_detected:
                # Track detection times for motion source determination (ONLY when IDLE)
                if state_manager.state == DetectionState.IDLE:
                    if pi_detected and motion_detection_time_cam1 is None:
                        motion_detection_time_cam1 = current_time
                        logging.debug(f"Cam1 first detect time: {motion_detection_time_cam1}")
                    if usb_detected and motion_detection_time_cam2 is None:
                        motion_detection_time_cam2 = current_time
                        logging.debug(f"Cam2 first detect time: {motion_detection_time_cam2}")
                    
                    # Determine motion source camera ONCE when first detecting motion
                    determined_source = "cam1"  # Default
                    if detection_camera == "both":
                        if usb_detected and motion_detection_time_cam2 is not None and \
                           (motion_detection_time_cam1 is None or motion_detection_time_cam2 < motion_detection_time_cam1 - 0.05):
                            determined_source = "cam2"
                        elif usb_detected and not pi_detected:
                            determined_source = "cam2"
                    elif detection_camera == "cam1":
                        determined_source = "cam1"
                    elif detection_camera == "cam2":
                        determined_source = "cam2"
                    
                    # Motion detected - start confirmation and lock source camera
                    state_manager.transition_to(DetectionState.MOTION_DETECTED)
                    state_manager.source_camera = determined_source
                    logging.debug(f"State machine: IDLE -> MOTION_DETECTED (source: {determined_source})")
                
                elif state_manager.state == DetectionState.MOTION_DETECTED:
                    # Check if 1 second of motion has passed for confirmation
                    if current_time - state_manager.motion_start_time >= 1.0:
                        trigger_camera = "Camera 1" if state_manager.source_camera == "cam1" else "Camera 2"
                        
                        if config.get('use_yolo_detection', False):
                            # Transition to YOLO ROI analysis
                            state_manager.transition_to(DetectionState.MOTION_CONFIRMED)
                            state_manager.transition_to(DetectionState.YOLO_ROI)
                            logging.info(f"Motion confirmed on {trigger_camera}, starting ROI YOLO analysis")
                            
                            # Clear old YOLO annotations
                            yolo_annotated_pi_frame = None
                            yolo_annotated_usb_frame = None
                        else:
                            # No YOLO - proceed directly to recording
                            logging.info(f"Motion confirmed on {trigger_camera}, proceeding to capture (YOLO disabled)")
                            state_manager.transition_to(DetectionState.RECORDING)
            
            else:
                # No motion - check if state machine needs timeout
                state_manager.check_timeout(current_time)
                
                
                # OLD LOGIC: Reset detection_check_start after 3 seconds of no motion
                # This is now handled by state_manager.check_timeout()
                if detection_check_start is not None and current_time - detection_check_start >= 3:
                    logging.debug("Motion confirmation timeout - resetting (3s no motion)")
                    detection_check_start = None
        # --- NEW ARCHITECTURE: YOLO Analysis with YOLOManager ---
        current_time = time.time()
        
        # Check state machine timeout (handles ROI and full frame timeouts automatically)
        state_manager.check_timeout(current_time)
        
        # Submit frames for YOLO analysis based on current state
        if state_manager.state == DetectionState.YOLO_ROI:
            # Submit ROI for analysis (throttled by yolo_manager)
            if yolo_manager.can_submit(current_time):
                camera = state_manager.source_camera
                
                # Get the appropriate frame
                if camera == "cam1" and original_pi_frame is not None:
                    frame = original_pi_frame
                elif camera == "cam2" and original_usb_frame is not None:
                    frame = original_usb_frame
                else:
                    frame = None
                
                if frame is not None:
                    # Get ROI coordinates
                    roi_key = "roi_coordinates_cam2" if camera == "cam2" else "roi_coordinates"
                    roi = config.get(roi_key, [0, 0, 640, 480])
                    
                    if isinstance(roi, list) and len(roi) == 4:
                        roi_coords = tuple(roi)
                    else:
                        roi_coords = (roi.get('x', 0), roi.get('y', 0), 
                                     roi.get('width', 640), roi.get('height', 480))
                    
                    yolo_manager.submit_roi(frame, camera, roi_coords, current_time)
        
        elif state_manager.state == DetectionState.YOLO_FULL:
            # Submit full frame for verification
            camera = state_manager.source_camera
            
            # Get the appropriate frame
            if camera == "cam1" and original_pi_frame is not None:
                frame = original_pi_frame
            elif camera == "cam2" and original_usb_frame is not None:
                frame = original_usb_frame
            else:
                frame = None
            
            if frame is not None:
                yolo_manager.submit_full_frame(frame, camera, current_time)
        
        # Process YOLO results
        if state_manager.state in [DetectionState.YOLO_ROI, DetectionState.YOLO_FULL]:
            result = yolo_manager.process_results(state_manager)
            
            if result:
                stage, contains_objects, annotated_frame = result
                camera = state_manager.source_camera
                
                # Store annotated frame
                if annotated_frame is not None:
                    if camera == "cam1":
                        yolo_annotated_pi_frame = annotated_frame
                    else:
                        yolo_annotated_usb_frame = annotated_frame
                
                # Handle state transitions based on result
                if stage == "roi" and contains_objects:
                    # Person detected in ROI - move to full frame verification
                    state_manager.transition_to(DetectionState.YOLO_FULL)
                
                elif stage == "full_frame" and contains_objects:
                    # Person confirmed in full frame - start recording
                    state_manager.transition_to(DetectionState.RECORDING)
                    # Mark detection as inactive (will record now)
                    detection_active = False
                
                elif stage == "full_frame" and not contains_objects:
                    # Verification failed - reset to idle
                    state_manager.transition_to(DetectionState.IDLE)
                    detection_active = True

        # --- NEW ARCHITECTURE: Recording with RecordingManager ---
        if state_manager.state == DetectionState.RECORDING:
            current_time = time.time()
            
            # Start recording if not active
            if not recording_manager.active:
                camera = state_manager.source_camera
                
                # Get the best available frame (YOLO-annotated preferred)
                if camera == "cam1":
                    frame = yolo_annotated_pi_frame if yolo_annotated_pi_frame is not None else processed_pi_frame
                else:
                    frame = yolo_annotated_usb_frame if yolo_annotated_usb_frame is not None else processed_usb_frame
                
                # Validate frame before starting
                if frame is None or frame.size == 0 or np.all(frame == 0):
                    logging.warning(f"Invalid frame from {camera} for recording. Resetting to IDLE.")
                    state_manager.transition_to(DetectionState.IDLE)
                    detection_active = True
                else:
                    # Get resolution for this camera
                    cam_res_key = "cam1_resolution" if camera == "cam1" else "cam2_resolution"
                    default_res = "640x480" if camera == "cam1" else "320x240"
                    res_str = config.get(cam_res_key, default_res)
                    
                    try:
                        w, h = map(int, res_str.split('x'))
                        # Sanity check
                        if not (100 < w < 4000 and 100 < h < 3000):
                            raise ValueError("Unreasonable resolution")
                        resolution = (w, h)
                    except Exception as e:
                        logging.warning(f"Invalid resolution '{res_str}': {e}. Using {default_res}")
                        w, h = map(int, default_res.split('x'))
                        resolution = (w, h)
                    
                    # Start recording
                    if recording_manager.start(frame, camera, resolution, current_time):
                        logging.info(f"Started recording from {camera} at {resolution}")
                    else:
                        logging.error("Failed to start recording. Resetting to IDLE.")
                        state_manager.transition_to(DetectionState.IDLE)
                        detection_active = True
            
            else:
                # Recording active - write frames
                camera = recording_manager.source_camera
                frame_to_record = last_pi_frame if camera == "cam1" else last_usb_frame
                
                if frame_to_record is not None and frame_to_record.size > 0:
                    recording_manager.write_frame(frame_to_record)
                
                # Check if recording should stop
                if recording_manager.should_stop(current_time):
                    video_path = recording_manager.stop()
                    
                    if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 100:
                        detected_image = recording_manager.detected_frame
                        
                        if detected_image is not None and detected_image.size > 0:
                            # Send email or save to storage
                            if email_armed and not cooldown_active:
                                email_trigger_source = "Camera 1" if camera == "cam1" else "Camera 2"
                                logging.info(f"Sending email triggered by {email_trigger_source}")
                                
                                email_thread = threading.Thread(
                                    target=send_email_in_thread,
                                    args=(detected_image.copy(), video_path, sender_email, receiver_emails, email_password, camera),
                                    daemon=True
                                )
                                email_thread.start()
                            else:
                                reason = "email not armed" if not email_armed else "cooldown active"
                                logging.info(f"Saving media to storage ({reason})")
                                save_media_to_storage(detected_image.copy(), video_path, camera)
                            
                            # Schedule video file deletion after email/save completes
                            threading.Timer(2.0, lambda p=video_path: safe_unlink(p)).start()
                        else:
                            logging.warning("Detected image is invalid. Cannot send/save.")
                            # Still cleanup video
                            threading.Timer(2.0, lambda p=video_path: safe_unlink(p)).start()
                    else:
                        logging.warning(f"Recording failed - invalid video file: {video_path}")
                    
                    # Transition to cooldown
                    state_manager.transition_to(DetectionState.COOLDOWN)
                    detection_active = True
                    
                    # Clear YOLO annotations
                    yolo_annotated_pi_frame = None
                    yolo_annotated_usb_frame = None
                    
                    # Reset motion times
                    motion_detection_time_cam1 = None
                    motion_detection_time_cam2 = None
                    
                    logging.debug("Recording complete, transitioned to COOLDOWN")


        # --- UI Updates and Key Handling (largely unchanged) ---
        # Ensure dialogs exist before updating
        try:
             if toggle_dialog and toggle_dialog.winfo_exists(): toggle_dialog.update()
             if settings_dialog and settings_dialog.winfo_exists(): settings_dialog.update()
             if root: root.update_idletasks()
        except tk.TclError as e:
             if "application has been destroyed" in str(e):
                  logging.warning("Tkinter root window destroyed, cannot update UI.")
             else:
                  logging.error(f"Tkinter error during UI update: {e}")
        except Exception as e:
             logging.error(f"Unexpected error during UI update: {e}")


        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            # Try to destroy UI elements gracefully
            try:
                if toggle_dialog and toggle_dialog.winfo_exists(): toggle_dialog.destroy()
            except (tk.TclError, AttributeError) as e:
                logging.debug(f"Could not destroy toggle_dialog: {e}")
                pass
            try:
                if settings_dialog and settings_dialog.winfo_exists(): settings_dialog.destroy()
            except (tk.TclError, AttributeError) as e:
                logging.debug(f"Could not destroy settings_dialog: {e}")
                pass
            try:
                if root: root.destroy()
            except (tk.TclError, AttributeError) as e:
                logging.debug(f"Could not destroy root: {e}")
                pass
            break # Exit main loop
        elif key == ord('s'):
             if settings_dialog: settings_dialog.show()
             # Reload config potentially changed by settings dialog (existing logic)
             # Be careful if settings change flask host/port while running
             authorized_users.update({user["email"]: user["password_hash"].encode('utf-8') for user in config.get("users", [])})
             sender_email = config.get('sender_email','')
             receiver_emails = config.get('receiver_emails',[])
             email_password = config.get('email_password','')
        elif key == ord('r'):
            if roi_selection_mode: deactivate_roi_selection()
            else: activate_roi_selection()
        elif key == ord('b'): # Add a key to toggle detection method for testing
            use_background_subtraction = not use_background_subtraction
            logging.info(f"Switched motion detection method to: {'Background Subtraction' if use_background_subtraction else 'Frame Differencing'}")
            # Reset state when switching methods to avoid interference
            pi_previous_frame = None
            usb_previous_frame = None
            bg_subtractor_cam1 = cv2.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=400.0, history=500)
            bg_subtractor_cam2 = cv2.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=400.0, history=500)
            knn_frame_counter_cam1 = 0  # Reset frame counters for new warmup periods
            knn_frame_counter_cam2 = 0
            motion_detection_time_cam1 = None
            motion_detection_time_cam2 = None
            detection_check_start = None


        # --- Handle Pending ROI/UI Actions (largely unchanged) ---
        if pending_roi_activation is not None:
             # Wrap in try-except for safety
             try:
                 camera = pending_roi_activation["camera"]
                 is_second_camera = pending_roi_activation.get("is_second_camera", False)
                 old_camera = detection_camera
                 detection_camera = camera
                 if is_second_camera and settings_dialog and settings_dialog.winfo_exists():
                     settings_dialog.roi_status_label.config(text="Now select ROI for Camera 2")
                     settings_dialog.roi_sequence_both = True
                 pending_roi_activation = None # Clear before activation
                 if is_second_camera:
                     roi_selection_mode = "both_cameras_cam2"
                     activate_roi_selection(is_both_cameras=False)
                     if settings_dialog: settings_dialog.original_detection_camera = "both"
                 else:
                     activate_roi_selection()
                 continue # Skip rest of loop to avoid interference
             except Exception as e:
                 logging.error(f"Error handling pending ROI activation: {e}")
                 pending_roi_activation = None # Clear on error
             
        if pending_ui_update is not None:
             # Wrap in try-except for safety
             try:
                 action = pending_ui_update["action"]
                 if action == "reset_roi_ui" and settings_dialog and settings_dialog.winfo_exists():
                    settings_dialog.roi_button.config(text="Select ROI")
                    settings_dialog.roi_status_label.pack_forget()
                    settings_dialog.update_roi_camera_label()
                    if hasattr(settings_dialog, 'roi_sequence_both') and settings_dialog.roi_sequence_both:
                        detection_camera = "both"
                        config["detection_camera"] = "both"
                        settings_dialog.detection_camera.set("both")
                        settings_dialog.roi_sequence_both = False
                        if toggle_dialog and toggle_dialog.winfo_exists():
                            detection_camera = "both" # Update global too
                        try:
                            with open(CONFIG_FILE, 'w') as f: json.dump(config, f, indent=4)
                        except Exception as e: logging.error(f"Failed to save updated detection_camera setting: {e}")
                 pending_ui_update = None # Clear after handling
             except Exception as e:
                 logging.error(f"Error handling pending UI update: {e}")
                 pending_ui_update = None # Clear on error

    cleanup() # Call cleanup when loop breaks

def safe_unlink(filepath):
    """Attempt to unlink a file, logging errors."""
    try:
        if filepath and os.path.exists(filepath):
            os.unlink(filepath)
            logging.debug(f"Successfully unlinked {filepath}")
        elif filepath:
             logging.debug(f"File already unlinked or never existed: {filepath}")
    except OSError as e:
        logging.error(f"Error unlinking file {filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error unlinking file {filepath}: {e}")


def mouse_callback(event, x, y, flags, param):
    global roi_selection_mode, roi_start_point, roi_end_point, roi_drawing, roi_temp, config
    # Unpack parameters: frame, settings_dialog, toggle_dialog
    try:
        param_frame, settings_dialog, toggle_dialog = param
    except (TypeError, ValueError):
        # Fallback or error if param is not the expected tuple (e.g., old callback was still set)
        logging.error("Mouse callback received unexpected parameters.")
        # Attempt to restore default callback
        try:
             cv2.setMouseCallback("Camera Feeds", globals().get('toggle_visibility'), globals().get('toggle_dialog'))
        except Exception as cb_err:
             logging.error(f"Failed to restore default mouse callback: {cb_err}")
        return
        
    # logging.debug(f"mouse_callback triggered: event={event}, x={x}, y={y}, roi_selection_mode={roi_selection_mode}")
    try:
        if not roi_selection_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            logging.debug("ROI drawing started")
            roi_drawing = True
            roi_start_point = (x, y)
            roi_temp = param_frame.copy() if param_frame is not None else None
        elif event == cv2.EVENT_MOUSEMOVE and roi_drawing:
            logging.debug("ROI drawing updated")
            roi_end_point = (x, y)
            if roi_temp is not None:
                display_frame = roi_temp.copy()
                cv2.rectangle(display_frame, roi_start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Camera Feeds", display_frame)
        elif event == cv2.EVENT_LBUTTONUP and roi_drawing:
            logging.debug("ROI drawing finished")
            roi_drawing = False
            roi_end_point = (x, y)
            x1, y1 = roi_start_point
            x2, y2 = roi_end_point
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            width = x2 - x1
            height = y2 - y1

            MIN_ROI_SIZE = 5
            if width < MIN_ROI_SIZE or height < MIN_ROI_SIZE:
                logging.warning(f"ROI selection too small ({width}x{height}). Not saving.")
                roi_selection_mode = False
                if settings_dialog: # Check if dialog object is valid
                    settings_dialog.roi_button.config(text="Select ROI")
                    settings_dialog.roi_status_label.pack_forget()
                logging.debug("Setting mouse callback back to toggle_visibility after small ROI")
                cv2.setMouseCallback("Camera Feeds", toggle_visibility, toggle_dialog)
                if roi_temp is not None:
                   cv2.imshow("Camera Feeds", roi_temp) 
                return

            config["roi_coordinates"] = [x1, y1, width, height]
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            
            logging.info(f"ROI updated and saved: x={x1}, y={y1}, width={width}, height={height}")
            roi_selection_mode = False

            if settings_dialog: # Check if dialog object is valid
                settings_dialog.roi_button.config(text="Select ROI")
                settings_dialog.roi_status_label.pack_forget()

            logging.debug("Setting mouse callback back to toggle_visibility after successful ROI selection")
            cv2.setMouseCallback("Camera Feeds", toggle_visibility, toggle_dialog)
            
    except Exception as e:
        logging.error(f"Error in mouse_callback: {e}")
        roi_selection_mode = False
        try:
            if settings_dialog: # Check if dialog object is valid
                 settings_dialog.roi_button.config(text="Select ROI")
                 settings_dialog.roi_status_label.pack_forget()
            logging.debug("Setting mouse callback back to toggle_visibility after mouse_callback error")
            cv2.setMouseCallback("Camera Feeds", toggle_visibility, toggle_dialog)
        except Exception as ui_e:
            logging.error(f"Error resetting UI/callback after mouse_callback error: {ui_e}")

def add_timer_info_to_frame(frame):
    """Add timer information to the upper right corner of frame if timer is enabled"""
    # Modify frame in-place for better performance (frame is already a copy in calling code)
    display_frame = frame

    # Get current time
    current_time_struct = time.localtime()
    current_time_str = time.strftime("%H:%M:%S", current_time_struct)
    current_seconds = current_time_struct.tm_hour * 3600 + current_time_struct.tm_min * 60 + current_time_struct.tm_sec
    
    # Different display based on timer status
    if timer_enabled:
        # Determine current arm state based on schedule
        if schedule_arm_seconds is not None and schedule_disarm_seconds is not None:
            # Calculate what the scheduled state should be right now
            scheduled_armed = False
            
            # Determine if we should be armed based on time
            if schedule_arm_seconds < schedule_disarm_seconds:
                # Time window doesn't cross midnight (e.g., 8:00 to 18:00)
                scheduled_armed = schedule_arm_seconds <= current_seconds < schedule_disarm_seconds
            else:
                # Time window crosses midnight (e.g., 18:00 to 8:00)
                scheduled_armed = current_seconds >= schedule_arm_seconds or current_seconds < schedule_disarm_seconds
            
            # Set status text based on ACTUAL state (email_armed), not scheduled state
            status_text = "ARMED" if email_armed else "DISARMED"
            font_color = (0, 255, 0) if email_armed else (0, 0, 255)  # Green for armed, Red for disarmed
            
            # Add indication for manual override if actual state differs from scheduled state
            if scheduled_armed != email_armed:
                status_text += " (Manual Override)"
                
            # Add debug text
            debug_text = f"Schedule: {'ARM' if scheduled_armed else 'DISARM'}"
        else:
            # Fallback if schedule not properly set
            status_text = "ARMED" if email_armed else "DISARMED"
            font_color = (0, 255, 0) if email_armed else (0, 0, 255)
            debug_text = "Schedule times not set"
        
        # Prepare text to display
        time_text = f"Current Time: {current_time_str}"
        schedule_text = f"Arm: {schedule_arm_time} | Disarm: {schedule_disarm_time}"
        status_color = font_color
    else:
        # Timer is disabled - show a simple status
        time_text = f"Current Time: {current_time_str}"
        status_text = "TIMER DISABLED"
        font_color = (0, 165, 255)  # Orange/amber color
        schedule_text = f"Detection is {'ARMED' if detection_active else 'DISABLED'}"
        debug_text = "Timer scheduling not active"
        status_color = font_color
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # Get text size to position correctly
    time_size = cv2.getTextSize(time_text, font, font_scale, font_thickness)[0]
    status_size = cv2.getTextSize(status_text, font, font_scale, font_thickness)[0]
    schedule_size = cv2.getTextSize(schedule_text, font, font_scale, font_thickness)[0]
    debug_size = cv2.getTextSize(debug_text, font, font_scale, font_thickness)[0]
    
    # Find the widest text for rectangle sizing
    max_width = max(time_size[0], schedule_size[0], status_size[0], debug_size[0])
    
    # Calculate positions (upper right with padding)
    padding = 10
    time_position = (display_frame.shape[1] - time_size[0] - padding, padding + time_size[1])
    status_position = (display_frame.shape[1] - status_size[0] - padding, padding + time_size[1] + status_size[1] + 10)
    schedule_position = (display_frame.shape[1] - schedule_size[0] - padding, padding + time_size[1] + status_size[1] + schedule_size[1] + 20)
    debug_position = (display_frame.shape[1] - debug_size[0] - padding, padding + time_size[1] + status_size[1] + schedule_size[1] + debug_size[1] + 30)
    
    # Add background rectangle for better readability
    rect_height = padding + time_size[1] + status_size[1] + schedule_size[1] + debug_size[1] + 40
    cv2.rectangle(
        display_frame, 
        (display_frame.shape[1] - max_width - padding*2, padding - 5), 
        (display_frame.shape[1] - padding, rect_height), 
        (0, 0, 0), 
        -1
    )
    
    # Add text to image
    cv2.putText(display_frame, time_text, time_position, font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(display_frame, status_text, status_position, font, font_scale, status_color, font_thickness)
    cv2.putText(display_frame, schedule_text, schedule_position, font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(display_frame, debug_text, debug_position, font, font_scale, (200, 200, 0), font_thickness)
    
    return display_frame

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    main()
