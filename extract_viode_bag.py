#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rosbag


THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")


IMAGE_MSG_TYPES = {
    "sensor_msgs/Image",
    "sensor_msgs/CompressedImage",
}
IMU_MSG_TYPES = {
    "sensor_msgs/Imu",
}
POSE_MSG_TYPES = {
    "geometry_msgs/PoseStamped",
    "geometry_msgs/PoseWithCovarianceStamped",
    "nav_msgs/Odometry",
}


def stamp_to_sec(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def sanitize_topic_name(topic: str) -> str:
    return topic.strip("/").replace("/", "_") or "root"


def infer_dtype_channels(encoding: str) -> Tuple[np.dtype, int]:
    enc = str(encoding).strip().lower()
    mapping = {
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "16uc1": (np.uint16, 1),
        "16sc1": (np.int16, 1),
        "bgr8": (np.uint8, 3),
        "rgb8": (np.uint8, 3),
        "bgra8": (np.uint8, 4),
        "rgba8": (np.uint8, 4),
        "8uc3": (np.uint8, 3),
        "8uc4": (np.uint8, 4),
    }
    if enc in mapping:
        return mapping[enc]

    m = re.fullmatch(r"(8|16)(u|s)c([1-4])", enc)
    if m:
        bits = int(m.group(1))
        sign = m.group(2)
        channels = int(m.group(3))
        if bits == 8 and sign == "u":
            return np.uint8, channels
        if bits == 16 and sign == "u":
            return np.uint16, channels
        if bits == 16 and sign == "s":
            return np.int16, channels

    raise ValueError(f"Unsupported image encoding: {encoding}")


def decode_ros_image(msg) -> np.ndarray:
    if msg._type == "sensor_msgs/CompressedImage":
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cv2.imdecode failed for compressed image")
        return img

    if msg._type != "sensor_msgs/Image":
        raise ValueError(f"Unsupported image msg type: {msg._type}")

    dtype, channels = infer_dtype_channels(msg.encoding)
    itemsize = np.dtype(dtype).itemsize
    elems_per_row = int(msg.step) // itemsize
    useful_elems = int(msg.width) * channels
    buf = np.frombuffer(msg.data, dtype=dtype)
    arr = buf.reshape((int(msg.height), elems_per_row))[:, :useful_elems]

    if channels == 1:
        img = arr.reshape((int(msg.height), int(msg.width)))
    else:
        img = arr.reshape((int(msg.height), int(msg.width), channels))

    if int(getattr(msg, "is_bigendian", 0)) and itemsize > 1:
        img = img.byteswap().newbyteorder()

    enc = str(msg.encoding).strip().lower()
    if enc == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc == "rgba8":
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    return img


def choose_topic(
    info_topics: Dict[str, object],
    wanted_types: Iterable[str],
    explicit_topic: str,
    kind: str,
) -> Optional[str]:
    if explicit_topic:
        if explicit_topic not in info_topics:
            raise KeyError(f"{kind} topic not found in bag: {explicit_topic}")
        return explicit_topic

    candidates = []
    wanted = set(wanted_types)
    for topic, meta in info_topics.items():
        msg_type = meta["msg_type"] if isinstance(meta, dict) else meta.msg_type
        message_count = int(meta["message_count"]) if isinstance(meta, dict) else int(meta.message_count)
        if msg_type not in wanted:
            continue
        score = 0
        low = topic.lower()
        if kind == "image":
            if "cam0" in low or "camera0" in low or "left" in low:
                score += 8
            if "image" in low or "rgb" in low:
                score += 5
            if "compressed" not in low:
                score += 2
            if any(bad in low for bad in ("depth", "mask", "seg", "label", "flow")):
                score -= 10
        elif kind == "imu":
            if "imu" in low:
                score += 10
            if "raw" in low:
                score += 2
        elif kind == "pose":
            if "ground" in low or "gt" in low:
                score += 10
            if "odom" in low or "pose" in low:
                score += 6
            if "vio" in low or "estimate" in low:
                score -= 4
        candidates.append((score, message_count, topic))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][2]


def list_topics(bag_path: str) -> Dict[str, dict]:
    with rosbag.Bag(bag_path, "r") as bag:
        info = bag.get_type_and_topic_info()
        topics = {}
        for topic, meta in sorted(info.topics.items()):
            topics[topic] = {
                "msg_type": meta.msg_type,
                "message_count": int(meta.message_count),
                "frequency": None if meta.frequency is None else float(meta.frequency),
                "connections": int(meta.connections),
            }
        return topics


def extract_pose_row(msg, t_ros) -> Optional[dict]:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    t_msg = stamp_to_sec(stamp) if stamp is not None else stamp_to_sec(t_ros)

    if msg._type == "geometry_msgs/PoseStamped":
        pose = msg.pose
    elif msg._type == "geometry_msgs/PoseWithCovarianceStamped":
        pose = msg.pose.pose
    elif msg._type == "nav_msgs/Odometry":
        pose = msg.pose.pose
    else:
        return None

    return {
        "stamp_sec": f"{t_msg:.9f}",
        "px": float(pose.position.x),
        "py": float(pose.position.y),
        "pz": float(pose.position.z),
        "qx": float(pose.orientation.x),
        "qy": float(pose.orientation.y),
        "qz": float(pose.orientation.z),
        "qw": float(pose.orientation.w),
    }


def extract_imu_row(msg, t_ros) -> Optional[dict]:
    if msg._type != "sensor_msgs/Imu":
        return None
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    t_msg = stamp_to_sec(stamp) if stamp is not None else stamp_to_sec(t_ros)
    return {
        "stamp_sec": f"{t_msg:.9f}",
        "gx": float(msg.angular_velocity.x),
        "gy": float(msg.angular_velocity.y),
        "gz": float(msg.angular_velocity.z),
        "ax": float(msg.linear_acceleration.x),
        "ay": float(msg.linear_acceleration.y),
        "az": float(msg.linear_acceleration.z),
    }


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_output_root(args, bag_path: Path) -> Path:
    seq_name = args.seq_name.strip() if args.seq_name else bag_path.stem
    return Path(args.out_root).resolve() / seq_name


def main():
    ap = argparse.ArgumentParser(
        description="Extract a ROS1 bag into image sequence / CSV sidecars for VIODE-style experiments."
    )
    ap.add_argument("--bag", required=True, help="Absolute path to a ROS1 .bag file.")
    ap.add_argument("--out_root", required=True, help="Directory to store extracted outputs.")
    ap.add_argument("--seq_name", default="", help="Optional output sequence name. Default: bag stem.")
    ap.add_argument("--image_topic", default="", help="Explicit image topic. Auto-detect if empty.")
    ap.add_argument("--imu_topic", default="", help="Explicit IMU topic. Auto-detect if empty.")
    ap.add_argument("--pose_topic", default="", help="Explicit GT/pose topic. Auto-detect if empty.")
    ap.add_argument("--img_subdir", default="rgb", help="Image subdir name under output sequence.")
    ap.add_argument("--img_ext", default=".png", choices=[".png", ".jpg"], help="Saved image extension.")
    ap.add_argument("--every_n", type=int, default=1, help="Keep one frame every N image messages.")
    ap.add_argument("--max_frames", type=int, default=-1, help="Stop after extracting this many image frames.")
    ap.add_argument("--start_sec", type=float, default=0.0, help="Start from this relative bag time in seconds.")
    ap.add_argument("--end_sec", type=float, default=-1.0, help="Stop at this relative bag time in seconds.")
    ap.add_argument("--list_topics", action="store_true", help="Only print topic summary, then exit.")
    ap.add_argument("--skip_imu", action="store_true", help="Do not export IMU CSV.")
    ap.add_argument("--skip_pose", action="store_true", help="Do not export pose CSV.")
    args = ap.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")
    if args.every_n <= 0:
        raise ValueError("--every_n must be >= 1")

    topics = list_topics(str(bag_path))
    if args.list_topics:
        print(json.dumps(topics, indent=2, ensure_ascii=False))
        return

    image_topic = choose_topic(topics, IMAGE_MSG_TYPES, args.image_topic, "image")
    imu_topic = None if args.skip_imu else choose_topic(topics, IMU_MSG_TYPES, args.imu_topic, "imu")
    pose_topic = None if args.skip_pose else choose_topic(topics, POSE_MSG_TYPES, args.pose_topic, "pose")

    if not image_topic:
        raise RuntimeError("No image topic found. Use --image_topic to specify it explicitly.")

    out_root = build_output_root(args, bag_path)
    img_dir = out_root / args.img_subdir
    img_dir.mkdir(parents=True, exist_ok=True)

    image_rows: List[dict] = []
    imu_rows: List[dict] = []
    pose_rows: List[dict] = []
    extracted_frames = 0
    seen_image_msgs = 0
    bag_start = None
    bag_end = None

    topic_filter = [image_topic]
    if imu_topic:
        topic_filter.append(imu_topic)
    if pose_topic:
        topic_filter.append(pose_topic)

    print(f"[extract] bag         : {bag_path}")
    print(f"[extract] out_root    : {out_root}")
    print(f"[extract] image_topic : {image_topic}")
    print(f"[extract] imu_topic   : {imu_topic}")
    print(f"[extract] pose_topic  : {pose_topic}")

    with rosbag.Bag(str(bag_path), "r") as bag:
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())
        rel_end = None if args.end_sec < 0 else float(args.end_sec)

        for topic, msg, t_ros in bag.read_messages(topics=topic_filter):
            t_sec = stamp_to_sec(t_ros)
            rel_sec = t_sec - bag_start

            if rel_sec < float(args.start_sec):
                continue
            if rel_end is not None and rel_sec > rel_end:
                break

            if topic == image_topic:
                seen_image_msgs += 1
                if (seen_image_msgs - 1) % int(args.every_n) != 0:
                    continue

                img = decode_ros_image(msg)
                frame_name = f"{extracted_frames:06d}"
                img_path = img_dir / f"{frame_name}{args.img_ext}"
                ok = cv2.imwrite(str(img_path), img)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite failed: {img_path}")

                image_rows.append(
                    {
                        "frame_index": extracted_frames,
                        "frame_name": frame_name,
                        "stamp_sec": f"{t_sec:.9f}",
                        "rel_sec": f"{rel_sec:.9f}",
                        "topic": topic,
                        "path": str(img_path),
                        "width": int(img.shape[1]),
                        "height": int(img.shape[0]),
                        "channels": 1 if img.ndim == 2 else int(img.shape[2]),
                    }
                )
                extracted_frames += 1
                if extracted_frames % 100 == 0:
                    print(f"[extract] image frames: {extracted_frames}")
                if 0 < int(args.max_frames) <= extracted_frames:
                    break
                continue

            if topic == imu_topic:
                row = extract_imu_row(msg, t_ros)
                if row is not None:
                    imu_rows.append(row)
                continue

            if topic == pose_topic:
                row = extract_pose_row(msg, t_ros)
                if row is not None:
                    pose_rows.append(row)
                continue

    write_csv(
        out_root / "timestamps.csv",
        ["frame_index", "frame_name", "stamp_sec", "rel_sec", "topic", "path", "width", "height", "channels"],
        image_rows,
    )

    if imu_topic and imu_rows:
        write_csv(out_root / "imu.csv", ["stamp_sec", "gx", "gy", "gz", "ax", "ay", "az"], imu_rows)

    if pose_topic and pose_rows:
        write_csv(out_root / "gt_pose.csv", ["stamp_sec", "px", "py", "pz", "qx", "qy", "qz", "qw"], pose_rows)

    summary = {
        "bag": str(bag_path),
        "out_root": str(out_root),
        "image_topic": image_topic,
        "imu_topic": imu_topic,
        "pose_topic": pose_topic,
        "img_subdir": args.img_subdir,
        "img_ext": args.img_ext,
        "every_n": int(args.every_n),
        "start_sec": float(args.start_sec),
        "end_sec": None if args.end_sec < 0 else float(args.end_sec),
        "bag_start_sec": bag_start,
        "bag_end_sec": bag_end,
        "num_frames": int(extracted_frames),
        "num_imu": int(len(imu_rows)),
        "num_pose": int(len(pose_rows)),
        "topics": topics,
    }
    with open(out_root / "bag_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] frames={extracted_frames}, imu={len(imu_rows)}, pose={len(pose_rows)}")
    print(f"[done] summary: {out_root / 'bag_summary.json'}")


if __name__ == "__main__":
    main()
