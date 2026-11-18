from __future__ import annotations

# ====================================================
# 1번. 필요한 라이브러리 가져오기
# ====================================================
# argparse: 터미널 실행 옵션 / csv: CSV 읽기·저장
# sys, Path: 프로젝트 모듈·경로 / cv2: MP4 Frame 읽기
# NumPy: Confidence 평균·대표 인물 선택 / YOLO: Pose 추론

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ====================================================
# 2번. 프로젝트 ROOT 설정 및 공통 모듈 가져오기
# ====================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
)
from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ====================================================
# 3번. 입력·출력 경로 정의 (바른품: day05 → 05_dataset_qa)
# ====================================================
# 입력: 25번이 만든 video_inventory.csv
#       (clip_id, subject, action_id, take, video_file, video_path)
# 출력: data/05_dataset_qa/keypoints/ 아래 Clip별 CSV
# 전체 목록: manifests/keypoint_dataset_manifest.csv
# 영상 이름이 촬영 시간이라 파일명 규칙 대신 inventory를 기준으로 씁니다.

DEFAULT_INVENTORY_PATH = (
    ROOT / "reports" / "05_dataset_qa" / "tables" / "video_inventory.csv"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT / "data" / "05_dataset_qa" / "keypoints"
)

ACTION_FILE = (
    ROOT / "configs" / "action_classes.csv"
)

DEFAULT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "05_dataset_qa"
    / "manifests"
    / "keypoint_dataset_manifest.csv"
)

UNLABELED = "UNLABELED"

REQUIRED_INVENTORY_COLUMNS = {
    "clip_id",
    "subject",
    "action_id",
    "take",
    "video_file",
    "video_path",
}


# ====================================================
# 4번. 터미널 실행 옵션 설정
# ====================================================
# --inventory: 25번 결과 CSV
# --output-root: Keypoint CSV 저장 폴더
# --manifest-path: 전체 Clip 목록표 저장 경로
# --expected-clips: 예상 Clip 수 (0이면 inventory 행 수를 그대로 사용)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inventory",
        default=str(DEFAULT_INVENTORY_PATH),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
    )
    parser.add_argument(
        "--expected-clips",
        type=int,
        default=0,
        help="0이면 inventory 행 수를 그대로 사용합니다.",
    )

    return parser.parse_args()


# ====================================================
# 5번. Action ID와 동작 이름 연결
# ====================================================
# action_classes.csv(바른품 12개 동작 A001~A012)를 읽습니다.
# 강사님 코드의 "A001~A006과 정확히 같아야 함" 검사는 뺐습니다.
# UNLABELED(아직 동작을 안 정한 영상)는 unlabeled로 연결합니다.

def load_action_map() -> dict[str, str]:
    action_map = {UNLABELED: "unlabeled"}

    if not ACTION_FILE.exists():
        print(f"[WARN] {ACTION_FILE.relative_to(ROOT)} 없음 → UNLABELED만 처리합니다.")
        return action_map

    with ACTION_FILE.open("r", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            action_id = row["action_id"].strip().upper()
            action_map[action_id] = row["action_label"].strip()

    return action_map


# ====================================================
# 6번. 17개 관절의 총 85개 컬럼 이름 생성 (강사님 코드 그대로)
# ====================================================
# 관절마다 x_px, y_px, x_norm, y_norm, conf → 17 × 5 = 85개

def build_keypoint_columns() -> list[str]:
    columns = []

    for name in COCO_KEYPOINT_NAMES:
        columns.extend(
            [
                f"{name}_x_px",
                f"{name}_y_px",
                f"{name}_x_norm",
                f"{name}_y_norm",
                f"{name}_conf",
            ]
        )

    return columns


# ====================================================
# 7번. video_inventory.csv 읽기 (파일명 규칙 대신)
# ====================================================
# 필수 컬럼이 있는지, clip_id가 겹치지 않는지,
# 영상 파일이 실제로 있는지 확인합니다.

def load_inventory(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 25_check_qa_data.py를 실행하세요."
        )

    with path.open("r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_INVENTORY_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise RuntimeError(f"inventory 컬럼 누락: {sorted(missing_columns)}")
        items = list(reader)

    if not items:
        raise RuntimeError("inventory에 Clip이 없습니다.")

    clip_ids = [item["clip_id"] for item in items]
    if len(clip_ids) != len(set(clip_ids)):
        raise RuntimeError("inventory에 같은 clip_id가 두 번 이상 있습니다.")

    for item in items:
        video_path = ROOT / item["video_path"]
        if not video_path.exists():
            raise FileNotFoundError(f"영상이 없습니다: {item['video_path']}")

    return items


# ====================================================
# 8번. 검출된 사람의 Pose 데이터 존재 여부 확인 (강사님 코드 그대로)
# ====================================================

def select_person(result):
    if (
        result.boxes is None
        or result.keypoints is None
        or result.keypoints.xy is None
        or len(result.boxes) == 0
        or len(result.keypoints.xy) == 0
    ):
        return None

    # ====================================================
    # 9번. BBox Confidence가 가장 높은 사람 선택 (강사님 코드 그대로)
    # ====================================================
    # 주의: Frame마다 대표자가 달라질 수 있으며 사람 추적은 아닙니다.
    # (바른품은 보모+아기가 함께 나오므로 8일차 Tracking에서 보완)

    box_conf = result.boxes.conf.detach().cpu().numpy()

    person_count = min(len(box_conf), len(result.keypoints.xy))

    if person_count == 0:
        return None

    return int(np.argmax(box_conf[:person_count]))


# ====================================================
# 10번. Clip 하나의 기본 정보 및 영상 Metadata 확인
# ====================================================
# Subject·Action·Take·clip_id를 inventory 행에서 가져옵니다.

def process_clip(
    model,
    item: dict,
    action_map: dict[str, str],
    config: dict,
    device,
    output_root: Path,
) -> dict:
    clip_id = item["clip_id"]
    subject = item["subject"].strip().upper()
    action_id = item["action_id"].strip().upper()
    take = item["take"].strip().upper()
    clip_path = ROOT / item["video_path"]

    action_label = action_map.get(action_id)

    if action_label is None:
        raise ValueError(f"action_classes.csv에 없는 Action: {action_id} ({clip_id})")

    capture = cv2.VideoCapture(str(clip_path))

    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {clip_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"영상 Metadata를 확인할 수 없습니다: {clip_path}")

    # ====================================================
    # 11번. Clip별 출력 폴더와 CSV 컬럼 구성
    # ====================================================
    # 예: keypoints/S01/unlabeled/S01_V01.csv  (파일 이름 = clip_id)
    # 기본정보 10개 + Keypoint 85개 = 95개 컬럼

    output_dir = output_root / subject / action_label
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{clip_id}.csv"

    keypoint_columns = build_keypoint_columns()

    fieldnames = [
        "clip_id",
        "frame_id",
        "time_sec",
        "subject",
        "action_id",
        "action_label",
        "take",
        "person_count",
        "selected_box_conf",
        "pose_detected",
        *keypoint_columns,
    ]

    # ====================================================
    # 12번. Frame 처리 결과와 집계 변수 초기화 (강사님 코드 그대로)
    # ====================================================

    rows = []
    frame_id = 0
    detected_frames = 0
    single_person_frames = 0
    frame_mean_confidences = []

    # ====================================================
    # 13번. MP4를 처음부터 끝까지 Frame 단위로 읽기
    # ====================================================

    while True:
        success, frame = capture.read()

        if not success:
            break

        # ====================================================
        # 14번. 현재 Frame에 YOLO Pose 추론 실행
        # ====================================================

        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )[0]

        # ====================================================
        # 15번. 검출 인원 수 계산 및 Frame 기본 행 생성
        # ====================================================
        # Pose가 없는 Frame도 빈 관절값으로 시간 순서대로 보존합니다.

        person_count = len(result.boxes) if result.boxes is not None else 0

        if person_count == 1:
            single_person_frames += 1

        row = {
            "clip_id": clip_id,
            "frame_id": frame_id,
            "time_sec": round(frame_id / fps, 4),
            "subject": subject,
            "action_id": action_id,
            "action_label": action_label,
            "take": take,
            "person_count": person_count,
            "selected_box_conf": "",
            "pose_detected": 0,
        }

        for column in keypoint_columns:
            row[column] = ""

        # ====================================================
        # 16번. 대표 인물 선택 및 COCO 17 Keypoint 수 검사
        # ====================================================

        selected_index = select_person(result)

        if selected_index is not None:
            xy = result.keypoints.xy[selected_index].detach().cpu().numpy()

            if len(xy) != len(COCO_KEYPOINT_NAMES):
                capture.release()
                raise RuntimeError(f"COCO 17 Keypoint 수가 아닙니다: {clip_path.name}")

            # ====================================================
            # 17번. 관절·BBox Confidence 가져오기 및 검출 통계 기록
            # ====================================================
            # Keypoint Confidence가 없으면 1.0으로 대체(실제 측정값 아님)

            if result.keypoints.conf is not None:
                kp_conf = result.keypoints.conf[selected_index].detach().cpu().numpy()
            else:
                kp_conf = np.ones(len(COCO_KEYPOINT_NAMES), dtype=np.float32)

            box_conf = float(result.boxes.conf[selected_index].detach().cpu().item())

            row["selected_box_conf"] = round(box_conf, 4)
            row["pose_detected"] = 1

            detected_frames += 1

            frame_mean_confidences.append(float(np.mean(kp_conf)))

            # ====================================================
            # 18번. Pixel·Image Normalized·Confidence 값 저장
            # ====================================================
            # x_px·y_px(원본 픽셀 좌표)는 손↔아기 거리 계산에 필요하므로 꼭 보존
            # 주의: 이미지 기준 정규화는 3일차 몸 중심 정규화와 다릅니다.

            for index, name in enumerate(COCO_KEYPOINT_NAMES):
                x_px, y_px = xy[index]

                row[f"{name}_x_px"] = round(float(x_px), 3)
                row[f"{name}_y_px"] = round(float(y_px), 3)
                row[f"{name}_x_norm"] = round(float(x_px) / width, 6)
                row[f"{name}_y_norm"] = round(float(y_px) / height, 6)
                row[f"{name}_conf"] = round(float(kp_conf[index]), 4)

        # ====================================================
        # 19번. Frame별 결과 보존 및 60 Frame마다 진행상황 출력
        # ====================================================

        rows.append(row)
        frame_id += 1

        if frame_id % 60 == 0:
            print(f"  frame {frame_id}")

    # ====================================================
    # 20번. 영상 해제 및 빈 영상 검사
    # ====================================================

    capture.release()

    if frame_id == 0:
        raise RuntimeError(f"읽은 Frame이 없습니다: {clip_path}")

    # ====================================================
    # 21번. Clip 한 개의 Keypoint CSV 저장
    # ====================================================

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ====================================================
    # 22번. Pose 성공률·1인 검출률·평균 관절 신뢰도 계산
    # ====================================================
    # pose_ratio = Pose 검출 Frame 수 / 전체 Frame 수
    # single_person_ratio = 정확히 1명 검출 Frame 수 / 전체 Frame 수
    # (바른품은 보모+아기라 single_person_ratio가 낮게 나올 수 있음)

    pose_ratio = detected_frames / frame_id
    single_person_ratio = single_person_frames / frame_id

    mean_kp_conf = (
        float(np.mean(frame_mean_confidences))
        if frame_mean_confidences
        else 0.0
    )

    # ====================================================
    # 23번. Clip별 Manifest 항목 반환
    # ====================================================
    # video_file(원래 영상 이름)을 추가로 남겨 원본과 다시 연결합니다.

    return {
        "clip_id": clip_id,
        "video_file": item["video_file"],
        "video_path": item["video_path"],
        "keypoint_csv": str(output_path.relative_to(ROOT)),
        "subject": subject,
        "action_id": action_id,
        "action_label": action_label,
        "take": take,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "metadata_frames": metadata_frames,
        "read_frames": frame_id,
        "pose_detected_frames": detected_frames,
        "pose_ratio": round(pose_ratio, 4),
        "single_person_ratio": round(single_person_ratio, 4),
        "mean_keypoint_conf": round(mean_kp_conf, 4),
    }


# ====================================================
# 24번. main 실행 옵션과 필수 입력 검증
# ====================================================

def main() -> None:
    args = parse_args()

    inventory_path = Path(args.inventory)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest_path)

    if args.expected_clips < 0:
        raise ValueError("--expected-clips는 0 이상이어야 합니다.")

    # ====================================================
    # 25번. Action 매핑·모델 설정·Clip 목록 준비
    # ====================================================
    # Clip 목록 = inventory 행 (25번에서 폴더·Metadata 검사 완료)

    action_map = load_action_map()
    config = load_config()
    device = resolve_device(config["device"])

    items = load_inventory(inventory_path)

    if args.expected_clips and len(items) != args.expected_clips:
        raise RuntimeError(
            "입력 Clip 개수가 예상과 다릅니다. "
            f"expected={args.expected_clips}, actual={len(items)}"
        )

    # ====================================================
    # 26번. YOLO Pose 모델 로드 후 모든 Clip 처리
    # ====================================================

    output_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(config["pose_model"])

    manifest_rows = []

    print(f"Clips: {len(items)}")
    print()

    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['clip_id']}  ({item['video_file']})")

        result_row = process_clip(
            model=model,
            item=item,
            action_map=action_map,
            config=config,
            device=device,
            output_root=output_root,
        )

        manifest_rows.append(result_row)

        print(
            f"  pose_ratio={result_row['pose_ratio']} "
            f"mean_conf={result_row['mean_keypoint_conf']}"
        )

    # ====================================================
    # 27번. 전체 Manifest CSV 저장 및 완료 메시지 출력
    # ====================================================

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("Keypoint extraction: COMPLETE")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")


# ====================================================
# 28번. Python 파일 직접 실행 시 main() 시작
# ====================================================

if __name__ == "__main__":
    main()
    