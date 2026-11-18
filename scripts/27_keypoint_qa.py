from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse: 실행 옵션 / csv: 결과 저장 / sys, Path: 모듈·경로
# NumPy: Confidence 계산 / Pandas: Keypoint CSV 통계

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 공통 설정 함수 연결
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config


# ==================================================
# ③ Manifest 입력 경로와 QA 출력 경로 설정 (day05 → 05_dataset_qa)
# ==================================================
# 입력: 26번이 만든 keypoint_dataset_manifest.csv
# 출력: 자동 QA 결과(keypoint_clip_qa.csv) + Human QA 작업표(keypoint_clip_review.csv)

DATA_DIR = ROOT / "data" / "05_dataset_qa"

DEFAULT_MANIFEST_PATH = DATA_DIR / "manifests" / "keypoint_dataset_manifest.csv"

DEFAULT_QA_PATH = DATA_DIR / "qa" / "keypoint_clip_qa.csv"

DEFAULT_REVIEW_PATH = DATA_DIR / "qa" / "keypoint_clip_review.csv"


# ==================================================
# ④ 중요 Keypoint 정의 (바른품)
# ==================================================
# DEFAULT_CRITICAL_KEYPOINTS:
#   보모 손과 아기의 관계가 핵심 → 어깨·팔꿈치·손목
#   + 3일차 정규화 기준(Hip Center) → 골반
# ACTION_CRITICAL_KEYPOINTS:
#   동작 이름이 정해지면 동작별로 추가합니다.
#   예) 바닥 동작은 무릎·발목 추가
# 표에 없는 동작(unlabeled 포함)은 DEFAULT로 검사합니다.

DEFAULT_CRITICAL_KEYPOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]

ACTION_CRITICAL_KEYPOINTS: dict[str, list[str]] = {
    # "동작이름": [관절, ...],
}


# ==================================================
# ⑤ 터미널 실행 옵션 설정
# ==================================================
# --expected-clips: 0이면 manifest 행 수 사용
# QA 기준 4개: 기본값 = 강사님 기준 (결과를 보고 바꿀지 결정)
# --overwrite-review: Human QA 내용이 있어도 Review 파일을 덮어쓸 때만 사용

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--qa-path", default=str(DEFAULT_QA_PATH))
    parser.add_argument("--review-path", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument("--expected-clips", type=int, default=0)

    parser.add_argument("--pose-min", type=float, default=0.90)
    parser.add_argument("--single-person-min", type=float, default=0.90)
    parser.add_argument("--mean-conf-min", type=float, default=0.50)
    parser.add_argument("--low-critical-max", type=float, default=0.20)

    parser.add_argument("--overwrite-review", action="store_true")

    return parser.parse_args()


# ==================================================
# ⑥ Clip 하나의 Keypoint CSV 읽기 및 빈 파일 검사
# ==================================================
# 빈 CSV는 학습 데이터로 볼 수 없으므로 CHECK

def inspect_keypoint_csv(
    csv_path: Path,
    action_label: str,
    threshold: float,
    args,
) -> dict:
    df = pd.read_csv(csv_path)

    # ==============================================
    # ⑦ 현재 Action에서 중요한 관절 선택
    # ==============================================
    # 표에 없으면 오류 대신 DEFAULT 관절 사용

    critical_names = ACTION_CRITICAL_KEYPOINTS.get(
        action_label,
        DEFAULT_CRITICAL_KEYPOINTS,
    )

    if df.empty:
        return {
            "qa_status": "CHECK",
            "frame_count": 0,
            "pose_ratio": 0.0,
            "single_person_ratio": 0.0,
            "mean_keypoint_conf": 0.0,
            "low_critical_ratio": 1.0,
            "missing_pose_frames": 0,
            "critical_keypoints": "|".join(critical_names),
            "check_reasons": "empty_csv",
        }

    # ==============================================
    # ⑧ Frame 수와 Pose·사람 검출 정보를 숫자로 변환
    # ==============================================

    frame_count = len(df)

    pose_series = pd.to_numeric(df["pose_detected"], errors="coerce").fillna(0)
    person_series = pd.to_numeric(df["person_count"], errors="coerce").fillna(0)

    # ==============================================
    # ⑨ Pose 검출률과 단일 인물 검출률 계산
    # ==============================================
    # pose_ratio = Pose 검출 Frame 비율
    # single_person_ratio = 정확히 1명 검출 Frame 비율
    # (바른품: 아기 인형도 사람으로 잡히면 2명이 되어 낮아질 수 있음)

    pose_ratio = float(pose_series.mean())
    single_person_ratio = float((person_series == 1).mean())

    # ==============================================
    # ⑩ 전체 Keypoint 평균 Confidence 계산
    # ==============================================
    # selected_box_conf(BBox 신뢰도)는 관절 Confidence가 아니므로 제외

    confidence_columns = [
        column
        for column in df.columns
        if column.endswith("_conf") and column != "selected_box_conf"
    ]

    confidence_values = (
        df[confidence_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )

    mean_keypoint_conf = (
        float(np.nanmean(confidence_values))
        if np.isfinite(confidence_values).any()
        else 0.0
    )

    # ==============================================
    # ⑪ 중요 Keypoint Confidence 컬럼 준비 및 확인
    # ==============================================

    critical_columns = [f"{name}_conf" for name in critical_names]

    missing_columns = [column for column in critical_columns if column not in df.columns]

    if missing_columns:
        raise RuntimeError(
            "중요 Keypoint Confidence 열이 없습니다: " + ", ".join(missing_columns)
        )

    # ==============================================
    # ⑫ 중요 Keypoint Confidence를 숫자 배열로 변환
    # ==============================================

    critical_values = (
        df[critical_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )

    # ==============================================
    # ⑬ 중요 관절의 Low Confidence 비율 계산
    # ==============================================
    # low_critical_ratio = threshold보다 낮은 값 수 / 유효한 값 수
    # 빈 값(Pose Missing)은 분모에서 제외 → pose_ratio로 따로 검사

    valid_mask = np.isfinite(critical_values)
    valid_count = int(valid_mask.sum())

    if valid_count > 0:
        low_count = int(((critical_values < threshold) & valid_mask).sum())
        low_critical_ratio = low_count / valid_count
    else:
        low_critical_ratio = 1.0

    # ==============================================
    # ⑭ Pose Missing 계산 및 AUTO_PASS·CHECK 판정
    # ==============================================
    # 기준(기본값 = 강사님): pose < 0.90 / single < 0.90 /
    #                        mean_conf < 0.50 / low_critical > 0.20 → CHECK
    # check_reasons: 어떤 기준에 걸렸는지 기록 (원인 추적용)
    # AUTO_PASS는 Human QA의 최종 PASS가 아닙니다.

    missing_pose_frames = int((pose_series == 0).sum())

    reasons = []

    if pose_ratio < args.pose_min:
        reasons.append("low_pose_ratio")

    if single_person_ratio < args.single_person_min:
        reasons.append("multi_or_no_person")

    if mean_keypoint_conf < args.mean_conf_min:
        reasons.append("low_mean_conf")

    if low_critical_ratio > args.low_critical_max:
        reasons.append("low_critical_conf")

    qa_status = "CHECK" if reasons else "AUTO_PASS"

    # ==============================================
    # ⑮ Clip 하나의 QA 검사 결과 반환
    # ==============================================

    return {
        "qa_status": qa_status,
        "frame_count": frame_count,
        "pose_ratio": round(pose_ratio, 4),
        "single_person_ratio": round(single_person_ratio, 4),
        "mean_keypoint_conf": round(mean_keypoint_conf, 4),
        "low_critical_ratio": round(low_critical_ratio, 4),
        "missing_pose_frames": missing_pose_frames,
        "critical_keypoints": "|".join(critical_names),
        "check_reasons": "|".join(reasons),
    }


# ==================================================
# ⑯ Review 파일에 이미 Human QA 내용이 있는지 확인
# ==================================================
# human_decision이 UNREVIEWED가 아닌 행이 하나라도 있으면 True

def review_has_human_input(review_path: Path) -> bool:
    if not review_path.exists():
        return False

    review = pd.read_csv(review_path)

    if "human_decision" not in review.columns:
        return False

    decisions = review["human_decision"].fillna("").astype(str).str.strip().str.upper()

    return bool(((decisions != "") & (decisions != "UNREVIEWED")).any())


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path)
    qa_path = Path(args.qa_path)
    review_path = Path(args.review_path)

    # ==============================================
    # ⑰ 입력 파일·Confidence 기준 확인
    # ==============================================
    # keypoint_confidence: configs/project.json 값(기본 0.50)

    if args.expected_clips < 0:
        raise ValueError("--expected-clips는 0 이상이어야 합니다.")

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} 가 없습니다. 먼저 26번을 실행하세요."
        )

    config = load_config()

    threshold = float(config.get("keypoint_confidence", 0.50))

    # ==============================================
    # ⑱ Manifest 읽기 및 예상 Clip 개수 확인
    # ==============================================

    manifest = pd.read_csv(manifest_path)

    if args.expected_clips and len(manifest) != args.expected_clips:
        raise RuntimeError(
            "Manifest Clip 수가 예상과 다릅니다. "
            f"expected={args.expected_clips}, actual={len(manifest)}"
        )

    rows = []

    # ==============================================
    # ⑲ Manifest의 Clip별 Keypoint CSV 경로 확인
    # ==============================================

    for _, item in manifest.iterrows():
        csv_path = ROOT / str(item["keypoint_csv"])

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        # ==========================================
        # ⑳ Clip별 자동 QA 실행 및 결과 행 생성
        # ==========================================
        # video_file(원래 영상 이름)도 같이 남겨 원본 영상과 다시 연결

        qa = inspect_keypoint_csv(
            csv_path=csv_path,
            action_label=str(item["action_label"]),
            threshold=threshold,
            args=args,
        )

        row = {
            "clip_id": item["clip_id"],
            "subject": item["subject"],
            "action_id": item["action_id"],
            "action_label": item["action_label"],
            "take": item["take"],
            "video_file": item.get("video_file", ""),
            "video_path": item["video_path"],
            "keypoint_csv": item["keypoint_csv"],
            **qa,
            "human_decision": "UNREVIEWED",
            "human_note": "",
        }

        # ==========================================
        # ㉑ QA 결과 저장 및 진행 상황 출력
        # ==========================================

        rows.append(row)

        print(
            f"{item['clip_id']:10s} "
            f"{qa['qa_status']:9s} "
            f"pose={qa['pose_ratio']:.3f} "
            f"single={qa['single_person_ratio']:.3f} "
            f"conf={qa['mean_keypoint_conf']:.3f} "
            f"low={qa['low_critical_ratio']:.3f} "
            f"{qa['check_reasons']}"
        )

    # ==============================================
    # ㉒ 폴더·컬럼 준비 + Review 덮어쓰기 보호
    # ==============================================
    # Human QA를 이미 적었다면 Review는 그대로 두고 QA 파일만 저장합니다.

    qa_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    output_paths = [qa_path]

    if review_has_human_input(review_path) and not args.overwrite_review:
        print()
        print(
            "[KEEP] Review 파일에 Human QA 내용이 있어 덮어쓰지 않았습니다. "
            "(덮어쓰려면 --overwrite-review)"
        )
    else:
        output_paths.append(review_path)

    # ==============================================
    # ㉓ 자동 QA CSV와 Human QA 작업표 저장
    # ==============================================

    for output_path in output_paths:
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ==============================================
    # ㉔ 결과 요약 출력
    # ==============================================

    auto_pass = sum(row["qa_status"] == "AUTO_PASS" for row in rows)
    check = len(rows) - auto_pass

    print()
    print(f"AUTO_PASS: {auto_pass} / CHECK: {check} / Total: {len(rows)}")
    print(f"QA     : {qa_path.relative_to(ROOT)}")
    print(f"Review : {review_path.relative_to(ROOT)}")


# ==================================================
# ㉕ Python 파일 직접 실행 시 main() 호출
# ==================================================

if __name__ == "__main__":
    main()
    