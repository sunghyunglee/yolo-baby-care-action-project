from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse: 실행 옵션 / json: Split 설정 / Path: 경로 / pandas: 분리·저장

import argparse
import json
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 기본 경로 (day05 → 05_dataset_qa)
# ==================================================
# 입력: 29번 approved_clip_manifest.csv
# 설정: configs/subject_split.json (어떤 Subject를 어느 Split에 넣을지)
# 출력: data/05_dataset_qa/splits/ train.csv · val.csv · test.csv

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "05_dataset_qa"

DEFAULT_MANIFEST_PATH = DATA_DIR / "manifests" / "approved_clip_manifest.csv"

DEFAULT_SPLIT_CONFIG = ROOT / "configs" / "subject_split.json"

DEFAULT_OUTPUT_DIR = DATA_DIR / "splits"

UNLABELED = "UNLABELED"


# ==================================================
# ③ 터미널 실행 옵션 설정
# ==================================================
# --expected-actions:
#   비워 두면(기본) 승인 데이터에 실제로 있는 동작 전부를 필수로 봅니다.
#   직접 주면 그 동작만 검사합니다. 예: A001,A002,A003

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--split-config", default=str(DEFAULT_SPLIT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--expected-actions",
        default="",
        help="쉼표로 구분. 비우면 승인 데이터의 동작 전부를 사용합니다.",
    )

    return parser.parse_args()


# ==================================================
# ④ 필수 Action 집합 만들기
# ==================================================
# 옵션 문자열이 있으면 → 그 목록
# 없으면 → 승인 데이터의 action_id 전부

def build_expected_actions(value: str, df: pd.DataFrame) -> set[str]:
    actions = {item.strip().upper() for item in value.split(",") if item.strip()}

    if not actions:
        actions = set(df["action_id"].astype(str).str.strip().str.upper())

    if not actions:
        raise ValueError("필수 Action이 비어 있습니다.")

    return actions


# ==================================================
# ⑤ Subject-wise Split 설정 JSON 읽기
# ==================================================

def load_split_config(split_config: Path) -> dict:
    with split_config.open("r", encoding="utf-8") as file:
        return json.load(file)


# ==================================================
# ⑥ Train·Validation·Test Subject 중복 검사 (강사님 코드 그대로)
# ==================================================
# 같은 사람이 두 Split에 들어가면 Subject Leakage → 중단

def validate_subject_sets(train_subjects, val_subjects, test_subjects) -> None:
    train = set(train_subjects)
    val = set(val_subjects)
    test = set(test_subjects)

    if train & val:
        raise RuntimeError("Train / Validation Subject 중복")

    if train & test:
        raise RuntimeError("Train / Test Subject 중복")

    if val & test:
        raise RuntimeError("Validation / Test Subject 중복")


# ==================================================
# ⑦ 각 Split이 비어 있지 않고 필수 Action을 모두 가졌는지 검사
# ==================================================

def require_action_coverage(
    split_name: str,
    df: pd.DataFrame,
    expected_actions: set[str],
) -> None:
    if df.empty:
        raise RuntimeError(f"{split_name} 데이터가 비어 있습니다.")

    actual_actions = set(df["action_id"].astype(str).str.strip().str.upper())

    missing_actions = expected_actions - actual_actions

    if missing_actions:
        raise RuntimeError(
            f"{split_name}에 없는 Action: " + ", ".join(sorted(missing_actions))
        )

    # ==============================================
    # ⑧ Split별 Subject·Clip 수·Action 분포 출력
    # ==============================================

    print()
    print(f"[{split_name}]")
    print("Subjects:", sorted(df["subject"].unique()))
    print("Clips:", len(df))
    print(df["action_label"].value_counts().sort_index())


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path)
    split_config = Path(args.split_config)
    output_dir = Path(args.output_dir)

    # ==============================================
    # ⑨ 입력 파일 존재 검사
    # ==============================================

    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} 가 없습니다. 먼저 29번을 실행하세요.")

    if not split_config.exists():
        raise FileNotFoundError(f"{split_config} 가 없습니다. Split 설정 JSON을 만드세요.")

    config = load_split_config(split_config)

    # ==============================================
    # ⑩ JSON에서 Subject 배정 가져오기 + 중복 검사
    # ==============================================
    # 바른품 기본: S01 → Train / S02 → Validation / S03 → Test

    train_subjects = config["train_subjects"]
    val_subjects = config["val_subjects"]
    test_subjects = config["test_subjects"]

    validate_subject_sets(train_subjects, val_subjects, test_subjects)

    # ==============================================
    # ⑪ Approved Manifest 읽기 + 필수 Action 결정
    # ==============================================

    df = pd.read_csv(manifest_path)

    expected_actions = build_expected_actions(args.expected_actions, df)

    print("Expected actions:", ", ".join(sorted(expected_actions)))

    # ==============================================
    # ⑫ 미배정 Subject와 누락 Subject 검사 (강사님 코드 그대로)
    # ==============================================

    configured_subjects = set(train_subjects + val_subjects + test_subjects)
    data_subjects = set(df["subject"].astype(str))

    unassigned = data_subjects - configured_subjects
    missing_from_data = configured_subjects - data_subjects

    if unassigned:
        raise RuntimeError("Split에 배정되지 않은 Subject: " + ", ".join(sorted(unassigned)))

    if missing_from_data:
        raise RuntimeError(
            "Approved 데이터에 없는 Subject가 설정에 있습니다: "
            + ", ".join(sorted(missing_from_data))
        )

    # ==============================================
    # ⑬ Subject 기준으로 세 데이터셋 분리
    # ==============================================
    # 무작위로 섞지 않고, 같은 사람의 Clip 전체를 한 Split에 넣습니다.

    train_df = df[df["subject"].isin(train_subjects)].copy()
    val_df = df[df["subject"].isin(val_subjects)].copy()
    test_df = df[df["subject"].isin(test_subjects)].copy()

    # ==============================================
    # ⑭ 세 Split의 Action 구성 검사
    # ==============================================

    require_action_coverage("TRAIN", train_df, expected_actions)
    require_action_coverage("VALIDATION", val_df, expected_actions)
    require_action_coverage("TEST", test_df, expected_actions)

    # ==============================================
    # ⑮ train.csv · val.csv · test.csv 저장
    # ==============================================

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    print()

    for name, split_df in outputs.items():
        path = output_dir / f"{name}.csv"
        split_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path.relative_to(ROOT)}")

    # ==============================================
    # ⑯ 동작 미정 경고 + 완료 메시지
    # ==============================================
    # UNLABELED가 있으면 Split은 됐지만 6일차 학습에는 쓸 수 없습니다.

    if UNLABELED in expected_actions:
        print()
        print(
            "[WARN] action_id가 UNLABELED인 Clip이 있습니다. "
            "Split은 완료됐지만 6일차 학습 전에 동작 Label을 채워야 합니다."
        )

    print()
    print("Subject-wise split: COMPLETE")


# ==================================================
# ⑰ Python 파일 직접 실행 시 main() 호출
# ==================================================

if __name__ == "__main__":
    main()
    