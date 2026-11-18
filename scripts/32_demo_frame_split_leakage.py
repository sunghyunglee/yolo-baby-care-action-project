from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# Path: 경로 / Pandas: CSV 읽기·합치기 / train_test_split: 무작위 분리
# 이 실습은 모델을 학습하지 않고, 잘못된 분리의 위험만 확인합니다.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# ② 프로젝트 ROOT와 입력 경로 (day05 → 05_dataset_qa)
# ==================================================
# 폴더를 통째로 검색(rglob)하지 않고 26번 manifest에 적힌 CSV만 읽습니다.
# → keypoints 폴더에 예전 결과(unlabeled 등)가 남아 있어도 섞이지 않습니다.

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "data"
    / "05_dataset_qa"
    / "manifests"
    / "keypoint_dataset_manifest.csv"
)


def main() -> None:

    # ==============================================
    # ③ manifest에서 Keypoint CSV 목록 가져오기
    # ==============================================
    # 강사님 코드의 "18개 고정" 검사 대신 manifest 행 수를 그대로 사용합니다.

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} 가 없습니다. 먼저 26번을 실행하세요.")

    manifest = pd.read_csv(MANIFEST_PATH)

    csv_paths = [ROOT / str(path) for path in manifest["keypoint_csv"]]

    missing = [path for path in csv_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(f"Keypoint CSV가 없습니다: {missing[0]}")

    print(f"Keypoint CSV: {len(csv_paths)}개 (manifest 기준)")

    # ==============================================
    # ④ CSV별 Frame 정보를 읽어 목록에 모으기 (강사님 코드 그대로)
    # ==============================================
    # clip_id · frame_id · subject · action_label 네 컬럼만 읽습니다.

    frames = []

    for path in csv_paths:
        df = pd.read_csv(
            path,
            usecols=["clip_id", "frame_id", "subject", "action_label"],
        )
        frames.append(df)

    # ==============================================
    # ⑤ 모든 Frame을 하나의 표로 결합
    # ==============================================
    # 각 행 = 영상 하나가 아니라 Frame 하나

    all_frames = pd.concat(frames, ignore_index=True)

    # ==============================================
    # ⑥ Frame 단위로 Train 80% / Test 20% 무작위 분리 (잘못된 방법 시연)
    # ==============================================
    # Subject·Clip을 묶지 않으므로 같은 사람·같은 영상의 Frame이 양쪽에 섞입니다.

    train_df, test_df = train_test_split(
        all_frames,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    # ==============================================
    # ⑦ Subject 겹침 계산
    # ==============================================

    train_subjects = set(train_df["subject"].astype(str))
    test_subjects = set(test_df["subject"].astype(str))

    subject_overlap = train_subjects & test_subjects

    # ==============================================
    # ⑧ Clip 겹침 계산
    # ==============================================
    # 같은 clip_id가 양쪽에 있다 = 같은 MP4의 Frame이 Train·Test로 나뉨
    # → 인접 Frame은 거의 같아서 점수가 실제보다 높게 나올 위험

    train_clips = set(train_df["clip_id"].astype(str))
    test_clips = set(test_df["clip_id"].astype(str))

    clip_overlap = train_clips & test_clips

    # ==============================================
    # ⑨ 결과 출력
    # ==============================================

    print()
    print("Random Frame Split Demo")
    print()
    print(f"Train frames: {len(train_df)}")
    print(f"Test frames : {len(test_df)}")
    print()
    print("Subject overlap:", sorted(subject_overlap))
    print(f"Clip overlap count: {len(clip_overlap)} / {len(csv_paths)}")
    print()

    # ==============================================
    # ⑩ 겹침이 있으면 LEAKAGE 경고 (시연이므로 중단하지 않음)
    # ==============================================

    if subject_overlap:
        print("[LEAKAGE] 같은 Subject가 Train과 Test에 존재")

    if clip_overlap:
        print("[LEAKAGE] 같은 Clip의 Frame이 Train과 Test에 존재")

    # ==============================================
    # ⑪ 사용 금지 안내
    # ==============================================
    # 실제 분할은 30번 Subject-wise Split을 사용합니다. 이 코드는 저장하지 않습니다.

    print()
    print("이 Random Frame Split은 최종 평가용으로 사용하지 않습니다.")


# ==================================================
# ⑫ Python 파일 직접 실행 시 main() 호출
# ==================================================

if __name__ == "__main__":
    main()
    