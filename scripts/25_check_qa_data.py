from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# csv: 결과 저장 / re: 파일명 규칙 검사 / Path: 경로 / cv2: 영상 정보

import csv
import re
from pathlib import Path

import cv2


# ==================================================
# ② 프로젝트 ROOT와 입출력 경로 (day05 → 05_dataset_qa)
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_ROOT = ROOT / "data" / "05_dataset_qa" / "videos"

ACTION_FILE = ROOT / "configs" / "action_classes.csv"

OUTPUT_PATH = ROOT / "reports" / "05_dataset_qa" / "tables" / "video_inventory.csv"


# ==================================================
# ③ Subject 기준과 파일명 규칙 (강사님 규칙)
# ==================================================
# 파일명: Subject_Action_Take.mp4  예) S01_A008_T01.mp4
# 강사님처럼 18개·A001~A006 고정이 아니라,
# action_classes.csv에 있는 동작(A001~A012)이면 통과합니다.

SUBJECTS = ["S01", "S02", "S03"]

FILENAME_PATTERN = re.compile(r"^(S\d{2})_(A\d{3})_(T\d{2})\.mp4$", re.IGNORECASE)


# ==================================================
# ④ action_classes.csv에서 등록된 Action ID 읽기
# ==================================================

def load_action_ids() -> set[str]:
    if not ACTION_FILE.exists():
        raise FileNotFoundError(f"{ACTION_FILE.relative_to(ROOT)} 가 없습니다.")

    with ACTION_FILE.open("r", encoding="utf-8-sig") as file:
        return {row["action_id"].strip().upper() for row in csv.DictReader(file)}


# ==================================================
# ⑤ MP4 하나를 열어 Metadata 읽기 (강사님 코드 그대로)
# ==================================================

def read_video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        return {
            "open_ok": 0,
            "read_ok": 0,
            "fps": 0.0,
            "frames": 0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    success, first_frame = capture.read()
    capture.release()

    read_ok = int(success and first_frame is not None)
    duration = frames / fps if fps > 0 else 0.0

    return {
        "open_ok": 1,
        "read_ok": read_ok,
        "fps": round(fps, 3),
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
    }


def main() -> None:

    # ==============================================
    # ⑥ 입력 폴더·Action 목록 확인
    # ==============================================

    if not VIDEO_ROOT.exists():
        raise FileNotFoundError(f"영상 폴더가 없습니다: {VIDEO_ROOT.relative_to(ROOT)}")

    action_ids = load_action_ids()

    rows = []
    problems = []

    # ==============================================
    # ⑦ MP4마다 파일명 규칙·폴더·Action·Metadata 검사
    # ==============================================
    # - 파일명이 S01_A008_T01.mp4 규칙인가?
    # - 파일명 Subject와 폴더 이름이 같은가?
    # - Subject가 S01~S03인가?
    # - Action이 action_classes.csv에 있는가?
    # - 영상이 열리고 FPS·Frame·해상도가 정상인가?

    for path in sorted(VIDEO_ROOT.rglob("*.mp4")):
        match = FILENAME_PATTERN.match(path.name)

        if match is None:
            problems.append(f"파일명 규칙 오류: {path.relative_to(ROOT)}")
            continue

        subject, action_id, take = (group.upper() for group in match.groups())

        if path.parent.name != subject:
            problems.append(f"폴더와 파일명 Subject가 다름: {path.relative_to(ROOT)}")

        if subject not in SUBJECTS:
            problems.append(f"SUBJECTS에 없는 Subject: {subject} ({path.name})")

        if action_id not in action_ids:
            problems.append(f"action_classes.csv에 없는 Action: {action_id} ({path.name})")

        info = read_video_info(path)

        metadata_ok = (
            info["open_ok"] == 1
            and info["read_ok"] == 1
            and info["fps"] > 0
            and info["frames"] > 0
            and info["width"] > 0
            and info["height"] > 0
        )

        if not metadata_ok:
            problems.append(f"영상 읽기 또는 Metadata 오류: {path.name}")

        print(
            f"[{'OK' if metadata_ok else 'FAIL'}] {path.name:20s} "
            f"{info['width']}x{info['height']} "
            f"fps={info['fps']:.3f} frames={info['frames']}"
        )

        rows.append(
            {
                "clip_id": path.stem,
                "subject": subject,
                "action_id": action_id,
                "take": take,
                "video_file": path.name,
                "video_path": str(path.relative_to(ROOT)),
                **info,
            }
        )

    # ==============================================
    # ⑧ Subject 누락·전체 오류 확인
    # ==============================================

    for subject in SUBJECTS:
        if not any(row["subject"] == subject for row in rows):
            problems.append(f"{subject} 영상이 0개입니다.")

    if problems:
        for problem in problems:
            print("[ERROR]", problem)
        raise RuntimeError("5일차 입력 영상을 다시 확인하세요.")

    # ==============================================
    # ⑨ Subject별·Action별 개수 확인
    # ==============================================
    # 사람마다 같은 동작을 찍었는지 확인합니다 (Split에서 동작이 빠지지 않게).

    actions_by_subject = {
        subject: sorted(row["action_id"] for row in rows if row["subject"] == subject)
        for subject in SUBJECTS
    }

    for subject, actions in actions_by_subject.items():
        print(f"{subject}: {', '.join(actions)}")

    if len({tuple(actions) for actions in actions_by_subject.values()}) > 1:
        print("[WARN] Subject마다 찍은 동작 구성이 다릅니다. 30번 Split에서 동작이 빠질 수 있습니다.")

    # ==============================================
    # ⑩ 영상 목록표 CSV 저장
    # ==============================================

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Total clips: {len(rows)}")
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Dataset QA video data: READY")


# ==================================================
# ⑪ Python 파일 직접 실행 시 main() 호출
# ==================================================

if __name__ == "__main__":
    main()
    