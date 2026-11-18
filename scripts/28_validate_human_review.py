from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
# argparse: 실행 옵션 / Path: 경로 / pandas: CSV 검사

import argparse
from pathlib import Path

import pandas as pd


# ==================================================
# ② 프로젝트 ROOT와 Human QA 기본 파일 경로 (day05 → 05_dataset_qa)
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REVIEW_PATH = (
    ROOT
    / "data"
    / "05_dataset_qa"
    / "qa"
    / "keypoint_clip_review.csv"
)


# ==================================================
# ③ 허용되는 Human QA 최종 판정값
# ==================================================
# PASS: 사용 / RESHOOT: 재촬영 / EXCLUDE: 제외
# UNREVIEWED는 아직 검토 전이라 최종 판정이 아닙니다.

ALLOWED = {
    "PASS",
    "RESHOOT",
    "EXCLUDE",
}


# ==================================================
# ④ 터미널 실행 옵션 설정
# ==================================================
# --expected-clips: 0이면 Review 행 수를 그대로 사용
# --check-only: CHECK인 Clip만 사람 검토를 필수로 봄
#   (기본은 전체 검토 — 바른품은 27번 기준이 아직 검증 전이라 전부 확인)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--review-path", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument("--expected-clips", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review_path = Path(args.review_path)

    # ==============================================
    # ⑤ Human QA 파일 존재 여부 확인
    # ==============================================

    if not review_path.exists():
        print("[ERROR] Human QA 파일을 찾을 수 없습니다. 먼저 27번을 실행하세요.")
        print(review_path)
        raise SystemExit(1)

    # ==============================================
    # ⑥ Review CSV 읽기 및 Clip 개수 검사
    # ==============================================

    df = pd.read_csv(review_path)

    if args.expected_clips and len(df) != args.expected_clips:
        print("[ERROR] Human QA Clip 수가 예상과 다릅니다.")
        print(f"expected={args.expected_clips}, actual={len(df)}")
        raise SystemExit(1)

    # ==============================================
    # ⑦ human_decision 값을 비교하기 쉽게 정리
    # ==============================================
    # " pass " → "PASS" (원본 CSV는 수정하지 않음)

    decisions = (
        df["human_decision"].fillna("").astype(str).str.strip().str.upper()
    )

    qa_status = df["qa_status"].fillna("").astype(str).str.strip().str.upper()

    # ==============================================
    # ⑧ 검토가 필요한 Clip 정하기
    # ==============================================
    # 기본: 전체 Clip이 검토 대상
    # --check-only: qa_status가 CHECK인 Clip만 검토 대상

    if args.check_only:
        required_mask = qa_status == "CHECK"
    else:
        required_mask = pd.Series(True, index=df.index)

    # ==============================================
    # ⑨ 검토 대상인데 아직 판정이 없는 Clip 찾기
    # ==============================================

    unreviewed_mask = required_mask & ((decisions == "") | (decisions == "UNREVIEWED"))

    show_columns = [
        column
        for column in ["clip_id", "video_file", "qa_status", "check_reasons", "human_decision"]
        if column in df.columns
    ]

    # ==============================================
    # ⑩ 미검토 Clip 목록 출력 및 중단
    # ==============================================
    # video_file(원래 영상 이름)을 같이 보여 줘서 mp4를 바로 찾을 수 있게 합니다.

    if unreviewed_mask.any():
        print()
        print("[HUMAN QA NOT COMPLETE]")
        print()
        print(df.loc[unreviewed_mask, show_columns].to_string(index=False))
        print()
        print(f"남은 Clip: {int(unreviewed_mask.sum())}")
        print()
        print("원본 MP4를 확인한 뒤 human_decision을")
        print("PASS / RESHOOT / EXCLUDE 중 하나로 입력하세요.")
        raise SystemExit(1)

    # ==============================================
    # ⑪ 허용되지 않은 판정값 검사
    # ==============================================
    # 빈칸·UNREVIEWED가 아닌데 PASS/RESHOOT/EXCLUDE도 아닌 값 → 오류
    # (--check-only에서 AUTO_PASS의 UNREVIEWED는 허용)

    filled_mask = (decisions != "") & (decisions != "UNREVIEWED")

    invalid_mask = filled_mask & ~decisions.isin(ALLOWED)

    if invalid_mask.any():
        print()
        print("[INVALID HUMAN DECISION]")
        print()
        print(df.loc[invalid_mask, show_columns].to_string(index=False))
        print()
        print("human_decision은 PASS / RESHOOT / EXCLUDE만 사용할 수 있습니다.")
        raise SystemExit(1)

    # ==============================================
    # ⑫ 판정별 개수 및 완료 출력
    # ==============================================

    mode = "CHECK only" if args.check_only else "all clips"

    print()
    print(f"[Human QA Result] ({mode})")
    print(decisions.value_counts())
    print()
    print(f"Reviewed clips: {int(filled_mask.sum())} / {len(df)}")
    print("Required human review: OK")


# ==================================================
# ⑬ Python 파일 직접 실행 시 main() 호출
# ==================================================

if __name__ == "__main__":
    main()
    