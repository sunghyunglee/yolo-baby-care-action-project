# 협업 규칙 — yolo-baby-care-action-project (바른품)

> 관리자(PM): Sung Hyung Lee · v1 (2026-11-17)
> 팀원은 처음 들어오면 이 문서를 먼저 읽어 주세요.

## 1. 역할

| 역할 | 하는 일 |
|---|---|
| PM (Sung Hyung Lee) | `main` 관리 · Pull Request 확인 후 합치기 · 규칙 관리 |
| 팀원 | 각자 작업 브랜치에서 작업 → Pull Request 올리기 |

## 2. 브랜치 규칙

```
main  ← PM만 합침 (직접 push 금지)
 ├─ feature/할일이름   새 기능   예: feature/detection, feature/interaction-feature
 ├─ fix/고칠것         오류 수정  예: fix/sequence-length
 └─ docs/문서이름      문서      예: docs/readme
```

- `main`에는 직접 push하지 않습니다. 항상 브랜치 → Pull Request → PM 확인 → 합치기 순서입니다.
- 브랜치 하나 = 할 일 하나. 끝나서 합쳐지면 브랜치는 지웁니다.
- 작업 시작 전에 항상 최신 `main`을 받고 브랜치를 만듭니다.

## 3. 커밋 메시지

| 머리말 | 뜻 | 예 |
|---|---|---|
| `feat:` | 새 기능 | `feat: add adult-baby bbox overlap check` |
| `fix:` | 오류 수정 | `fix: handle missing wrist keypoint` |
| `docs:` | 문서 | `docs: update README run steps` |
| `refactor:` | 동작 그대로, 코드 정리 | `refactor: split feature functions` |
| `chore:` | 설정·패키지 | `chore: add tqdm to pyproject` |

## 4. 작성자 · AI 표시

- 커밋 작성자는 **본인 GitHub 계정의 이름·이메일**로 설정합니다. (안 하면 Contributors에 안 나옵니다)
- 커밋 메시지·PR 설명에 `Co-Authored-By: Claude ...`, `Claude-Session: ...`, `Generated with Claude Code` 같은 **AI 표시 줄을 넣지 않습니다.**

## 5. 올리지 않는 것

`.venv/` · `data/` · `reports/` · `artifacts/` · `*.pt` · 영상(mp4)·이미지 · 생성 CSV · **촬영 원본(사람 얼굴 포함)**

- Git에는 코드와 문서만 올립니다. `git add` 전에 반드시 `git status`로 확인합니다.
- 패키지는 `uv`로 관리하고 `uv.lock`은 커밋합니다. (`pip install` 대신 `uv add`)

## 6. 팀원 작업 순서

처음 한 번:

```bash
git clone https://github.com/sunghyunglee/yolo-baby-care-action-project.git
cd yolo-baby-care-action-project
git config user.name "본인 GitHub 이름"
git config user.email "본인 GitHub 이메일"
uv sync
```

매번 작업할 때:

```bash
git switch main
git pull
git switch -c feature/할일이름
# ... 작업 ...
git status
git add 바꾼파일
git commit -m "feat: 무엇을 했는지"
git push -u origin feature/할일이름
```

→ GitHub에 뜨는 **Compare & pull request** 버튼 → 무엇을·왜 바꿨는지 적고 PR 생성 → PM 확인 후 합치기

## 7. Pull Request 체크리스트

- [ ] 최신 `main`에서 브랜치를 만들었다
- [ ] `git status`로 data·영상·*.pt가 빠진 것을 확인했다
- [ ] 프로젝트 루트에서 실행해 보고 오류가 없었다
- [ ] 커밋 작성자가 본인 계정이다 · AI 표시 줄이 없다
- [ ] PR 설명에 바꾼 내용·실행 방법·결과(숫자나 캡처)를 적었다
