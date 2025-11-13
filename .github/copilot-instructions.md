# GitHub Copilot Instructions

## Project Context
- NeuroCombat is a Python 3.10+ pipeline that ingests MMA fight footage, extracts MediaPipe poses, tracks players, classifies moves, and produces templated commentary.
- Core modules live under `backend/` (e.g., `pose_extractor.py`, `tracker.py`, `move_classifier.py`, `commentary_engine.py`, `utils.py`). Streamlit UI (`app.py`) and CLI (`main.py`) orchestrate the pipeline.
- Large media assets and generated artifacts are intentionally **not** tracked in git (`data/raw`, `data/processed`, `artifacts`, `temp_uploads`). Keep examples lightweight or mocked.

## Coding Guidelines
- Favor dataclasses or lightweight typed containers when passing structured pose or classification data (see `PoseLandmarks` in `backend/pose_extractor.py`).
- Use `backend.utils.setup_logging` for new loggers; keep log output informative but concise.
- Add docstrings and inline comments only where logic is non-obvious. Follow existing narrative/emoji style in user-facing CLI prints.
- Prefer type hints and explicit return values. Keep functions small and composable.
- When touching video I/O, ensure OpenCV resources are released (`cap.release()`, `cv2.destroyAllWindows()`) and guard against missing files.

## Implementation Patterns
- Pose pipeline order is fixed: extract poses → track players → classify moves → generate commentary → annotate frame. Respect this sequencing when adding features.
- Tracking relies on IoU thresholds and `max_missing_frames`; avoid introducing global state outside tracker classes.
- Move classification currently supports both mock and ML-backed modes. New classifiers should expose the same interface (`classify_move`, `window_size`, `confidence_threshold`).
- Commentary events are objects with `timestamp`, `text`, `event_type`, `players_involved`. Maintain this schema for downstream compatibility.

## Testing & Tooling
- Lightweight smoke tests live in `test_pose_extraction.py` and helper scripts (`run_pose_extraction.py`, `run_move_classification.py`, etc.). Favor small, deterministic samples or mocks so tests run without large media files.
- If you add functionality that depends on additional packages, update `requirements.txt` and mention optional dependencies in relevant READMEs.
- Manual verification is usually done by running `streamlit run app.py` (UI) or `python main.py --video <path>` (CLI). Document new flags or configuration keys in `README.md` and `config.py` comments.

## Operational Notes
- Respect the `.gitignore`: do not reintroduce large binaries under `data/` or `artifacts/`.
- Keep generated commentary/text assets in `artifacts/` only when necessary for debugging; otherwise prefer deterministic generation.
- When unsure about pose formats, reference MediaPipe conventions (33 keypoints with xyz + visibility).
- Prefer ASCII characters in source files unless extending existing emoji-rich CLI output.
