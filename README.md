# Grammar Scoring Engine for Spoken English

Predicts a 0–5 grammar score from a 45–60 second recording of spoken English, built for the
[SHL Hiring Assessment 2026](https://www.kaggle.com/competitions/shl-hiring-assessment-2026) Kaggle challenge.

## Approach

The engine has two halves: a model that **reads** what was said and a model that **listens** to how it was said.

```
audio ─► noise filter ─► voice activity detection ─┬─► Whisper medium + wav2vec2 ─► clean and raw transcripts
                                                   │                                   │
                                                   │                  DeBERTa (clean)  │  DeBERTa (raw)
                                                   │                                   │
                                                   └─► WavLM audio embeddings ─► ridge ┤
                                                                                       ▼
                                                            weighted blend ─► calibration ─► score 0–5
```

1. **Audio cleanup.** 37 training clips scored 0 are pure noise (dynamic range ≈ 3 dB vs ≈ 40 dB for speech); they
   are excluded from training. Silero VAD finds the speech in each clip.
2. **Speech to text.** Whisper medium gives the main transcript with per-word confidence. wav2vec2 (no language
   model) gives a literal second transcript, because Whisper tends to quietly correct grammar errors.
3. **Two transcript views.** *Raw* keeps fillers, repeats and restarts; *clean* removes them, so self-corrections
   (rewarded by the rubric) are not counted as errors.
4. **Text models.** DeBERTa-v3-base fine-tuned to predict the score, once on each view.
5. **Audio model.** WavLM-base-plus hidden states (layers 6, 9, 12) mean-pooled over speech, then PCA + ridge.
   Captures rhythm, hesitation and clarity that transcripts lose.
6. **Blend.** Non-negative weights (≈ 0.39 clean text, 0.20 raw text, 0.41 audio) and a linear calibration.

Hand-built grammar features (LanguageTool errors, CoLA acceptability, spaCy syntax, disfluency rates, ASR confidence)
are computed for interpretation and ablation; once the text and audio models are in, they add < 0.001 Pearson.

## Results

Speaker-grouped, nested 5-fold cross-validation on 732 training clips:

| Model | Pearson | RMSE |
|---|---|---|
| Hand-built features only | 0.73 | 0.69 |
| Text only (2 × DeBERTa) | 0.812 | 0.593 |
| Audio only (WavLM) | 0.774 | 0.642 |
| **Engine: text + audio** | **0.837** | **0.555** |
| Training RMSE | | **0.514** |

Public leaderboard RMSE: **0.3448** (0.4634 with hand-built features only, 0.3623 before adding audio).

## Evaluation checks

* **Label-shuffle test:** every model trained on shuffled scores falls to r ≈ 0, so nothing leaks the target.
* **Nested CV:** blend weights and calibration are learned inside each outer fold.
* **Speaker grouping:** voice embeddings showed neighbouring file IDs share a speaker or session. Folds keep blocks of
  25 IDs together so no voice is in both training and validation.
* **Accent robustness:** using Whisper confidence and voice clusters as accent proxies, error drops in every group
  when audio is added, and mean bias stays within ±0.10 points.

The full experiment history, including approaches that did not work, is in
[`development/EXPERIMENTS.md`](development/EXPERIMENTS.md).

## Repository

| File | Contents |
|---|---|
| `grammar_scoring.ipynb` | Full pipeline with executed outputs, plots and the written report (Section 11) |
| `submission.csv` | Test-set predictions |
| `cache/` | Saved transcripts, embeddings and model predictions, so the notebook reruns in ~2 minutes |
| `environment.yml` | Conda environment |
| `development/EXPERIMENTS.md` | Experiment log |

## Running

```bash
conda env create -f environment.yml
conda activate grammar
python -m spacy download en_core_web_sm
```

Download the competition data into `Dataset_Final/` (or let the notebook do it on Colab with a Kaggle token), then
run `grammar_scoring.ipynb`. With `cache/` present, every slow step is loaded from disk. Without it, transcription
and fine-tuning take roughly 3–4 hours on an Apple M4 or about 1 hour on a Colab T4 GPU.
