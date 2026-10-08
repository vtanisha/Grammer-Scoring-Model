# Experiment log

Each entry is a hypothesis, the test, the evidence and the decision.

## Summary

**Final engine:** DeBERTa fine-tuned on the clean transcript + DeBERTa fine-tuned on the raw transcript + WavLM audio
embeddings, blended with non-negative weights and linearly calibrated.

| | Pearson | RMSE |
|---|---|---|
| Speaker-grouped, nested 5-fold CV | **0.837** | **0.555** |
| Training (in-sample) | | 0.514 |
| Public leaderboard | | **0.3448** |

Checks passed: label-shuffle null test, nested CV, speaker-grouped folds, accent-proxy fairness (|bias| ≤ 0.10 in
every group).

**Evaluation notes.** CV = stratified 5-fold, repeated 3 times, on the 732 non-noise training clips. Experiments 0–23
used random folds; from experiment 24 onwards (after same-speaker clips were found) all numbers use speaker-grouped
folds, which are about 0.01–0.03 lower for audio models. LB = public leaderboard RMSE on 60 % of the test set;
submission names (`sub05`, …) label the leaderboard runs.

## Experiments

| # | Change | CV Pearson | CV RMSE | LB RMSE | Decision |
|---|---|---|---|---|---|
| 0 | Audio statistics only (quality + VAD) | 0.43 | 0.92 | — | baseline |
| 1 | Text pipeline: Whisper + wav2vec2, LanguageTool, CoLA, spaCy, disfluency, ASR reliability, sentence embeddings; Ridge + LightGBM + embedding Ridge, NNLS blend, linear calibration | 0.783 | 0.630 | 0.4634 | kept |
| 1a | Same, no calibration | 0.783 | 0.649 | 0.4757 | calibration helps |
| 1b | Same, predictions shrunk 30 % toward the mean | — | — | 0.5211 | shrinking hurts |
| 1c | Constant mean prediction (probe) | — | — | 1.0231 | test labels spread ≈ 1.0, like train |
| 2 | + advanced structures, structure variety, correct advanced use, systematic errors, coherence | 0.783 | 0.631 | — | no gain |
| 3 | + fixed-window answer quantity; isotonic calibration; stacking | 0.783 | 0.630–0.633 | — | no gain |
| 4 | Adversarial validation (classifier: train vs test) | AUC 0.725 | | | recording shift found |
| 5 | + SVR, ExtraTrees, regularised LightGBM; Whisper/wav2vec2 inflection disagreement | 0.785 | 0.628 | — | small gain |
| 6 | #5 without the shifted recording features | 0.774 | 0.642 | 0.4982 | rejected |
| 7 | Fine-tuned DeBERTa-v3-base on clean transcripts, alone | 0.817 | 0.597 | — | strongest single model |
| 8 | #5 + DeBERTa (blend weight 0.64) | 0.830 | 0.565 | 0.3637 | kept |
| 9 | #8 without the shifted recording features | 0.826 | 0.571 | 0.3690 | rejected |
| 10 | Local LLM judge (Qwen2.5-7B, 4 rubric dimensions) as features | 0.829 | 0.567 | 0.3632 | no gain, dropped |
| 11 | DeBERTa on raw transcripts (fillers and repeats kept), alone | 0.813 | 0.599 | — | 0.96 correlated with clean |
| 12 | #8 + raw-transcript DeBERTa | 0.8325 | 0.562 | 0.3629 | kept |
| 13 | Second clean DeBERTa seed, alone | 0.806 | 0.611 | — | seed spread ≈ 0.01 |
| 14 | Clean DeBERTa averaged over 2 seeds, alone | 0.820 | 0.590 | — | averaging beats either seed |
| 15 | Blend with seed-averaged DeBERTa (and 3-run variant) | 0.832–0.833 | 0.561–0.563 | 0.3623–0.3631 | differences are noise |
| 16 | DeBERTa-v3-large | — | — | — | does not fit in 16 GB |
| 17 | Leakage audit: train on shuffled labels | r ≈ 0 | | | pass |
| 18 | Leakage audit: nested CV for blend weights + calibration | 0.8295 vs 0.8319 | 0.566 | | pass (optimism 0.002) |
| 19 | Random-noise features; Boruta-style in-fold selection | 0.738 → 0.710 | | | keep all features |
| 20 | Frozen WavLM-base-plus embeddings (layers 6/9/12, PCA 64 + ridge), alone | 0.808 | 0.598 | — | audio ≈ text, 0.81 correlated with DeBERTa |
| 21 | Blend + WavLM (nested) | 0.856 | 0.524 | 0.3488 | kept |
| 22 | DeBERTa ×2 + WavLM only, no hand-built features (nested) | 0.856 | 0.525 | 0.3504 | same accuracy, simpler |
| 23 | Accent-proxy fairness (ASR-confidence tertiles, 4 voice clusters) | | | | error drops in every group |
| 24 | Speaker check (WavLM-SV voice embeddings): nearest voice within ±3 file IDs 48.8 % of the time (chance 1.3 %) | | | | same-speaker clips found |
| 25 | Random vs speaker-grouped folds: text −0.00…−0.01, **WavLM 0.808 → 0.777** | | | | grouped folds adopted |
| 26 | Fine-tuned WavLM (attention pooling, band-reweighted loss) | — | — | — | does not fit in 16 GB |
| 27 | DeBERTa raw, grouped folds | 0.805 | 0.611 | — | |
| 28 | DeBERTa clean, grouped folds | 0.810 | 0.605 | — | |
| 29 | **Final engine: DeBERTa clean + raw + WavLM, grouped nested CV** | **0.837** | **0.555** | **0.3448** | **final** |
| 30 | Same + hand-built features | 0.837 | 0.554 | — | features used for interpretation only |

## Findings

**Score-0 clips are noise, not grammar.** 37 training clips are steady white noise (dynamic range ≈ 3 dB vs ≈ 40 dB
for speech). A three-statistic rule flags exactly these and no test clip, so they are excluded from training.

**Train and test do not overlap.** They reuse filenames, but the audio differs (cross-correlation at chance level)
and transcripts share no content (median 6-gram overlap 0 %).

**The test set is as spread out as train.** A constant prediction scores 1.02 on the leaderboard, so calibration
(stretching predictions to the label range) helps rather than hurts.

**ASR confidence is the strongest hand-built signal** (+0.43 Pearson in the ablation). Clearer, more proficient
speakers get more confident transcripts, so it measures intelligibility as well as grammar.

**Rule-based grammar checking is weak on speech.** LanguageTool adds +0.01; most of its hits are spelling and style.

**Advanced-grammar counts don't separate speakers.** Inversion, modal perfects and conditionals almost never occur in
one-minute spontaneous answers (≤ 0.15 per 100 words); common clause types are already covered by syntax features.

**Answer length and calibration were not the bottleneck.** Fixed-window word counts, isotonic calibration and stacking
all left CV unchanged, so the remaining error came from what the features could see.

**Train and test were recorded differently** (adversarial AUC 0.725, driven by dynamic range, noise floor and speech
rate). Dropping those features made the leaderboard worse (0.3637 → 0.3690), so they carry real signal on test too.

**Whisper hides some errors.** Where wav2vec2 (no language model) hears a different inflection or function word, the
score tends to be lower (r ≈ −0.13).

**Letting a model read the transcripts was the first breakthrough.** Fine-tuned DeBERTa alone (r 0.817) beat the
whole hand-built feature blend (0.785) and moved the leaderboard from 0.463 to 0.362.

**A local LLM judge is redundant next to DeBERTa.** Qwen2.5-7B scoring the rubric directly correlates 0.50–0.61 with
the labels but adds nothing to the blend. Scoring four rubric questions off one cached transcript prefix cut its
runtime from ~5 h to ~1.5 h.

**The evaluation is honest.** Training on shuffled labels gives r ≈ 0; nested CV for the blend costs only 0.002.

**Audio and text are complementary**, as the spoken-assessment literature reports (Speak & Improve 2025, NTNU
system: wav2vec 2.0 0.394, text model 0.389, fused 0.375 RMSE). Frozen WavLM embeddings alone nearly match DeBERTa
and correlate only 0.81 with it. Adding audio moved the leaderboard from 0.362 to 0.345. With both modalities in, the
hand-built features add nothing, so the engine is *a model that reads the transcript and a model that listens to the
speech*.

**Neighbouring file IDs share a speaker or session, and random folds let the audio model recognise voices.** Under
speaker-grouped folds WavLM drops from 0.808 to 0.777 while text models barely move. A real engine always meets new
speakers, so grouped folds became the official evaluation. The grouped-fold engine also scored best on the
leaderboard (0.3448 vs 0.3488 for the random-fold version).

**The engine treats accented speakers fairly.** Using Whisper confidence and early-layer WavLM voice clusters as
accent proxies, adding audio lowered error in every group and mean bias stays within ±0.10 points. Lower-clarity
speakers have lower true scores on average (2.80 vs 4.26), but the engine does not score them below the raters.

**Hardware set the ceiling.** DeBERTa-large and fine-tuned WavLM both exceed 16 GB of unified memory. Running two
GPU jobs at once caused a system restart; the fix was one heavy job at a time, audio read lazily from disk, per-fold
checkpoints and a watchdog that stops training if swap passes 6 GB.

## Design reviews

Twice, two opposing proposals were written up (one for model-side changes, one for data/audio-side changes) and the
cheapest decisive tests were run first.

* **Round 1.** Model side: fine-tune a transformer on transcripts. Data side: restore answer quantity, measure errors
  Whisper hides, replace LanguageTool. Cheap data-side items failed (#2–#3), which pointed to the model side (#7).
* **Round 2.** Model side: task-adaptive pretraining, ordinal loss, more seeds. Audio side: fine-tune a speech model,
  train on 45 s windows. The spoken-assessment literature favoured fusing a speech model with the text model; frozen
  WavLM was the cheap test (#20), and gave the largest late gain.

## Next steps (not done here)

* Fine-tune WavLM and try DeBERTa-large on a GPU with ≥ 24 GB.
* Label clips with true accent / first-language information to replace the accent proxies.
* Separate grammar from fluency in the labels if a pure grammar score is needed.
