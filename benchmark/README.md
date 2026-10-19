# Benchmark

How well does an agent segment a structure named in plain text, with nnInteractive as its tool?
13 public images (CT, MRI, PET and micro-CT; people, a mouse and an ant), 34 targets, none of them
used to train nnInteractive. Every target is something an expert finds from the request alone: it
is named, unique, and its side is given where that matters.

We ship no data. `prepare.py` builds every image and reference mask from its original source,
which you download yourself (most of it automatically), and checks the result against published
checksums, so everyone runs on the same benchmark.

## Tracks

| Track | The agent gets | Measures |
|---|---|---|
| `unguided` | the image and the list of targets by name | finding and segmenting |
| `guided` | the same, plus one voxel inside each target | segmenting, given where |
| `point` | no agent: one positive point at that voxel per target | nnInteractive alone, for comparison |
| `oracle-points` | no agent: up to 5 simulated points per target, placed with the reference mask | what a user who knows the answer gets |
| `oracle-scribbles` | the same with simulated scribbles | the same |
| `voxtell` | no agent: [VoxTell](https://github.com/MIC-DKFZ/VoxTell) with each target's text as its prompt | a text-promptable 3D model, for comparison |

The guided voxel is the point deepest inside the reference mask. The request lists the targets in
order and asks the agent to commit each object before it starts the next.

**Oracles:** each prompt goes to the largest error of the current mask, the largest connected
region it misses (a positive prompt) or adds (a negative one). Before the first prompt that is
the whole target. `oracle-points` puts a point at the voxel deepest inside that region (the first
point is the guided voxel). `oracle-scribbles` draws a line along the middle of that region (the
longest path through its skeleton) in the slice where the region is largest; a region too small
for a line gets a point. Large objects are passed to `new_object` with their extent, as the skill
tells agents to; thin ones (a vessel tree) stay at full resolution. The code is in `oracle.py`.
The oracles use the same tools as the agents and show how far an agent's prompts are from ideal
ones. They are compared with an agent only at the same number of prompts: `summary` gives each
agent's result next to the oracles' Dice after as many prompts on each target as the agent sent
(after five where the agent sent more).

**VoxTell:** a 3D segmentation model that takes free-text prompts (CT, MRI, PET). It gets exactly
the target texts the agents get ("the tumor in the right lung"), all targets of an image in one
call. It runs in its own environment: `pip install voxtell` there, then pass its command to
`--voxtell`. Its weights are CC BY-NC-SA 4.0.

**Score:** Dice per target, with the target's object as it was when the agent last committed it,
so a later object that takes voxels from an earlier one (a tumour inside the liver) does not count
against the earlier one. A target without an object scores 0. A run's score is the mean over all
targets; `summary` also reports it per tier (E easy, M medium, H hard) and per image, and for the
oracles after every prompt.

## Images

| Image | Targets | Source | Access |
|---|---|---|---|
| `pediatric_ct` | liver, spleen, right kidney, bladder, heart, pancreas, gallbladder, left adrenal | [Pediatric-CT-SEG](https://doi.org/10.7937/TCIA.X0H0-1706) (TCIA), CC BY 4.0 | download |
| `hcc_ct` | liver, liver tumour | [HCC-TACE-Seg](https://doi.org/10.7937/TCIA.5FNA-0924) (TCIA), CC BY 4.0 | download |
| `adrenal_ct` | adrenal tumour | [Adrenal-ACC-Ki67-Seg](https://doi.org/10.7937/1FPG-VM46) (TCIA), CC BY 4.0 | download |
| `lung_ct` | lung tumour | [RIDER Lung CT](https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR) (TCIA), CC BY 4.0 | download |
| `lung_pet` | FDG-avid tumour | [ACRIN-NSCLC-FDG-PET](https://doi.org/10.7937/tcia.2019.30ilqfcl) (TCIA), CC BY 3.0, with the [BAMF/AIMI](https://doi.org/10.5281/zenodo.13244892) lesion mask, CC BY 4.0 | download |
| `headneck_mr` | mandible, brainstem, left parotid, right submandibular gland, thyroid, spinal cord, left optic nerve | [HaN-Seg](https://doi.org/10.5281/zenodo.7442914), CC BY-NC-ND 4.0 | download |
| `headneck_tumor_mr` | primary tumour | [HNTS-MRG 2024](https://doi.org/10.5281/zenodo.11199559), CC BY-NC 4.0 | download |
| `bladder_mr` | bladder tumour | [FedBCa](https://doi.org/10.5281/zenodo.13622759), CC BY 4.0 | download (1 GB RAR) |
| `thigh_mr` | rectus femoris, sartorius, gracilis, vastus lateralis | [Thigh and calf muscle MRI](https://doi.org/10.6084/m9.figshare.31042489.v1), CC BY 4.0, non-commercial | by hand |
| `calf_mr` | soleus, tibialis anterior, medial gastrocnemius | same | by hand |
| `liver_vessels_ct` | liver, portal vein, hepatic veins | [VEELA 2025](https://www.synapse.org/Synapse:syn65471967/wiki/) (images from CHAOS, CC BY-NC-SA 4.0) | Synapse login |
| `mouse_microct` | subcutaneous tumour | [TumSeg](https://doi.org/10.17894/UCPH.F7BCF864-BE18-4A16-95AD-6F22DEDB4265) (ERDA), CC BY | download |
| `ant_microct` | brain | [Ant brains in micro-CT](https://doi.org/10.5061/dryad.qz612jmgv) (Dryad), CC0 | browser |

`cases.json` names the exact case, series or file of each source and how each reference is made.
Most references are the source's mask as it is. Where they are made:

- `hcc_ct` liver: the source's liver, tumour and portal vein segments together (the whole liver).
- `liver_vessels_ct` liver: the liver mask plus the vessel voxels next to it (the whole liver).
- `headneck_mr`: HaN-Seg draws its organs on the CT. They are moved onto the MR with one fixed rigid
  transform (in `cases.json`), so everyone gets the same masks.
- `lung_pet`: the image is SUV (body weight), decay-corrected to the start of the scan.
- `ant_microct`: the release has no slice 101 of this stack; it is left out. The voxel size is
  unknown (the slices were resized), so the image has 1 x 1 x 1 spacing.

## Results

Our runs on all 13 images (October 2026):
- **nnInteractive:** the official weights (v1.0), on this machine, one RTX 4090.
- **Agents:** Claude Code 2.1 with Opus 5.5 and Codex 0.162.1 with gpt-6.1-sol, both at medium
  effort, three repeats per track (± is the spread of the track's mean over repeats). The
  45-minute limit per image was never reached and no target was left without an object.
- **No-agent tracks** (point, oracles, VoxTell v1.1): run once. Local GPU inference is not exactly
  repeatable; their per-image results vary by about 0.01 between runs.

| Track | Mean Dice | Easy (9) | Medium (20) | Hard (5) | Prompts per target | Oracle, same prompts: scribbles / points | Minutes per image |
|---|---|---|---|---|---|---|---|
| Codex, guided | 0.815 ± 0.004 | 0.917 | 0.824 | 0.598 | 2.8 | 0.848 / 0.803 | 3.2 |
| Claude Code, guided | 0.811 ± 0.005 | 0.917 | 0.815 | 0.603 | 2.3 | 0.813 / 0.767 | 2.7 |
| Codex, unguided | 0.809 ± 0.005 | 0.922 | 0.811 | 0.597 | 2.8 | 0.859 / 0.788 | 3.3 |
| Claude Code, unguided | 0.772 ± 0.004 | 0.911 | 0.774 | 0.517 | 2.5 | 0.832 / 0.787 | 3.4 |
| One point, no agent | 0.654 | 0.601 | 0.719 | 0.487 | 1 | 0.726 / 0.654 | 0.1 |
| VoxTell, text only | 0.632 | 0.940 | 0.465 | 0.746 | 0 | | 0.3 |

"Oracle, same prompts" is the oracles' Dice after as many prompts on each target as the agent sent
for it; where the agent sent more than five (6% to 17% of the targets), after five.

What we read from it:
- **An agent makes nnInteractive far better than one click on the same targets**, even when the
  agent has to find the targets itself: 0.77 to 0.82 against 0.65. The gain is largest where one
  point does not describe the target: two of the three livers (one point 0.00 and 0.61, the agents
  0.95 to 0.97), the liver tumour (0.13 against 0.96) and the ant brain.
- **At the same number of prompts, the agents' prompts are close to the oracles'.** The scribble
  oracle is 0.002 ahead of Claude Code (guided) and 0.033 ahead of Codex (guided); the point oracle
  is behind both. Unguided, the scribble oracle is 0.05 to 0.06 ahead, which includes the targets
  the agents did not find exactly.
- **The agents stop early.** They send 2.3 to 2.8 prompts per target, while the oracles keep
  improving with every further prompt.
- **Hard targets stay hard:** the portal and hepatic veins (agents 0.27 to 0.53), the optic nerve
  (0.02 to 0.34) and the spinal cord (0.46 to 0.55): thin structures that are hard to prompt and to
  judge from rendered views.
- **VoxTell, from the text alone, is strong on standard radiology and fails outside it.** On the 25
  targets of the head-and-neck, chest, abdominal and pelvic images it reaches 0.82, more than the
  agents (0.77 to 0.79), and it does well on thin structures the agents find hard (optic nerve
  0.59, spinal cord 0.61, hepatic veins 0.64). On the leg muscles it reaches 0.11 (agents 0.77 to
  0.88), on the mouse and the ant 0.17 (agents 0.80 to 0.92): structures, sequences and species it
  was not trained on. Several source datasets of the first group are in VoxTell's training data
  (HaN-Seg, HNTS-MRG, RIDER Lung CT, HCC-TACE-Seg), so those numbers may be optimistic. It is also
  sensitive to wording: "the tumor of the right adrenal gland" gives an empty mask, "adrenal tumor"
  Dice 0.79 (a check outside the benchmark). An agent with nnInteractive works on any of these, at
  the price of minutes per image.
- **Codex and Claude Code** are about equal with the guided voxel. Unguided, Codex is ahead (0.809
  against 0.772), mostly on the calf muscles and the ant brain.
- **Cost:** Claude Code reported $1.04 (guided) and $1.23 (unguided) per image at API prices, with
  about 2 to 2.5 million cached input tokens per image. Codex used about 1.5 million input tokens
  per image, 95% of them cached.

The oracles follow simple rules. They show what well-placed prompts achieve, not the best
possible result.

Per image (mean Dice over its targets and repeats):

| Image | One point | VoxTell | Codex, guided | Claude Code, guided | Codex, unguided | Claude Code, unguided |
|---|---|---|---|---|---|---|
| `adrenal_ct` | 0.82 | 0.00 | 0.91 | 0.91 | 0.90 | 0.90 |
| `ant_microct` | 0.44 | 0.33 | 0.93 | 0.79 | 0.93 | 0.79 |
| `bladder_mr` | 0.81 | 0.86 | 0.76 | 0.77 | 0.71 | 0.76 |
| `calf_mr` | 0.73 | 0.02 | 0.81 | 0.82 | 0.79 | 0.56 |
| `hcc_ct` | 0.37 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |
| `headneck_mr` | 0.43 | 0.76 | 0.65 | 0.64 | 0.67 | 0.62 |
| `headneck_tumor_mr` | 0.74 | 0.88 | 0.79 | 0.79 | 0.80 | 0.78 |
| `liver_vessels_ct` | 0.19 | 0.71 | 0.59 | 0.65 | 0.54 | 0.58 |
| `lung_ct` | 0.92 | 0.95 | 0.91 | 0.91 | 0.92 | 0.90 |
| `lung_pet` | 0.92 | 0.96 | 0.93 | 0.94 | 0.93 | 0.94 |
| `mouse_microct` | 0.91 | 0.00 | 0.91 | 0.89 | 0.83 | 0.82 |
| `pediatric_ct` | 0.84 | 0.92 | 0.90 | 0.88 | 0.89 | 0.87 |
| `thigh_mr` | 0.87 | 0.17 | 0.92 | 0.91 | 0.93 | 0.93 |

`python benchmark/run.py summary` prints all of this for your own runs.

## Get the data

From a clone of this repository:

```bash
pip install -e ".[local,benchmark]"     # local nnInteractive and pydicom
python benchmark/prepare.py fetch       # downloads what can be downloaded (about 1.5 GB)
python benchmark/prepare.py status      # lists what is missing and where to put it
python benchmark/prepare.py build       # writes benchmark/data/<image>/
```

`fetch` gets the TCIA series from TCIA's public API and, from the large zip archives on Zenodo and
ERDA, only the files it needs. The bladder case needs the whole 1 GB RAR archive and something that
reads RAR5: `bsdtar`, `7z` or `unrar` on the PATH, or `pip install libarchive-c`. All of `fetch`
took under 10 minutes in our test; `build` takes about a minute.

The rest needs you (`status` prints these instructions with the full paths):

- **VEELA** (`liver_vessels_ct`): log in at Synapse (free), download `VEELA Train Set.zip`
  (syn65472816) and put it in `benchmark/downloads/_archives/`.
- **Ant brains** (`ant_microct`): download `training.zip` (9.3 GB) from Dryad in a browser and put it
  in `benchmark/downloads/_archives/` (the browser may wrap it in another zip; that works too).
- **Thigh and calf MRI** (`thigh_mr`, `calf_mr`): from the figshare release, copy
  `MRI_data/13/Thigh/{Fat,mask_muscles}.nii.gz` into `benchmark/downloads/thigh_mr/` and
  `MRI_data/05/Calf/{Fat,mask_muscles}.nii.gz` into `benchmark/downloads/calf_mr/`. Note: at the
  moment the figshare record lists no image files. We have asked the authors and will update this
  section when the data are back.

`build` converts whatever is there and skips the rest. It reports any image or mask that differs
from the published checksums. When you report results, say which images you ran.

## Run

```bash
python benchmark/run.py --track point                                   # about a minute
python benchmark/run.py --track oracle-scribbles                        # a few minutes; also oracle-points
python benchmark/run.py --track voxtell --voxtell "conda run -n voxtell voxtell-predict"   # VoxTell in its own env
python benchmark/run.py --track guided --agent claude --model opus --effort medium
python benchmark/run.py --track unguided --agent codex --model <model> --effort medium
python benchmark/run.py --track guided --agent claude --llm-url http://localhost:8000 --model <open model>
python benchmark/run.py summary                                         # all results so far
```

Results go to `benchmark/results/<track>-<agent>-<model>-<effort>/<image>/rep<k>/`: the request,
the agent's transcript, its workspace (tool log, renders, saved masks) and `score.json`. Finished
runs are skipped when you start again, so an interrupted run continues. When the agent hits a usage
limit of its subscription, the run is not scored: `run.py` waits until the reset time the agent
names (or 30 minutes) and tries that image again; `--stop-on-limit` stops instead. `--repeats 3` runs every
image three times; `summary` then reports the spread between repeats.

The agents run headless and work only through their tools (no shell, no file writes); Claude Code
may search the web. Claude Code and Codex take a few minutes per image, so a track takes about an
hour and uses your subscription or API key accordingly. Open models on vLLM can take ten times as
long. `--max-minutes` stops an agent that runs too long.

An agent run sends rendered views of the image (slices, montages, 3D renderings) to the agent's
LLM provider. Check that the terms of each dataset allow this before you run an agent on it.
HaN-Seg, HNTS-MRG, VEELA (CHAOS images) and the thigh and calf data are for non-commercial use, and
some of them allow no derivatives or require share-alike: for these, at least make sure that the
provider does not train on your conversations (the model-training setting of your Claude or ChatGPT
account), and do not send feedback on these runs. The point, oracle and VoxTell tracks, and agents
on an open model that you serve yourself, keep the images on your machine.

nnInteractive runs on your machine with its official weights (`--server local`, the default). A
server with another checkpoint gives other numbers: compare only runs on the same backend.

## License

The benchmark code is part of this repository (Apache-2.0). The data stay under their own terms
(table above); some are non-commercial or allow no derivatives, which is one reason we ship none of
it. nnInteractive's weights are CC BY-NC-SA 4.0, so results with them are for non-commercial use.
