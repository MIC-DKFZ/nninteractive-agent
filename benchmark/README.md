# Benchmark

How well does an agent segment a structure named in plain text, with nnInteractive as its tool?
14 public images (CT, MRI, PET and micro-CT; people, a mouse and an ant), 51 targets, none of them
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
`--voxtell`. Its weights are CC BY-NC-SA 4.0. VoxTell keeps the output of every prompt on the GPU;
for `foot_ct` (17 prompts on a whole-body CT) that does not fit on a 24 GB GPU, so we ran it with
`--voxtell-group 3` (three prompts per call). VoxTell decodes each prompt on its own, so splitting
changes only the numerics: its masks on `pediatric_ct` (8 prompts) changed by less than 0.002 Dice.

**Score:** Dice per target, with the target's object as it was when the agent last committed it,
so a later object that takes voxels from an earlier one (a tumour inside the liver) does not count
against the earlier one. A target without an object scores 0. Each target's Dice is averaged over
the repeats, then over the targets of its image, then over the images: every image weighs the same,
whether it has one target or 17 (the targets of one image are not independent). `summary` also
reports the plain mean over targets, the tiers (E easy, M medium, H hard; each averaged the same
way), each image, and for the oracles the Dice after every prompt.

## Images

| Image | Targets | Source | Access |
|---|---|---|---|
| `pediatric_ct` | liver, spleen, right kidney, bladder, heart, pancreas, gallbladder, left adrenal | [Pediatric-CT-SEG](https://doi.org/10.7937/TCIA.X0H0-1706) (TCIA), CC BY 4.0 | download |
| `hcc_ct` | liver, liver tumour | [HCC-TACE-Seg](https://doi.org/10.7937/TCIA.5FNA-0924) (TCIA), CC BY 4.0 | download |
| `adrenal_ct` | adrenal tumour | [Adrenal-ACC-Ki67-Seg](https://doi.org/10.7937/1FPG-VM46) (TCIA), CC BY 4.0 | download |
| `lung_ct` | lung tumour | [RIDER Lung CT](https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR) (TCIA), CC BY 4.0, with the [RIDER-LungCT-Seg](https://doi.org/10.7937/tcia.2020.jit9grk8) tumour mask, CC BY 3.0 | download |
| `lung_pet` | FDG-avid tumour | [ACRIN-NSCLC-FDG-PET](https://doi.org/10.7937/tcia.2019.30ilqfcl) (TCIA), CC BY 3.0, with the [BAMF/AIMI](https://doi.org/10.5281/zenodo.13244892) lesion mask, CC BY 4.0 | download |
| `headneck_mr` | mandible, brainstem, left parotid, right submandibular gland, thyroid, spinal cord, left optic nerve | [HaN-Seg](https://doi.org/10.5281/zenodo.7442914), CC BY-NC-ND 4.0 | download |
| `headneck_tumor_mr` | primary tumour | [HNTS-MRG 2024](https://doi.org/10.5281/zenodo.11199559), CC BY-NC 4.0 | download |
| `bladder_mr` | bladder tumour | [FedBCa](https://doi.org/10.5281/zenodo.13622759), CC BY 4.0 | download (1 GB RAR) |
| `thigh_mr` | rectus femoris, sartorius, gracilis, vastus lateralis | [Thigh and calf muscle MRI](https://doi.org/10.6084/m9.figshare.31042489.v1) (former figshare release: CC BY 4.0, non-commercial use only) | on request from the authors |
| `calf_mr` | soleus, tibialis anterior, medial gastrocnemius | same | on request from the authors |
| `foot_ct` | the 17 bones of the left foot: talus, calcaneus, navicular, cuboid, three cuneiforms, five metatarsals, the phalanges of each toe | [Visible Human Full-Skeleton 3D Bone Models](https://doi.org/10.57967/hf/10464) (Hugging Face), CC BY 4.0, on the Visible Human Female CT as aligned by the [University of Denver](https://doi.org/10.56902/COB.vh.2022.0), CC BY 4.0 | download |
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

`foot_ct` is a whole-body CT; the targets are the bones of the left foot only. Its intensities are
not Hounsfield units (air 0, soft tissue about 1000); the request says so. Its references are the
source's segments as they are, so neighbouring bones overlap by a few voxels. The source's
annotators made them with nnInteractive in 3D Slicer and checked them, which may favour the tracks
that use nnInteractive. The Visible Human images are courtesy of the U.S. National Library of
Medicine.

## Results

Our runs on all 14 images (October 2026):
- **nnInteractive:** the official weights (v1.0), on this machine, one RTX 4090.
- **Agents:** Claude Code 2.1 with Opus 5.5 and Codex 0.162.1 with gpt-6.1-sol, both at medium
  effort, three repeats per track (± is the spread of the track's mean over repeats). No run came
  near the 45-minute limit per image (the longest, Claude Code on `foot_ct`, took 26 minutes) and
  no target was left without an object.
- **No-agent tracks:** the point track ran three times, the oracles and VoxTell v1.1 once. Local
  GPU inference is not exactly repeatable; per-image results vary by about 0.01 between runs.

| Track | Mean Dice | Easy (11) | Medium (29) | Hard (11) | Prompts per target | Oracle, same prompts: scribbles / points | Minutes per image | Mean over targets |
|---|---|---|---|---|---|---|---|---|
| Codex, guided | 0.855 ± 0.002 | 0.927 | 0.841 | 0.659 | 2.9 | 0.869 / 0.827 | 3.8 | 0.870 |
| Claude Code, guided | 0.847 ± 0.006 | 0.926 | 0.831 | 0.663 | 2.2 | 0.844 / 0.789 | 3.6 | 0.866 |
| Codex, unguided | 0.842 ± 0.004 | 0.931 | 0.826 | 0.660 | 2.8 | 0.886 / 0.812 | 4.2 | 0.867 |
| Claude Code, unguided | 0.814 ± 0.006 | 0.923 | 0.802 | 0.592 | 2.3 | 0.854 / 0.806 | 4.3 | 0.841 |
| One point, no agent | 0.709 ± 0.000 | 0.541 | 0.706 | 0.559 | 1 | 0.769 / 0.710 | 0.2 | 0.751 |
| VoxTell, text only | 0.539 | 0.791 | 0.459 | 0.623 | 0 | | 1.5 | 0.431 |

"Mean Dice" is the mean over the 14 images of each image's mean over its targets; the tiers are
averaged the same way over the images that have targets of that tier (in brackets: the number of
targets); prompts per target are averaged per image as well. "Mean over targets" weighs all 51 targets equally, which gives the 17 bones of `foot_ct`
a third of the weight. "Oracle, same prompts" is the oracles' Dice after as many prompts on each
target as the agent sent for it; where the agent sent more than five (4% to 11% of the targets),
after five. VoxTell's minutes are 0.3 per image except `foot_ct` (17 minutes in six calls).

What we read from it:
- **An agent makes nnInteractive far better than one click on the same targets**, even when the
  agent has to find the targets itself: 0.81 to 0.85 against 0.71. The gain is largest where one
  point does not describe the target: two of the three livers (one point 0.00 and 0.61, the agents
  0.95 to 0.97), the liver tumour (0.13 against 0.96) and the ant brain.
- **At the same number of prompts, the agents' prompts are close to the oracles'.** Claude Code
  (guided) is level with the scribble oracle (0.847 against 0.844) and Codex (guided) 0.015 behind
  it; the point oracle is behind all four agent tracks. Unguided, the scribble oracle is 0.04 ahead,
  which includes the targets the agents did not find exactly.
- **The agents stop early.** They send 2.2 to 2.9 prompts per target, while the oracles keep
  improving with every further prompt.
- **Hard targets stay hard:** the portal and hepatic veins (agents 0.27 to 0.53), the optic nerve
  (0.02 to 0.34) and the spinal cord (0.46 to 0.55): thin structures that are hard to prompt and to
  judge from rendered views.
- **The foot bones are easy for every track that uses nnInteractive** (one point 0.95, agents and
  oracles 0.98): the agents found and named all 17 bones of the left foot in the whole-body CT
  without the hint voxel.
- **VoxTell, from the text alone, is more accurate within standard radiology, but brittle.** On
  seven of the nine head-and-neck, chest, abdominal and pelvic images it is ahead of every agent (by
  0.02 to 0.09), also on thin structures the agents find hard (optic nerve 0.59, spinal cord 0.61,
  hepatic veins 0.64). For the adrenal tumour it returns an empty mask, so its mean over the nine
  images is 0.78 (agents 0.81 to 0.83; without that image 0.88 against 0.80 to 0.82). That failure
  is a matter of wording: "the tumor of the right adrenal gland" gives the empty mask, "adrenal
  tumor" Dice 0.79 (a check outside the benchmark). Several source datasets of the radiology images
  are in VoxTell's training data (HaN-Seg, HNTS-MRG, RIDER Lung CT, HCC-TACE-Seg), so those numbers
  may be optimistic. On the leg muscles it reaches 0.10 (agents 0.74 to 0.87), on the bones of the
  foot 0.03 (agents 0.98; it mostly segments a neighbouring bone), on the mouse and the ant 0.17
  (agents 0.80 to 0.92). Its precomputed prompts name all of these muscles and each tarsal bone (the
  metatarsals and toe bones only as groups); the images,
  however, are far from standard radiology (fat-weighted research MRI, a whole-body CT of a cadaver,
  animals). An agent with nnInteractive works on any of these, at the price of minutes per image.
- **Codex and Claude Code** are about equal with the guided voxel. Unguided, Codex is ahead (0.842
  against 0.814), mostly on the calf muscles and the ant brain.
- **Cost:** Claude Code reported $1.29 (guided) and $1.46 (unguided) per image at API prices ($4
  to $5 for the 17 foot bones), with about 3 million cached input tokens per image. Codex used
  about 1.8 million input tokens per image, 96% of them cached.

The oracles follow simple rules. They show what well-placed prompts achieve, not the best
possible result.

Per image (mean Dice over its targets and repeats):

| Image | One point | VoxTell | Codex, guided | Claude Code, guided | Codex, unguided | Claude Code, unguided |
|---|---|---|---|---|---|---|
| `adrenal_ct` | 0.82 | 0.00 | 0.91 | 0.91 | 0.90 | 0.90 |
| `ant_microct` | 0.44 | 0.33 | 0.93 | 0.79 | 0.93 | 0.79 |
| `bladder_mr` | 0.81 | 0.86 | 0.76 | 0.77 | 0.71 | 0.76 |
| `calf_mr` | 0.73 | 0.02 | 0.81 | 0.82 | 0.79 | 0.56 |
| `foot_ct` | 0.95 | 0.03 | 0.98 | 0.98 | 0.98 | 0.98 |
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
python benchmark/prepare.py fetch       # downloads what can be downloaded (about 2 GB)
python benchmark/prepare.py status      # lists what is missing and where to put it
python benchmark/prepare.py build       # writes benchmark/data/<image>/
```

`fetch` gets the TCIA series from TCIA's public API, two files from Hugging Face and, from the large
zip archives on Zenodo and ERDA, only the files it needs. The bladder case needs the whole 1 GB RAR archive and something that
reads RAR5: `bsdtar`, `7z` or `unrar` on the PATH, or `pip install libarchive-c`. All of `fetch`
took under 10 minutes in our test; `build` takes about a minute.

The rest needs you (`status` prints these instructions with the full paths):

- **VEELA** (`liver_vessels_ct`): log in at Synapse (free), download `VEELA Train Set.zip`
  (syn65472816) and put it in `benchmark/downloads/_archives/`.
- **Ant brains** (`ant_microct`): download `training.zip` (9.3 GB) from Dryad in a browser and put it
  in `benchmark/downloads/_archives/` (the browser may wrap it in another zip; that works too).
- **Thigh and calf MRI** (`thigh_mr`, `calf_mr`): the data are no longer in the figshare record;
  request them from the authors (the contact is in https://github.com/VinceWohl/DL-Segmentation-of-Leg-Muscles). Then copy
  `MRI_data/13/Thigh/{Fat,mask_muscles}.nii.gz` into `benchmark/downloads/thigh_mr/` and
  `MRI_data/05/Calf/{Fat,mask_muscles}.nii.gz` into `benchmark/downloads/calf_mr/`.

`build` converts whatever is there and skips the rest. It reports any image or mask that differs
from the published checksums. When you report results, say which images you ran.

## Run

```bash
python benchmark/run.py --track point                                   # about a minute
python benchmark/run.py --track oracle-scribbles                        # a few minutes; also oracle-points
python benchmark/run.py --track voxtell --voxtell "conda run -n voxtell voxtell-predict"   # VoxTell in its own env (--voxtell-group 3 on a 24 GB GPU)
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

The agents run headless and work through their tools, including a Python tool that runs with your user
rights (`--python guarded` limits it to computations). Codex reads the skill through its shell, which `run.py`
sets to Codex's `workspace-write` sandbox (the run folder, no network); where that sandbox cannot start, Codex
asks to run the command outside it and an automatic reviewer decides. Claude Code may also search the web. Claude Code and Codex take a few minutes per image, so a track takes about an
hour and uses your subscription or API key accordingly. Open models on vLLM can take ten times as
long. `--max-minutes` stops an agent that runs too long. For long unattended runs, `--max-retries N` gives
up on an image after N failed retries and goes on (the image stays unscored), and `--pause-file F` holds every
run while the file F exists. `--no-image-info` leaves the description of the image out of the request ("This is
an image: <path>") and gives the agent neutral folder and file names, to test how much the agents rely on it.

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
