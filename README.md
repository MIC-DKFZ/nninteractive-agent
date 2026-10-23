# nnInteractive Agent

**Say what to segment. The agent does the rest.**

[![arXiv](https://img.shields.io/badge/arXiv-2610.31080-b31b1b.svg)](https://arxiv.org/abs/2610.31080)

<p align="center">
  <img src="assets/nninteractive-agent-hero.gif" width="100%" alt="The web demo on a CT scan of a packed
  suitcase: the request 'Segment the snorkel and the flashlight.', the agent searching the bag, and both
  objects segmented and shown in 3D.">
</p>
<p align="center"><sub>The web demo on a CT scan of a packed suitcase, sped up. Image provided by Smiths
Detection.</sub></p>

[nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) as a tool for AI agents. Tell your agent
in plain text what to segment ("segment every rib individually and save the result"). The agent
prompts nnInteractive with scribbles (boxes where a box fits the object, points for tiny regions),
checks the 3D result and corrects it, as a human annotator would. It works on CT, MRI, microscopy,
electron microscopy and photographs.

What you get:

- **An MCP server** (`nninteractive`) with the tools listed under [Tools](#tools). Any agent that
  speaks MCP can use it: Claude Code, Codex, open models behind Claude Code, and others.
- **A skill** (`nninteractive-segmentation`) that teaches the agent the workflow: prompt rules, a
  correction playbook and how to verify. The MCP server also offers it as an MCP prompt for agents
  that do not load skills.
- **A web demo** (optional): a browser page with 2D/3D views and a chat, to show the agent to
  people who have no agent set up.

**How well it works:** on our [benchmark](benchmark/README.md#results) (14 public CT, MRI, PET and
micro-CT images, 51 named targets, nnInteractive v1.0), Claude Code and Codex reach a mean Dice of
0.81 to 0.85 (averaged per image) in about four minutes per image. nnInteractive alone, with one
click per target, reaches 0.71. When the agents are told where each target is, their prompts are
about as good as simulated prompts placed with the reference mask, at the same number of prompts.
[VoxTell](https://github.com/MIC-DKFZ/VoxTell), a 3D model that segments from text alone, reaches
0.54: it does well on standard radiology targets and fails on the leg muscles, the foot bones and
the animals.

**Technical report:** [*nnInteractive Agent: General-Purpose AI Agents as 3D Image
Annotators*](https://arxiv.org/abs/2610.31080) (arXiv, 2026). It describes the tools and the skill
and evaluates them on the benchmark.

nnInteractive itself runs on your machine (a CUDA GPU is strongly recommended) or on an
`nninteractive-server` on a GPU machine elsewhere.

| You want to | Read |
|---|---|
| segment images from Claude Code or Codex | [Use it from Claude Code or Codex](#use-it-from-claude-code-or-codex) |
| show the agent to other people in a browser | [Web demo](#web-demo) |
| run nnInteractive on another machine than the agent | [nnInteractive: local or on a server](#nninteractive-local-or-on-a-server) |
| measure how well an agent segments | [benchmark/README.md](benchmark/README.md) |

## Installation

Requires Python 3.10 or newer.

```bash
pip install "nninteractive-agent[local] @ git+https://github.com/MIC-DKFZ/nninteractive-agent"
```

| Extra | Adds | Needed for |
|---|---|---|
| `[local]` | nnInteractive with PyTorch and nnU-Net | running nnInteractive on this machine (the default) |
| `[web]` | FastAPI, the Claude Agent SDK | the [web demo](#web-demo) |
| none | the torch-free nnInteractive client | using an [nnInteractive server](#on-a-server) only |

Extras combine: `[local,web]`. Local inference needs an NVIDIA GPU with about 10 GB of memory
(small objects work with less); on a CPU it works, but every prompt takes long. If pip installs a
PyTorch that does not match your CUDA driver, install PyTorch first
([pytorch.org](https://pytorch.org/get-started/locally/)), then this package.

## Use it from Claude Code or Codex

### Set up

Install [Claude Code](https://docs.claude.com/en/docs/claude-code) or
[Codex](https://github.com/openai/codex) and log in. Then:

```bash
nninteractive-agent setup claude        # or: nninteractive-agent setup codex
```

`setup` does three things:

- **nnInteractive:** for local inference it checks PyTorch and your GPU and downloads the
  nnInteractive weights (about 400 MB, into `~/.nninteractive`). With `--server` it connects to
  the server once instead.
- **The MCP server:** it registers the server for your user, with the Python you installed into,
  so it works without activating that environment.
- **The skill:** Claude Code gets it in `~/.claude/skills`, plus a verifier subagent in
  `~/.claude/agents`. Codex gets it in `~/.codex/skills`.

From then on, every new Claude Code or Codex session has the tools and the skill, in any directory.
A session that never segments anything only starts a small Python process: PyTorch and the model
load with the first image.

| `setup` option | |
|---|---|
| `--server URL` | use an `nninteractive-server` instead of local inference (see [On a server](#on-a-server)) |
| `--python full\|guarded\|off` | the agent's `python` tool (see [Security](#security)); default `full` |
| `--workdir DIR` | where renders, the tool log and default outputs go (default: `./runs/<time>` in the directory the agent runs in) |
| `--skip-model` | do not check PyTorch or download the weights now |
| `--uninstall` | remove the MCP server, the skill and the verifier agent again |

The choices are stored in the agent's MCP configuration. To change them, for example to switch from
local inference to a server, run `setup` again with the new options.

### Use

Start the agent in the directory where results should go, and ask in plain text:

```bash
claude "Segment the liver and the spleen in /data/ct.nii.gz, each as its own object, and save the result."
codex "Segment every nucleus in /data/cells.tif and save the result."
```

Or start `claude` or `codex` without a prompt and ask there. The agent loads the skill on its own
when you ask it to segment, delineate, outline, label or count structures in an image file. The
image and its objects stay loaded for the whole session, so you can follow up: "the liver leaks
into the heart, fix that", "now also both kidneys, each as its own object".

- **Inputs:** NIfTI, NRRD, MHA, TIFF (stacks), PNG/JPG and NPY.
- **Results:** an instance map with one id per object (`<image>_instances` in the input's
  format) and `objects.json` with names and notes. They go wherever you tell the agent, or to
  `./runs/<time>/output/`.
- **Good requests** say what counts as one object ("each rib", "each nucleus") and name the
  structures the way a radiologist or biologist would.
- **Tool approvals:**
  - Claude Code asks before each nnInteractive tool it has not been allowed to use yet. Answer
    "don't ask again", or add `"mcp__nninteractive"` to `permissions.allow` in
    `~/.claude/settings.json` to allow all of them.
  - `codex exec` (non-interactive) needs the tools approved up front:
    `codex exec -c mcp_servers.nninteractive.default_tools_approval_mode="approve" "..."`.
  - Before you allow the `python` tool for good, read [Security](#security).

Tested with Claude Code 2.1 (Opus 5.5), Codex 0.162.1 (gpt-6.1-sol) and Claude Code on vLLM
(Qwen3.8-27B, see [Open models](#open-models)).

### Other agents

```bash
nninteractive-agent setup print
```

This prints an `mcpServers` entry for any MCP client (Cursor, opencode, Goose, ...) and the path of
the skill folder. Copy the folder to wherever your agent looks for skills. If the agent has no
skills, it can read the MCP prompt `nninteractive-segmentation` (also available as resources
under `nninteractive://skill/`); the server's instructions point the agent to it.

The agent needs a model that sees images: the tools return rendered views.

Claude Code can also load the tools and the skill as a plugin, for one session only, from an
environment where `nninteractive-agent-mcp` is on the PATH:

```bash
claude --plugin-dir "$(python -c 'import nninteractive_agent.skill as s; print(s.PLUGIN_DIR)')"
```

### Open models

Claude Code can drive any model behind an Anthropic-compatible API, for example a
[vLLM](https://docs.vllm.ai) server (it serves `/v1/messages`). The model must:

- see images, including images inside tool results;
- call tools. In vLLM this needs `--enable-auto-tool-choice` and the model's `--tool-call-parser`;
  otherwise tool calls come back as plain text.

We tested Qwen3.8-27B:

```bash
nninteractive-agent setup claude
export ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=dummy ENABLE_TOOL_SEARCH=false
export ANTHROPIC_DEFAULT_OPUS_MODEL=Qwen/Qwen3.8-27B ANTHROPIC_DEFAULT_SONNET_MODEL=Qwen/Qwen3.8-27B \
       ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen/Qwen3.8-27B
claude --model Qwen/Qwen3.8-27B --effort medium
```

Pick an effort the server accepts: vLLM with Qwen takes low, medium and xhigh, not high. Open
models work, but they are slower and less precise. On a spleen CT (`spleen_10` of the Medical
Segmentation Decathlon, nnInteractive v1.0 on the same machine), Qwen3.8-27B at medium effort
reached Dice 0.93 with five prompts, but took 34 minutes; Claude Code and Codex need about four
minutes per image in the [benchmark](benchmark/README.md#results), for 3.6 targets on average.
Its first boxes are often too large, so do not tell it to skip verification.

## Web demo

A demonstrator, separate from the tools above: the same tools and skill, driven by an agent behind
a browser page. The people using it need no agent of their own, and the web demo does not need
`setup`.

```bash
pip install "nninteractive-agent[local,web] @ git+https://github.com/MIC-DKFZ/nninteractive-agent"
nninteractive-agent-web --examples ~/images
```

It prints a link, `http://127.0.0.1:8443/?key=...`: open it in a browser on the same machine.
The start-up lines also show where nnInteractive runs, its model license, the agent's LLM and the
`python` mode. If the demo runs on a remote GPU machine, forward the port and open the link on
your own computer:

```bash
ssh -N -L 8443:127.0.0.1:8443 you@gpu-machine
```

- **The agent's LLM** (`--llm`, checked at start-up). `--model` and `--effort` set the defaults,
  and the page lets users change them:

  | `--llm` | Agent | Signs in with |
  |---|---|---|
  | `claude` (default) | Claude Code | the Claude subscription of the account running the demo (log in once with `claude`) |
  | `claude-api` | Claude Code | an Anthropic API key: `$ANTHROPIC_API_KEY` or `--api-key-file` |
  | `codex` | Codex | the ChatGPT subscription of that account (`codex login`) |
  | `codex-api` | Codex | an OpenAI API key: `$OPENAI_API_KEY` or `--api-key-file` |
  | `server` | Claude Code | your own model on an Anthropic-compatible server such as vLLM, on this machine (default `http://localhost:8000`) or another (`--llm-url http://host:port`); `--model Qwen/Qwen3.8-27B` picks one of its models (a key, if needed, in `$LLM_API_KEY` or `--api-key-file`) |

  Codex needs the `codex` CLI. From Codex 0.162 on, it starts verifier subagents as Claude Code
  does.
- **Subscription or API key:** `claude` and `codex` run on the personal Claude or ChatGPT
  subscription of the account that runs the demo. Anthropic's and OpenAI's terms allow that only
  for your own use (as of October 2026): do not let other people use a demo that runs on your
  subscription. To host the demo for others, use an open model (`server`) or an API key
  (`codex-api`, `claude-api`). An Anthropic API key may serve your own organization's users;
  paying for the Claude usage of anyone else needs an agreement with Anthropic. The demo warns
  at start-up when it serves a network on a subscription. Terms change, so check the current
  ones: [Anthropic](https://code.claude.com/docs/en/legal-and-compliance),
  [OpenAI](https://learn.chatgpt.com/docs/app-server) (section "Auth endpoints").
- **nnInteractive:** local by default, which needs the `[local]` extra. Each browser session then
  loads its own model, so only one browser can work at a time (`--max-sessions` 1). With
  `--server URL` (or `$NNINTERACTIVE_SERVER_URL`) the demo uses an
  [nnInteractive server](#on-a-server) instead: `[web]` alone is enough, and up to 10 browsers can
  work at the same time (`--max-sessions`; the server's own limit applies too).
- **Examples:** `--examples DIR` puts every image in `DIR` on the start page, with a thumbnail.
  An optional `DIR/examples.json` adds titles, a subtitle, a modality badge and suggested
  requests per file:

  ```json
  {"cases": [{"path": "spleen_10.nii.gz", "title": "Abdominal CT", "subtitle": "MSD spleen, case 10",
              "modality": "CT", "tasks": ["Segment the spleen.", "Segment both kidneys, each as its own object."]}]}
  ```

  `--examples list.json` takes such a file directly; paths in it are relative to the file and
  may contain `$VARIABLES`. Users can always upload their own image (up to 2 GB).
- **Sharing it on a network:** `--host 0.0.0.0` serves the demo to the network over HTTPS with a
  self-signed certificate (`--certfile`/`--keyfile` for a real one; `--insecure-http` only behind
  a proxy that terminates TLS) and a password. The password is in
  `~/.config/nninteractive_agent/web_password`, or set `NNINTERACTIVE_WEB_PASSWORD`. Every
  logged-in browser gets its own image, workspace and conversation. A session idle for
  `--idle-minutes` (default 30) pauses, frees its nnInteractive slot and continues with the next
  message. Read [Security](#security) before you open the demo to a network, and choose the
  agent's LLM accordingly (see "Subscription or API key" above).
- **The page:**
  - **Left:** axial, coronal and sagittal slices, and a 3D view with every object as a surface
    (CT presets: Bones, Bones + skin, Skin).
  - **Right:** the object list and the conversation. Each of the agent's steps shows thumbnails
    of the images it looked at.
  - **While the agent works:** its open object, its prompts and the surfaces update live.
  - **Pointing:** "Mention location" puts a clicked voxel into your message.
  - **Help:** "Controls" in the viewer lists the mouse and keyboard controls.
- **Browser:** the page needs WebGL2: any current Chrome, Edge, Firefox or Safari with hardware
  acceleration on. It does not work through remote desktop or in a VM without a GPU.

## nnInteractive: local or on a server

The agent tools and the web demo both run nnInteractive either in their own process or on an
`nninteractive-server`:

| | Local (default) | Server |
|---|---|---|
| Where nnInteractive runs | inside the MCP server or web demo process | in `nninteractive-server` on a GPU machine |
| Install | the `[local]` extra | no extra: the client is torch-free |
| Choose it with | nothing, or `--server local` | `--server URL` or `$NNINTERACTIVE_SERVER_URL` |
| GPU memory | one model per agent session | one model, shared by all sessions |
| Good for | one user on a GPU machine | several users, or agents on machines without a GPU |

`--server` works the same for `nninteractive-agent setup` and `nninteractive-agent-web`. Both read
`$NNINTERACTIVE_SERVER_URL` when `--server` is not given; for the agent tools the choice is made
at `setup`.

The license of the nnInteractive weights in use is shown when `setup` downloads them or connects to
the server, in the result of `load_image` (the agent passes it on), and on the web demo's start-up
lines and header. The official weights are CC BY-NC-SA 4.0.

### Local

- The weights are downloaded on first use (`setup` does it ahead of time) into
  `$NNINTERACTIVE_MODEL_DIR` (default `~/.nninteractive`).
- `--server local:<model id>` picks another official model; `nninteractive-available-models`
  lists them.
- Every agent session that loads an image holds its own model on the GPU: two sessions at the same
  time need the memory for two.

### On a server

`nninteractive-server` is part of [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive).
It loads the model once and serves several clients. On the GPU machine:

```bash
pip install nnInteractive
nninteractive-server --host 0.0.0.0 --port 1527 --max-sessions 4 --idle-timeout-seconds 1800 \
    --api-key "$(openssl rand -hex 32)"
```

A Docker image is available as well (`ghcr.io/mic-dkfz/nninteractive-server`).
nnInteractive's [`SERVER_CLIENT.md`](https://github.com/MIC-DKFZ/nnInteractive/blob/master/SERVER_CLIENT.md)
describes both, plus HTTPS behind a reverse proxy.

- **`--max-sessions`:** how many agents (or web demo browsers) can work at the same time. When
  the server is full, a new agent waits up to 2 minutes for a free slot, then reports an error.
- **`--idle-timeout-seconds`:** an agent often thinks for minutes between two tool calls, and the
  server's default of 10 minutes can expire its session in between. The tools then open a new
  session by themselves and restore the image and the object being worked on; only undo of earlier
  prompts is lost. A timeout of 30 minutes avoids most of these restarts.
- **The server speaks plain HTTP.** Outside a trusted network, put it behind HTTPS, or keep it on
  `127.0.0.1` (its default) and use an SSH tunnel:
  `ssh -N -L 1527:127.0.0.1:1527 you@gpu-machine`, then `--server http://127.0.0.1:1527`.

Then point the agent or the web demo to it:

```bash
nninteractive-agent setup claude --server http://gpu-machine:1527     # or: setup codex --server ...
nninteractive-agent-web --server http://gpu-machine:1527 --examples ~/images
```

If the server was started with `--api-key`, the client sends the key from
`$NN_INTERACTIVE_API_KEY`. Where that variable has to be set:

- **`setup` and the web demo** read it from your shell: `export NN_INTERACTIVE_API_KEY=...` first.
- **Claude Code** passes its environment on to the MCP server: export the variable before you
  start `claude` (for example in your shell profile).
- **Codex** starts MCP servers with a minimal environment. Add the key to `~/.codex/config.toml`,
  and again after every `setup codex`:

  ```toml
  [mcp_servers.nninteractive.env]
  NN_INTERACTIVE_API_KEY = "..."
  ```

## Security

- **Images leave your machine.** The agent's LLM sees rendered views of your image. Use only
  images you may share with the provider of that LLM (or run an open model yourself).
- **The `python` tool.** The agent can run Python inside the MCP server process (to find
  candidate structures, to loop over many instances). Choose how much it may do:

  | `--python` | |
  |---|---|
  | `full` (default) | unrestricted: the code runs as you and can read and write files, start programs and use the network |
  | `guarded` | computations only. Imports come from an allowlist (numpy, scipy, scikit-image, matplotlib, math, collections, ...). Blocked: file access (`open`, `savefig`, `np.save`, `imread`, ...), processes, `getattr`/`eval`/`exec`, and attributes starting with `_`. Each call is checked before it runs. |
  | `off` | no `python` tool. The agent still works, but many-instance tasks (all nuclei, all ribs) get slower and less thorough. |

  `guarded` stops accidents and simple prompt injection. **It is not a sandbox**: Python cannot
  be isolated inside the process that runs it. It does not limit memory or run time either. Real
  isolation needs a container, a VM or a separate user account.
- **Other tools.** `load_image` reads any image path the agent passes, and `save` writes where it
  is told. Codex and Claude Code ask before tool calls unless you allow them; reading the code of
  `python` calls before approving them is the simplest safeguard.
- **Web demo.** Anyone who can log in drives the agent: with `--python full` that means running
  code on your machine as you. The demo serves only `127.0.0.1` by default. Before you open it to
  a network (`--host 0.0.0.0`), consider `--python guarded` or `off`, and give the password only
  to people you trust. In the demo the agent cannot choose file paths: the page picks the image,
  and results stay in the session's folder.

## Tools

| Tool | Purpose |
|---|---|
| `load_image` | Load NIfTI/NRRD/MHA, TIFF, PNG/JPG or NPY; returns statistics, orientation and overview images |
| `new_object`, `commit`, `discard`, `edit_object`, `delete_object`, `list_objects` | One nnInteractive object per instance; `new_object(extent=...)` segments an object larger than about 320 voxels on a downsampled image |
| `scribble` | Polyline in one slice, rasterized 2 px wide; positive or negative. The preferred first prompt (previewed) and the prompt for corrections |
| `bbox` | Positive 2D box, only as an object's first prompt and only where a box encloses the object without much else; previewed before it is sent |
| `point` | Positive or negative point, only for tiny objects or tiny errors; optional preview |
| `undo` | Revert the last prompt |
| `stats` | Size, components, intensity, overlaps, slice-area warnings |
| `view`, `montage`, `projection` | Off-screen renders with absolute voxel ticks and overlays |
| `render3d` | Depth-shaded 3D views of one object's surface from several directions, neighbours faint; same axes and ticks as the slice views |
| `python` | Persistent Python namespace with the workspace, for candidate detection and loops (see [Security](#security)) |
| `save` | Instance map + `objects.json` |

**Large objects:** nnInteractive predicts 192³ patches. It reaches larger objects by zooming out,
which works less well than when the object fits. So an object larger than about 320 voxels along
any axis is segmented on a copy of the image downsampled to about 256 voxels along that axis, and
its mask is scaled back up. Thin structures stay at full resolution. The agent gives its size
estimate to `new_object`, and coordinates stay at full resolution.

## Layout

- `src/nninteractive_agent/`: the tool
  - `core/`: image I/O, the nnInteractive session, prompt rasterization, metrics, rendering,
    downsampling (`scale.py`), the `python` guard (`pyguard.py`) and `AgentWorkspace`
  - `tools.py`: tool names, schemas, descriptions and handlers
  - `mcp_server.py`: stdio MCP server (`nninteractive-agent-mcp`)
  - `cli.py`: `nninteractive-agent setup`
  - `claude_plugin/`: the skill, the verifier agent and the Claude Code plugin manifest
- `src/nninteractive_agent/web/`: the web demo. This covers the FastAPI server, sessions, and the
  in-process agent runners for Claude (`agent_sdk.py`) and Codex (`agent_codex.py`, through
  `codex app-server`). The page vendors NiiVue for the slices and has its own WebGL2 3D view.
- `benchmark/`: a benchmark on public images, built from the original sources (`prepare.py`), and
  its runner (`run.py`); see [benchmark/README.md](benchmark/README.md)
- `tests/`: unit tests (`pytest`, no GPU or LLM needed)
- `assets/`: the animation at the top of this README

## Development

- `pytest` runs the unit tests.
- `python tests/web_dev.py [--examples DIR]` serves the web page on `http://127.0.0.1:8780/`
  (password `dev`). It uses a fake nnInteractive session and a scripted agent, for UI work without
  an LLM or a GPU.
- `python tests/browser_checks.py {controls,window-under-load,rotate,no-webgl} --gpu {soft,igpu,rtx}`
  checks the page in headless Chromium against that dev server (needs `playwright`).
- Scripts that start an agent from inside a Claude Code session must strip its variables
  (`CLAUDECODE`, `CLAUDE_CODE_*`) first, as `nninteractive-agent-web` does.
- Anything that creates an `AgentWorkspace` should call `ws.close()` when done, to free its
  nnInteractive session.

## License and citation

The code is licensed under Apache-2.0 (see `LICENSE`). The nnInteractive model weights that local
inference downloads are licensed under **CC BY-NC-SA 4.0** (non-commercial). The vendored web
libraries keep their own licenses (`src/nninteractive_agent/web/static/vendor/`).

If you use this, please cite the report and nnInteractive:

> Isensee, F., Ulrich Harsy, C., Rokuss, M., Hamm, B., Vollmuth, P., & Maier-Hein, K. H. (2026).
> nnInteractive Agent: General-Purpose AI Agents as 3D Image Annotators. https://arxiv.org/abs/2610.31080

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B.,
> Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025).
> nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373
