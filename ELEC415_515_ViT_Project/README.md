# ELEC 415/515 Course Project: Accelerating a Vision Transformer in Hardware

> This README has the same content as `project_description.pdf` in this folder.

## Overview

In this project you get a complete, working hardware accelerator for a small Vision
Transformer (ViT) that recognizes handwritten MNIST digits. Your job is to make it faster
using two model compression techniques from the lectures: **attention-head pruning** and
**INT8 quantization**. Along the way you'll see, cycle by cycle, how a change to the model
turns into a speed-up in hardware, and what it costs in accuracy.

Here's what we give you:

- **A trained model.** A ViT with 6 encoder blocks, 4 attention heads per block and an
  embedding size of 64. It reaches 99.46 % test accuracy on MNIST.
- **The accelerator, written in SystemVerilog.** It runs the encoder blocks and the
  classifier in fixed point, and it already has two hooks for compression:
  - a *head mask* input that tells the hardware which attention heads to skip, and
  - a compile-time switch that turns the linear layers into an *INT8 datapath*.
- **A bit-exact Python model of the hardware** (we call it the *golden model*). The
  testbench checks that, for every test image, the hardware produces exactly the same ten
  output values as this model. So one simulation tells you two things: *is the hardware
  correct*, and *how many clock cycles does it need*.
- **Scripts** that export a model to the hardware, measure its accuracy on all 10,000 test
  images, and run the simulation.

Here's what you'll do, and how it is graded:

| Part | Task | Points |
|---|---|---|
| 0 | Set up the tools and reproduce the baseline results | 10 % |
| 1 | Read the RTL and draw the hardware block diagram | 30 % |
| 2 | Prune attention heads and measure accuracy and speed | 30 % |
| 3 | Quantize the linear layers to INT8 and measure accuracy and speed | 30 % |
| 4 | Make the accelerator as fast as you can | Bonus |

How it works:

- **This is a group project.** Each group hands in **one final report**, covering Parts 0–3
  (and Part 4 if you do it). Each part below ends with a short list of what to include.
- **Submit the report on your group's homepage on Canvas.** It is due by the end of the
  project presentation week, i.e. the last week of the semester (tentative; we'll confirm
  the exact date on Canvas).
- **Write down who did what.** The report must state the contributions of each group
  member.
- **Everyone presents.** Each group presents its project in class during the presentation
  week (this is also where Part 4 is presented), and every member of the group takes part.

Good news for Parts 2 and 3: **you don't need to touch the hardware.** It already supports
both techniques; you control it by producing a file in Python (a head mask, or a set of
quantization scales). In every part, keep an eye on two numbers: **accuracy** and **clock
cycles**. A compression trick is only worth it if the hardware gets faster without losing
too much accuracy.

---

## Part 0: Setup and baseline reproduction (10 %)

The goal of this part is simple: get the tools running and reproduce our numbers. Once
you see the same results we do, you know your setup is good.

### 0.1 What you need

Everything runs on **Linux**, **macOS** (Intel or Apple Silicon) and **Windows 10/11
through WSL2** (a Linux system that runs inside Windows). Plan for about 6 GB of free disk
space and 8 GB of RAM or more. You'll need an internet connection for the installation, and
again the first time you run a script that trains or calibrates (it downloads the MNIST
training set automatically). No GPU needed.

The tools:

| Tool | What it's for |
|---|---|
| Python 3 with `numpy`, `torch`, `torchvision` | the model, pruning, quantization, the golden model |
| Verilator 5 | the open-source SystemVerilog simulator |
| `make` and a C++ compiler | Verilator uses them to build the simulator |

The steps below install all of them into one conda environment, the same way on every
system. Already comfortable with Python environments and prefer your own setup? Jump to
0.5 for the exact requirements.

### 0.2 Windows only: install WSL2

On Windows, the project runs inside WSL2 with Ubuntu. You only do this once:

1. Open PowerShell as Administrator: right-click the Start button and choose *Terminal
   (Admin)* on Windows 11, or *Windows PowerShell (Admin)* on Windows 10. Then run:

   ```text
   wsl --install
   ```

2. Restart the computer when asked. After the restart an *Ubuntu* window opens; pick a
   Linux user name and password (they're separate from your Windows login).
3. From now on, whenever these instructions say "terminal", open **Ubuntu** from the Start
   menu and type the commands there. First, install two small tools:

   ```text
   sudo apt update
   sudo apt install -y wget unzip
   ```

4. Download the project zip in Windows as usual, then copy it into your Linux home folder
   and unzip it there (replace `<WindowsUser>` with your Windows user name, and the zip name
   with the name of your download if it differs):

   ```text
   cp /mnt/c/Users/<WindowsUser>/Downloads/ELEC415_515_ViT_Project.zip ~/
   cd ~
   unzip ELEC415_515_ViT_Project.zip
   ```

   Keep the project in your Linux home folder (`~/`), not under `/mnt/c/`. Building the
   simulator from a Windows folder is very slow.

That's it for Windows; now follow the **Linux** instructions below.

### 0.3 macOS only: install the Command Line Tools

Open the *Terminal* app and run:

```text
xcode-select --install
```

Click *Install* in the window that pops up. (If it says the tools are already installed,
you're all set.)

### 0.4 Install Miniconda and create the course environment

If `conda --version` already prints a version number in your terminal, skip step 1
(Anaconda, Miniconda and Miniforge all work).

1. **Install Miniconda.** Download the installer for your system and run it.

   Linux or WSL2 (most PCs):

   ```text
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

   macOS on Apple Silicon (M1/M2/M3/M4):

   ```text
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
   bash Miniconda3-latest-MacOSX-arm64.sh
   ```

   macOS on Intel:

   ```text
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   bash Miniconda3-latest-MacOSX-x86_64.sh
   ```

   Press Enter to read the license, type `yes` to accept it, press Enter to keep the
   default location, and answer `yes` when it asks whether to initialize conda. Then
   **close the terminal, open a new one**, and check:

   ```text
   conda --version
   ```

2. **Go to the project folder.** On macOS and Linux, unzip the project zip first if you
   haven't yet (double-click it in Finder, or run `unzip ELEC415_515_ViT_Project.zip` in the
   folder you downloaded it to). Then `cd` into the folder it created, the one that
   contains this README and `run_sim.sh`. We call it `ELEC415_515_ViT_Project` in this
   document; if yours has another name (for example `project515-main` when you download
   it from GitHub), just use that name. For example:

   ```text
   cd ~/Downloads/ELEC415_515_ViT_Project
   ```

3. **Create the environment** from `environment.yml`. This downloads about 1–1.5 GB
   (2–3 GB once installed) and takes a few minutes, so it's a good time for a coffee:

   ```text
   conda env create -f environment.yml
   ```

4. **Activate it.** Do this in every new terminal before you work on the project:

   ```text
   conda activate vit-accel
   ```

   Your prompt now starts with `(vit-accel)`.

5. **Check that everything is there:**

   ```text
   verilator --version
   python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__)"
   ```

   The first command should print `Verilator 5.x`; the second prints two version numbers.

### 0.5 Using your own environment instead (optional)

If you'd rather manage your own environment, here's what you need: Python 3.10 or newer
with `numpy`, `torch` and `torchvision`; **Verilator 5.0 or newer** (version 4 won't work,
because the testbench needs `--timing`); `make`; and `g++` or `clang++`. On Ubuntu 24.04,
`sudo apt install verilator build-essential` gives you Verilator 5.020, which works;
Ubuntu 22.04 ships Verilator 4, which doesn't. On macOS, `brew install verilator` works.

### 0.6 Reproduce the baseline

Time to run the full model: no pruning, no quantization. **Run every command in this
document from the project folder** (the one that contains `run_sim.sh`), with the
environment activated.

**Step 1: export the model to the hardware format.**

```text
python python/export_float_weights.py
python python/export_rtl_data.py --samples 20
```

The first command turns the PyTorch checkpoint into plain arrays (`weights_float/`). The
second converts the weights to the hardware's fixed-point format (`weights/`), prepares 20
test images as hardware input (`data/`), and computes the golden outputs the testbench will
compare against. Together they print (with the full path of your project folder in place
of `...`):

```text
Exported 96 tensors of .../checkpoint/ViT_model.pt to .../weights_float
[keep_all] samples=20 fixed_acc=1.0000 float_acc=1.0000 fixed_vs_float_agree=1.0000
Weights written to .../weights
Test data written to .../data
```

`keep_all` is the name of the head mask that keeps every head (`head_masks/keep_all.txt`).

**Step 2: measure the accuracy on the whole test set.**

```text
python python/eval_fixed.py
```

This runs the floating-point model and the bit-exact hardware model on all 10,000 MNIST
test images (one to a few minutes, depending on your computer) and prints:

```text
case=keep_all samples=10000
weights of checkpoint/ViT_model.pt (sha256 88ed33458b36)
float accuracy : 99.4600%
Q4.12 accuracy : 99.4600%   (agrees with float on 100.0000%)
```

The second line names the checkpoint the numbers belong to (with the start of its content
hash, so two different files with the same name can't be confused). `Q4.12` is the
hardware's 16-bit fixed-point number format. With every head kept, the two accuracies are
identical: the hardware loses nothing compared with floating point. (Once you prune heads,
the two can differ by an image or two; that's normal.)

**Step 3: simulate the hardware.**

```text
./run_sim.sh
```

The first run compiles the simulator with Verilator. That takes about a minute (a few
minutes on a slow computer); later runs skip this step and take from a few seconds to about
a minute. The script sends the 20 exported images through the accelerator and checks every
result. You'll see one line per image as it comes out, then a summary:

```text
Compiling the build_16bit simulator with Verilator (about a minute)...
exported data: weights of checkpoint/ViT_model.pt (sha256 88ed33458b36)

======== ViT ACCELERATOR CHECK ========
build      = 16-bit (1 linear unit(s) per linear layer)
case       = keep_all   images = 20
head mask  = 1111 1111 1111 1111 1111 1111   (blocks 1 .. 6, heads 0 .. 3 as in the mask file)
MATCH    image=0 label=7 pred=7 latency=4311
...
======== SUMMARY ========
received_images             = 20 / 20
golden_mismatch_count       = 0
golden_logits_max_abs_diff  = 0
duplicate_results           = 0
label_accuracy              = 20 / 20
latency_single_image        = 4311 cycles
latency_all_images          = min 4311  avg 4311.0  max 4311 cycles
cycles_between_results      = 719.0
RESULT: PASS (RTL matches the Python fixed-point model bit-exactly)
- tb/tb_vit.sv:258: Verilog $finish
```

A few notes on this output:

- The last line is just Verilator telling you where the testbench ended the simulation.
- `MATCH` means the hardware output for that image is identical to the golden model. It
  doesn't say whether the prediction is *right*: with a pruned model you may see lines like
  `MATCH image=3 label=0 pred=2`, and that's fine. A difference from the golden model would
  show up as a `MISMATCH` line.
- `exported data` tells you which checkpoint the simulated data came from (more on this in
  0.8).

How to read the summary:

| Field | Meaning |
|---|---|
| `golden_mismatch_count`, `golden_logits_max_abs_diff` | should be 0: the hardware output equals the golden model exactly |
| `label_accuracy` | how many of the simulated images were classified correctly |
| `latency_single_image` | **the speed metric of this project**: clock cycles from the moment the first image enters the empty accelerator until its result is ready |
| `latency_all_images` | the same for every image; later images may have to wait behind earlier ones |
| `cycles_between_results` | average number of cycles between two results when images are sent back to back (the throughput) |
| `RESULT` | `PASS` when every image matches the golden model; `./run_sim.sh` then exits with status 0 (status 1 otherwise) |

Why both a simulation *and* an accuracy script? Simulating all 10,000 images would take a
long time. So the simulation checks a sample of images and proves that the hardware
computes *exactly* what the golden model computes; `eval_fixed.py` then runs that same
arithmetic on all 10,000 images to measure accuracy. Both start from the same exported
model (`weights_float/`), and the first lines of their output name it, so you can check
that they describe the same model.

You're set up when you see `RESULT: PASS`, a latency of 4311 cycles, and 99.46 % accuracy.

**What to include in your report (Part 0):**

- A screenshot of the terminal output of Step 2 (`eval_fixed.py`), showing the float and
  Q4.12 accuracy.
- A screenshot of the terminal output of Step 3 (`./run_sim.sh`), showing the summary with
  `RESULT: PASS`, `latency_single_image` and `cycles_between_results`.

### 0.7 Where to run commands, and how to name files

- **This document runs every command from the project folder.** The project folder can live
  anywhere and have any name, even one with spaces. The scripts find it from their own
  location, so they also work from other directories: `cd python` followed by
  `python eval_fixed.py` works too, and so does `/path/to/project/run_sim.sh`.
- Copy commands from this document as they are. Lines in the code boxes contain no
  comments, so they work in every shell (including zsh, the default on macOS).
- `./run_sim.sh` compiles the simulators into `obj_dir/`. If your project path contains a
  space, it compiles them into a folder under the system's temporary directory instead (the
  build tool Verilator uses can't handle spaces) and tells you where. Nothing changes for
  you: it recompiles whenever the content of `rtl/`, `tb/` or `run_sim.sh` changes.
- **File arguments accept a name or a path.** Wherever a command takes a file, you can just
  give its name, with or without the extension, and the script looks for it in the
  matching project folder:

  | Option | Looked up in | Example |
  |---|---|---|
  | head mask (`--mask`, `--case`, `./run_sim.sh <mask>`) | `head_masks/` | `prune4` |
  | checkpoint (`--checkpoint`) | `checkpoint/` | `ViT_model_finetuned.pt` |
  | quantization scales (`--int8`) | `quant/` | `scales_ptq_per_tensor` |

  The Python scripts also accept a path to an existing file (absolute, or relative to
  where you are), so `--mask head_masks/prune4.txt` works too. Output files given as a
  bare name (`--out`, `--out_checkpoint`) are written into the matching folder, and the
  extension is added if you leave it out; a name with a directory part (`./mine.pt`) is
  written exactly there.
- **Keep your head masks in `head_masks/`.** `export_rtl_data.py` exports every mask in
  that folder, and `./run_sim.sh` only runs masks that have been exported. So give
  `./run_sim.sh` the mask name (a path to a file inside `head_masks/` works as well), one
  mask per run. Only mask files belong in `head_masks/`.

**If something goes wrong**

- `verilator not found`: activate the environment with `conda activate vit-accel`.
- `No complete export ...`, `No exported test data for mask ...` or `... changed after the
  last export`: run the export command that `./run_sim.sh` prints (see 0.8).
- `Permission denied` for `./run_sim.sh`: run `chmod +x run_sim.sh` once.
- The compilation fails or the computer freezes: close other programs (compiling needs a
  few GB of memory). On WSL2, make sure the project is in `~/`, not `/mnt/c/`.
- `conda` asks you to accept Terms of Service: accept them, or run the command it prints.
- When a build fails, `./run_sim.sh` prints the end of the Verilator log and where to find
  the full log (usually `obj_dir/build_16bit.log` or `obj_dir/build_int8.log`).
- `TIMEOUT`: the hardware produced no result for 100,000 cycles. This only happens if you
  changed the RTL (Part 4); the design is stuck somewhere.

### 0.8 What the exported files describe

This one is worth reading carefully, because it saves you from confusing results later.
The simulation and `eval_fixed.py` never read a checkpoint directly. They read the files
written by the export scripts, and those files always describe **whatever you exported
last**:

| Folder | Written by | Contains | Read by |
|---|---|---|---|
| `weights_float/` | `export_float_weights.py [--checkpoint ...]` | the checkpoint as NumPy arrays | `eval_fixed.py`, `export_rtl_data.py` |
| `weights/`, `data/`, `data/<mask>/` | `export_rtl_data.py` | the fixed-point weights of `weights_float/`, the test images, golden outputs for every mask in `head_masks/` | `./run_sim.sh` |
| `quant/blockN/`, `data/int8/<mask>/` | `export_rtl_data.py --int8 [<scales>]` | the INT8 parameters and the INT8 golden outputs | `./run_sim.sh --int8` |

So, three rules of thumb:

1. **Changed the model or a mask? Run both exports again.** A new checkpoint needs both
   export scripts; a new or edited mask needs `export_rtl_data.py`. `./run_sim.sh` checks
   this for you: it refuses to run when `weights_float/` is newer than the last
   `export_rtl_data.py`, when a mask is missing from the export or was edited after it, or
   when the last export was interrupted.
2. **Using the INT8 build? Always export with `--int8`.** The INT8 build also reads
   `weights/` (biases, LayerNorm and the classifier stay 16-bit), so an export without
   `--int8` replaces part of the INT8 model. That's why such an export deletes
   `data/int8/`, and `./run_sim.sh --int8` then asks you to export with `--int8` again.
3. **Calibrate the INT8 scales on the checkpoint you export.** `export_rtl_data.py` and
   `eval_fixed.py` print a `WARNING` if the scales were calibrated on a different
   checkpoint than the one in `weights_float/` (they compare the content, not just the
   name).

The `exported data:` line of `./run_sim.sh` and the `weights of` line of `eval_fixed.py`
show which checkpoint (and which scales) each run used, so your logs always tell you what
you measured.

---

## Part 1: Understand the hardware (30 %)

Before you start making the accelerator faster, get to know it. Read the RTL and draw a
**block diagram of the accelerator** that shows how one image flows through it.

Good places to start (Appendix A describes every file):

- `rtl/vit_top.sv`: the top level, which connects the encoder blocks and the final stage;
- `rtl/transformer_block_head_serial.sv`: one encoder block, the most important file;
- `rtl/linear_row_unit.sv`: the unit that computes the linear layers;
- `rtl/final_stage.sv`: the final LayerNorms and the classifier;
- `tb/tb_vit.sv`: how the testbench drives the accelerator.

The comments in the RTL walk you through each step. You don't need to understand every
line of the fixed-point arithmetic; focus on the structure and on how the data moves.

Your diagram should show:

1. **The top level:** the encoder blocks and the final stage, how they hand images to each
   other, and where the weights, the INT8 parameters and the head mask enter the hardware.
   In this design the weights aren't streamed in: two memories load them from files when
   the simulation starts and feed them to every block, so feel free to draw them as ROMs.
2. **The inside of one encoder block, with its main datapath:** every step of the
   computation in the order the hardware performs it (from the input token matrix to the
   block output), the registers that hold intermediate results between steps, and the
   compute units that do each step.
3. **The two compression hooks:** where the head mask changes what the block computes, and
   what the `LINEAR_INT8` build changes in the hardware.

A quick note on terminology: a *linear unit* (`linear_row_unit.sv`) processes one token
row per cycle and computes all the output channels it is responsible for at once. Every
block has its own units for Q, K, V, FC1 and FC2. The FC1 and FC2 units compute all 128
or 64 outputs of a layer; the Q, K and V units compute only the 16 channels of the head
currently being processed, so they run once per kept head. When we say "1 linear unit per
linear layer" (16-bit build) or "4" (INT8 build), we mean the number of copies of each of
them. The units differ in size: a Q unit has 16 × 64 multipliers, an FC1 unit 128 × 64.

Also answer these questions briefly:

- How many clock cycles does one encoder block spend on one image, and which steps
  contribute those cycles? The header comments of `vit_top.sv` and
  `transformer_block_head_serial.sv` give you the cycle formulas. Explain where every term
  comes from, and use them to explain the 4311-cycle latency from Part 0, including the
  `- 1` and the `+ 3`. (Hint: look at what happens in the cycle in which one stage hands an
  image to the next.)
- Why can a new image enter the accelerator before the previous one has left it, and what
  limits how often a new result comes out?

**What to include in your report (Part 1):** the block diagram and your answers.

---

## Part 2: Attention-head pruning (30 %)

### How the hardware supports pruning

Each encoder block computes attention **one head at a time**, and a **head mask** input
tells it which heads to compute. A pruned head is skipped completely: none of its work is
done, and it costs zero clock cycles. Its output is treated as zero, which is exactly what
removing the head from the model means.

You control the mask with a text file in `head_masks/`: six rows (encoder blocks 1 to 6)
of four numbers (heads 0 to 3), where `1` = keep and `0` = prune. For example,
`head_masks/keep_all.txt` keeps every head:

```text
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
```

To try a mask `head_masks/<name>.txt` on the hardware, run these three commands. The first
exports the golden outputs for every mask in `head_masks/`, the second measures the
accuracy with this mask on all 10,000 images, and the third simulates it (correctness and
cycles):

```text
python python/export_rtl_data.py --samples 20
python python/eval_fixed.py --case <name>
./run_sim.sh <name>
```

The PyTorch model understands the same mask: `model.set_head_mask(mask)` multiplies the
output of every head by the matching entry of a 6 × 4 tensor.

### Your task

1. **Rank the heads by importance.** Fill in the three TODOs in `python/head_pruning.py`.
   The script follows Michel et al., *"Are Sixteen Heads Really Better than One?"*
   (NeurIPS 2019): every head gets a gate ξ that multiplies its output, and a head's
   importance is the magnitude of the gradient of the loss with respect to its gate,
   |∂L/∂ξ|, measured on training data with every gate at 1. Our script takes this gradient
   per batch of images and averages its magnitude over all batches (the `raw_importance`).
   As in the paper, the four scores of each block are then normalized to unit L2 norm (the
   `normalized_importance`), so that blocks can be compared; heads are ranked and pruned by
   the **normalized** score. Each TODO comes with a hint about what the missing code should
   do.
2. **Prune.** Run

   ```text
   python python/head_pruning.py --prune 4
   ```

   The script masks heads one at a time, least important first, and prints the test
   accuracy after each step. It saves the accuracy curve to `results/head_pruning.csv` and
   both importance scores to `results/head_importance.csv`, and writes the mask with 4
   heads removed to `head_masks/prune4.txt`. A run takes a few minutes (the first run also
   downloads the MNIST training set). The script uses a fixed random seed, so every run
   gives the same ranking: running it with different `--prune` values (1 to 24) gives you
   nested masks and the same CSV files.

   Try several values of `--prune`. One thing to notice: the greedy ranking doesn't care
   which block a head belongs to. Think about which block limits `cycles_between_results`,
   and write at least one mask by hand (in `head_masks/`) that takes this into account.
   Give your own masks names that don't start with `prune` (say, `balanced6.txt`), because
   `head_pruning.py` overwrites `prune<N>.txt`.
3. **Measure each mask on the hardware.** Export, evaluate and simulate it as shown above.
   Check that the simulation passes, and write down the accuracy (the `Q4.12 accuracy` line
   of `eval_fixed.py`), `latency_single_image` and `cycles_between_results`.
4. **Fine-tune (optional, but recommended).** Pruning costs some accuracy, and a short
   fine-tuning with the mask applied lets the remaining heads make up for it (3 epochs take
   a few minutes on a laptop CPU). Give each fine-tuned model its own name, then export,
   evaluate and simulate it:

   ```text
   python python/finetune.py --mask prune4 --out_checkpoint ViT_model_ft_prune4.pt
   python python/export_float_weights.py --checkpoint ViT_model_ft_prune4.pt
   python python/export_rtl_data.py --samples 20
   python python/eval_fixed.py --case prune4
   ./run_sim.sh prune4
   ```

   Keep in mind that the fine-tuned weights are tuned for this one mask: while they're
   exported, only the `prune4` numbers mean anything. Fine-tuning changes the weights, not
   the schedule, so the cycle counts stay the same. To go back to the original model, run
   `python python/export_float_weights.py` without arguments and export again.

### What to include in your report (Part 2)

- The importance of the 24 heads (the `normalized_importance` column, as a table or a heat
  map) and the accuracy as heads are removed.
- For several masks, including your hand-written one: accuracy, `latency_single_image` and
  `cycles_between_results`, before and (if you did it) after fine-tuning. Plot accuracy
  against `latency_single_image`; any tool is fine (a spreadsheet, or matplotlib, which you
  can add with `conda install -c conda-forge matplotlib`).
- An explanation of how the cycle counts change with the number of pruned heads, and why.
  Compare the improvement in latency with the improvement in `cycles_between_results`, and
  explain any difference you see.

---

## Part 3: INT8 quantization of the linear layers (30 %)

### What gets quantized

Only the **basic linear layers**: in each of the 6 encoder blocks, the Q, K and V
projections and the two MLP layers (FC1, FC2), so 30 layers in total. Everything else stays
as it is, in fixed point: LayerNorm, softmax, GELU, the attention products Q·Kᵀ and P·V,
and the classifier. You don't touch any of that.

### How the hardware supports INT8

The hardware comes in two builds:

| Build | Linear layers | Linear compute units per layer |
|---|---|---|
| 16-bit (default, Parts 0–2) | 16-bit × 16-bit multiplications | 1 |
| `LINEAR_INT8` | 8-bit × 8-bit multiplications | 4 |

An 8-bit multiplier takes about a quarter of the area of a 16-bit one, so four INT8
multiplier arrays take about the area of the one 16-bit array they replace. The narrower
data lets the hardware do more work per cycle in roughly the same area. (Strictly
speaking, each INT8 unit also needs a few wider multipliers to convert its inputs to 8 bit
and its outputs back to 16 bit; we leave those out of the comparison.)

Switch to the INT8 build with `--int8`:

```text
./run_sim.sh --int8
./run_sim.sh --int8 prune4
```

The first runs the INT8 build with every head kept, the second with a head mask. The INT8
hardware reads the quantized weights and scale factors from `quant/`. You don't write those
files by hand: you compute the **scales** in Python, and `export_rtl_data.py --int8` turns
them into the integers the hardware uses.

Note that `--int8` means slightly different things to the two kinds of commands:

- For the Python scripts (`export_rtl_data.py`, `eval_fixed.py`), `--int8` can be followed
  by the name of a scales file: `--int8 scales_ptq_per_tensor`. Without a name it uses
  `quant/scales_ptq.npz`. The head mask is a separate option (`--case` for
  `eval_fixed.py`).
- For `./run_sim.sh`, `--int8` only selects the INT8 build. The simulation always runs the
  scales of your most recent `export_rtl_data.py --int8`, and a word after `--int8` is
  the head mask.

### The quantization scheme

Use symmetric linear quantization with zero point 0 and 8-bit integers in [−127, 127]:

```text
q = clip(round(r / s), -127, 127)        r ≈ s · q
```

Each of the 30 layers needs two kinds of scale:

- **Weights: one scale per output channel** (per row of the weight matrix W):
  s_w = max |W_row| / 127, so the largest weight of each row maps to ±127. Also try
  **per-tensor** scales (one scale for the whole matrix) to compare.
- **Activations (the layer input): one scale per layer:** s_x = max |x| / 127, where
  max |x| is the largest input magnitude the layer sees while the model runs on a few
  batches of **training** images (this is called calibration). Never calibrate on test
  images.

This is post-training quantization (PTQ): the trained weights are quantized as they are,
with no retraining.

For reference, here's what the INT8 hardware computes for one linear layer. You don't
implement this yourself: `export_rtl_data.py` derives `x_mult`, `y_mult` and the 8-bit
weights `w8` from your scales, and `rtl/linear_row_unit.sv` and `python/vit_fixed.py`
already implement it.

```text
x_q  = clip(round(x / s_x), -127, 127)                  input -> 8 bit
acc  = sum over d of  x_q[d] * w8[j][d]                 8 bit x 8 bit, 32-bit sum
y[j] = acc * s_x * s_w[j] + bias[j]                     back to 16-bit fixed point
```

### Your task

1. **Implement PTQ.** Fill in the three TODOs in `python/quantize_ptq.py` (the weight
   scales, the calibration of the activation range, and the activation scale), then run

   ```text
   python python/quantize_ptq.py
   ```

   It calibrates on training images and saves the scales to `quant/scales_ptq.npz`.
2. **Export, evaluate and simulate the INT8 model:**

   ```text
   python python/export_rtl_data.py --samples 20 --int8
   python python/eval_fixed.py --int8
   ./run_sim.sh --int8
   ```

   `eval_fixed.py` now also reports the accuracy of the INT8 hardware model, next to the
   float and 16-bit ones. The simulation should report `RESULT: PASS`, and its
   `exported data:` line tells you which scales it used.

   Calibration always runs with every head kept, so one set of scales works for every
   mask. If you fine-tuned a model in Part 2, calibrate on the checkpoint you export
   (rule 3 in 0.8), for example:

   ```text
   python python/quantize_ptq.py --checkpoint ViT_model_ft_prune4.pt --out scales_ft_prune4
   ```

3. **Compare per-channel and per-tensor weight scales:**

   ```text
   python python/quantize_ptq.py --per_tensor_weights --out scales_ptq_per_tensor
   python python/export_rtl_data.py --samples 20 --int8 scales_ptq_per_tensor
   python python/eval_fixed.py --int8 scales_ptq_per_tensor
   ./run_sim.sh --int8
   ```

   Both scale files give the same cycle counts, so this comparison is about accuracy only.
   With such a small model the difference may be tiny, or even go the "wrong" way; that's
   a fine result to report.
4. **Combine with pruning.** Run your pruning masks from Part 2 on the INT8 build. Export
   once the mask files exist, then measure the accuracy and simulate each mask, for
   example:

   ```text
   python python/export_rtl_data.py --samples 20 --int8
   python python/eval_fixed.py --case prune4 --int8
   ./run_sim.sh --int8 prune4
   ```

   Note that `--int8` without a file name switches back to the per-channel scales in
   `scales_ptq.npz`.

### What to include in your report (Part 3)

- The accuracy of the float, 16-bit and INT8 models (per channel and per tensor).
- `latency_single_image` and `cycles_between_results` of the 16-bit and the INT8 build,
  with and without pruning.
- An explanation of where the INT8 speed-up comes from, and why it's smaller than the 4×
  increase in linear units.

---

## Part 4 (Bonus): Make it as fast as you can

Now it's your turn to get creative. Make the accelerator as fast as possible while keeping
the accuracy at **98.5 % or higher** (at most about one point below the original 99.46 %).

- **Speed** is `latency_single_image` from `./run_sim.sh` for your final design.
- **Accuracy** is measured on all 10,000 test images with `eval_fixed.py`, and your final
  design must pass the simulation (`RESULT: PASS`), so the accuracy you report really is
  the accuracy of your hardware.

Anything goes: more aggressive or better-balanced pruning, fine-tuning, quantizing more of
the network or to fewer bits, or changing the hardware itself (the dataflow, the
scheduling, how the compute units are used). There are just two rules:

1. **Comparable compute resources.** You can't get your speed-up by simply adding hardware,
   for example by doubling the number of compute units. Your design should use compute
   resources comparable to the provided one. Count an 8-bit multiplier as a quarter of a
   16-bit multiplier (the same rule that gives the INT8 build its four units), and keep
   track of the resources you use.
2. **Keep the golden model exact.** If you change the hardware's arithmetic, update
   `python/vit_fixed.py` (and `export_rtl_data.py` if needed) so it still models your
   hardware exactly and the simulation still passes.

In your presentation, explain **where your speed-up comes from, how you built it, and why
it's valid**: show the cycle counts and the accuracy, and make the case that your resources
are comparable to the original design. Convince the instructor and the TAs that your
approach is sound and effective. The four fastest valid designs earn extra bonus points,
ranked by speed.

---

## Appendix A: Project files

```text
ELEC415_515_ViT_Project/
├── README.md                 this document
├── project_description.pdf   the same document as a PDF
├── environment.yml           conda environment: Python packages, Verilator, compiler
├── run_sim.sh                builds (when needed) and runs the simulation
├── checkpoint/
│   └── ViT_model.pt          the trained PyTorch model
├── data/
│   └── mnist/                MNIST test set (the training set is downloaded when needed)
├── head_masks/
│   └── keep_all.txt          head mask that keeps every head; add your own masks here
├── python/                   Python scripts
├── rtl/                      the accelerator in SystemVerilog
└── tb/
    └── tb_vit.sv             the testbench
```

The scripts create a few more folders as you go: `weights_float/`, `weights/`,
`data/<mask>/`, `data/int8/<mask>/` and `quant/` (by the export and quantization scripts;
see 0.8), `results/` (by `head_pruning.py`) and `obj_dir/` (the compiled simulators). The
small `source.txt` files in `weights_float/`, `data/` and `data/int8/` record which
checkpoint and scales were exported.

### Python scripts (`python/`)

| File | Purpose |
|---|---|
| `head_pruning.py` | **Part 2, you complete it.** Head importance (Michel et al.) and greedy pruning; writes `head_masks/prune<N>.txt` and `results/*.csv` |
| `quantize_ptq.py` | **Part 3, you complete it.** Post-training INT8 quantization; writes the scales to `quant/scales_ptq.npz` |
| `finetune.py` | fine-tunes the model with a head mask applied; writes `checkpoint/ViT_model_finetuned.pt` or the file given with `--out_checkpoint` |
| `export_float_weights.py` | converts a checkpoint into one NumPy array per tensor (`weights_float/`) |
| `export_rtl_data.py` | exports everything the simulation needs: fixed-point weights, test images, golden outputs for every mask; with `--int8 <scales>` also the INT8 parameters and INT8 golden outputs |
| `eval_fixed.py` | accuracy of the float model and the bit-exact hardware model(s) on the 10,000 test images; `--case <mask>`, `--int8 <scales>` |
| `vit_fixed.py` | the golden model: a bit-exact Python model of the 16-bit and the INT8 hardware, plus the floating-point reference model |
| `model.py` | the PyTorch ViT, including `set_head_mask()` |
| `data.py` | MNIST data loaders and the accuracy evaluation used by the PyTorch scripts |

### Hardware (`rtl/`)

| File | Purpose |
|---|---|
| `vit_top.sv` | top level: six encoder blocks and the final stage connected by a valid/ready handshake; the head mask is an input |
| `transformer_block_head_serial.sv` | one encoder block: a state machine that computes attention head by head and streams the tokens through its compute units |
| `linear_row_unit.sv` | a linear compute unit (one token per cycle); the 16-bit or the INT8 datapath, selected by `LINEAR_INT8` |
| `final_stage.sv` | the two final LayerNorms and the classifier |
| `norm_stage_50x64.v` | LayerNorm over the 50 tokens |
| `gelu_stage_50x128.v` | GELU through a lookup table |
| `residual_add_50x64.v` | residual addition |
| `classifier_head_64x10.v` | the classifier on the CLS token and the arg-max |
| `vit_weight_memory.sv` | loads the 16-bit weights from `weights/` at the start of the simulation |
| `vit_quant_memory.sv` | loads the INT8 parameters from `quant/` (INT8 build) |
| `fixed_params.vh` | model dimensions, number formats, and the `LINEAR_INT8` switch with the number of linear units |
| `fixed_math.vh` | saturation and integer square root functions |
| `exp_lut.vh`, `gelu_lut.vh`, `tanh_lut.vh` | lookup tables for exp, GELU and tanh |

### Simulation

| File | Purpose |
|---|---|
| `run_sim.sh` | `./run_sim.sh [--int8] [mask]`; checks that the mask and the model have been exported, builds the simulator with Verilator when the RTL has changed, then runs it; exits with status 0 only on `RESULT: PASS` |
| `tb/tb_vit.sv` | the testbench: loads the head mask and the exported images, runs them through the accelerator, compares with the golden outputs, reports the cycle counts |

## Appendix B: Command reference

Run these from the project folder, with `conda activate vit-accel` (see 0.7).

| Goal | Command |
|---|---|
| Export a checkpoint | `python python/export_float_weights.py [--checkpoint <checkpoint>]` |
| Export hardware data (16-bit) | `python python/export_rtl_data.py --samples 20` |
| Export hardware data (16-bit and INT8) | `python python/export_rtl_data.py --samples 20 --int8 [<scales>]` |
| Accuracy on the test set | `python python/eval_fixed.py [--case <mask>] [--int8 [<scales>]]` |
| Head pruning | `python python/head_pruning.py --prune <N>` |
| Fine-tune with a mask | `python python/finetune.py --mask <mask> [--checkpoint <checkpoint>] [--out_checkpoint <checkpoint>] [--epochs N]` |
| INT8 post-training quantization | `python python/quantize_ptq.py [--checkpoint <checkpoint>] [--per_tensor_weights] [--out <scales>]` |
| Simulate | `./run_sim.sh [--int8] [<mask>]` |

Square brackets mark optional parts; leave the brackets out when you type a command.
`<mask>`, `<checkpoint>` and `<scales>` are file names in `head_masks/`, `checkpoint/` and
`quant/` (extension optional), or paths; see 0.7. Without a file name, `--int8` uses
`quant/scales_ptq.npz`.
