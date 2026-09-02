# 🪟 Complete Windows Setup, Execution & Results Guide

This document provides a comprehensive, step-by-step guide for **Windows users** to clone the repository, set up the environment, run unit tests, execute evaluation master scripts, generate all benchmark results, and view the generated reports, plots, and LaTeX presentation on Windows.

---

## 📑 Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Step 1: Cloning the Repository on Windows](#step-1-cloning-the-repository-on-windows)
3. [Step 2: Virtual Environment & Dependency Setup](#step-2-virtual-environment--dependency-setup)
4. [Step 3: Running All Unit & Integration Tests](#step-3-running-all-unit--integration-tests)
5. [Step 4: Running Master Scripts & Generating Results](#step-4-running-master-scripts--generating-results)
   - [4.1 End-to-End Pipeline Evaluation](#41-end-to-end-pipeline-evaluation)
   - [4.2 Master Research & Ablation Comparison Suite](#42-master-research--ablation-comparison-suite)
   - [4.3 Single Video Pipeline Runner](#43-single-video-pipeline-runner)
   - [4.4 Sub-Component Experiments](#44-sub-component-experiments)
6. [Step 5: How to View Generated Results on Windows](#step-5-how-to-view-generated-results-on-windows)
   - [Viewing CSV Data Tables in Microsoft Excel](#viewing-csv-data-tables-in-microsoft-excel)
   - [Viewing Markdown Reports in VS Code or Notepad](#viewing-markdown-reports-in-vs-code-or-notepad)
   - [Viewing PNG Publication Figures in Windows Photos](#viewing-png-publication-figures-in-windows-photos)
   - [Viewing Evidence Video Clips & Thumbnails](#viewing-evidence-video-clips--thumbnails)
7. [Step 6: Compiling & Viewing the Beamer Presentation (`1_presentation.tex`)](#step-6-compiling--viewing-the-beamer-presentation)
   - [Option A: Overleaf (Recommended - Fast & No Setup)](#option-a-overleaf-recommended---fast--no-setup)
   - [Option B: Local LaTeX via PowerShell / Command Prompt](#option-b-local-latex-via-powershell--command-prompt)
8. [Troubleshooting & Windows Tips](#8-troubleshooting--windows-tips)

---

## 1. Prerequisites

Before starting, ensure the following software is installed on your Windows machine:

1. **Python 3.10 or 3.11** (64-bit):
   - Download from [python.org/downloads](https://www.python.org/downloads/).
   - ⚠️ **CRITICAL:** During installation, check the box **"Add python.exe to PATH"**.
2. **Git for Windows**:
   - Download and install from [git-scm.com](https://git-scm.com/).
3. *(Optional for NVIDIA GPU acceleration)*:
   - Latest NVIDIA GPU Drivers + **CUDA Toolkit 11.8 or 12.1**.

---

## Step 1: Cloning the Repository on Windows

1. Open **Windows PowerShell** or **Command Prompt** (press `Win + R`, type `powershell`, and press Enter).
2. Navigate to your desired working directory (e.g., `C:\Projects`):
   ```powershell
   cd C:\Projects
   ```
3. Clone the GitHub repository to your Windows machine:
   ```powershell
   git clone https://github.com/thilaganiniyavan/chainSnatching.git
   ```
4. Navigate into the cloned folder:
   ```powershell
   cd chainSnatching
   ```

---

## Step 2: Virtual Environment & Dependency Setup

### 1. Create Python Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate Virtual Environment
- **PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Command Prompt (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

> 💡 **PowerShell Execution Policy Error Fix:**
> If PowerShell displays an error saying `running scripts is disabled on this system`, run this one-time command:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then run `.\venv\Scripts\Activate.ps1` again. You should see `(venv)` at the beginning of your terminal prompt.

### 3. Install PyTorch (Choose GPU or CPU version)

- **For NVIDIA GPU (CUDA 12.1):**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **For CPU-Only:**
  ```powershell
  pip install torch torchvision torchaudio
  ```

### 4. Install Project Requirements
```powershell
pip install -r requirements.txt
```

---

## Step 3: Running All Unit & Integration Tests

To run the master unit test suite (164 passing tests across 27 test modules) and confirm that all code logic works cleanly:

```powershell
python -m pytest -v
```

**Expected Output:**
```text
tests/test_action_post_processor.py :: PASSED
tests/test_action_recognizer.py :: PASSED
tests/test_behaviour_engine.py :: PASSED
...
=================== 164 passed in ~12.5s ===================
```

---

## Step 4: Running Master Scripts & Generating Results

The system contains master evaluation scripts that execute the 13-stage AI pipeline, run benchmarks, measure throughput, and generate output reports, CSV tables, and charts.

### 4.1 End-to-End Pipeline Evaluation
Executes the full pipeline across all benchmark CCTV videos in `Snatch 1.0`, measuring precision, recall, F1-score, stage latency, frame reduction cascade, and resource usage:

```powershell
python apps\run_end_to_end_evaluation.py `
    --input-dir "Snatch 1.0\Chain Snatching Videos\Snatch Theft" `
    --output-dir "outputs\evaluation_results"
```

### 4.2 Master Research & Ablation Comparison Suite
Runs systematic comparison across object detectors (YOLOv8 vs YOLO11), trackers (ByteTrack vs BoT-SORT), pose models, fusion engines, and pipeline configurations (Config A, B, C, D):

```powershell
python apps\run_all_research_comparisons.py
```

### 4.3 Single Video Pipeline Runner
Runs the full pipeline interactively on a specific CCTV video file and displays keyframe overlays / progress logs:

```powershell
python apps\pipeline_runner.py `
    --input "Snatch 1.0\Chain Snatching Videos\Snatch Theft\1.mp4" `
    --backend mediapipe `
    --norm hip_centered `
    --action-backend stgcn `
    --fusion-strategy weighted_confidence
```

### 4.4 Sub-Component Experiments
To execute specific targeted component experiments:

- **Motion Triage Benchmark:**
  ```powershell
  python apps\run_motion_experiment.py
  ```
- **Forensic Search Index Benchmark:**
  ```powershell
  python apps\run_forensic_validation.py
  ```
- **Pipeline Architecture Validation:**
  ```powershell
  python apps\run_architecture_validation.py
  ```

---

## Step 5: How to View Generated Results on Windows

After running the master scripts above, all outputs are saved under the `outputs\` directory. Here is how to view each type of result on Windows:

```text
outputs/
├── evaluation_results/
│   ├── pipeline_statistics.csv         <-- Per-video precision, recall, F1, FPS
│   ├── stage_statistics.csv            <-- Frame retention & search-space reduction
│   ├── runtime_statistics.csv          <-- Millisecond timing per stage
│   ├── system_resource_usage.csv       <-- CPU, GPU VRAM, RAM utilization
│   ├── framework_summary.md            <-- Comprehensive markdown report
│   └── figures/                        <-- Generated PNG plots & charts
├── research_comparison/
│   ├── master_results.csv              <-- Master experimental matrix table
│   ├── master_results.json             <-- Full machine-readable dataset
│   ├── master_research_report.md       <-- Auto-compiled research report
│   └── figures/                        <-- Comparative bar charts & tradeoff plots
├── forensic_thumbnails/                <-- Keyframe image crops of detected events
└── forensic_clips/                     <-- Annotated MP4 evidence video clips
```

### Viewing CSV Data Tables in Microsoft Excel
1. Open **File Explorer** (`Win + E`) and navigate to `C:\Projects\chainSnatching\outputs\evaluation_results\`.
2. Double-click any `.csv` file (`pipeline_statistics.csv`, `stage_statistics.csv`, `runtime_statistics.csv`, or `master_results.csv`).
3. It will automatically open in **Microsoft Excel** (or any spreadsheet viewer like LibreOffice Calc or Notepad).
4. You will see columns for `Precision`, `Recall`, `F1-Score`, `FPS`, `Latency_ms`, `Search_Space_Reduction_%`, etc.

### Viewing Markdown Reports in VS Code or Notepad
1. Open **VS Code** (or Notepad).
2. Open `outputs\evaluation_results\framework_summary.md` or `outputs\research_comparison\master_research_report.md`.
3. In VS Code, press `Ctrl + Shift + V` to render the formatted **Markdown Preview** with headers, bold text, and tables.

### Viewing PNG Publication Figures in Windows Photos
1. Open **File Explorer** and navigate to `outputs\evaluation_results\figures\` or `outputs\research_comparison\figures\`.
2. Double-click any image (`model_f1_comparison.png`, `search_space_progression.png`, `ablation_runtime_searchspace.png`, `stage_runtime_contribution.png`).
3. Images will open in **Windows Photos App** for inspection.

### Viewing Evidence Video Clips & Thumbnails
1. Navigate to `outputs\forensic_clips\`.
2. Double-click any generated `.mp4` clip to play in **Windows Media Player** or **VLC Media Player**.
3. View associated keyframe thumbnails in `outputs\forensic_thumbnails\`.

---

## Step 6: Compiling & Viewing the Beamer Presentation (`1_presentation.tex`)

The review presentation source file is located at:
`documentation\presentation\1_presentation.tex`

### Option A: Overleaf (Recommended - Fast & No Setup)
1. Open [Overleaf.com](https://www.overleaf.com) and log in.
2. Click **New Project** $\to$ **Blank Project** (name it `Chain_Snatching_Presentation`).
3. Upload all files from `documentation\presentation\` (`1_presentation.tex`, `ref.bib`).
4. Upload required diagram images from `documentation\paper\` into the project root:
   - `logo.png`
   - `methodology.png`
   - `search_space_progression.png`
   - `ablation_runtime_searchspace.png`
   - `stage_runtime_contribution.png`
5. Select **pdfLaTeX** as the compiler in **Menu** $\to$ **Compiler**.
6. Click **Recompile** to build and download `1_presentation.pdf`.

### Option B: Local LaTeX via PowerShell / Command Prompt
If you have **MiKTeX** installed on Windows:

```powershell
cd documentation\presentation

# Pass 1: Initial LaTeX compile
pdflatex -interaction=nonstopmode 1_presentation.tex

# Pass 2: BibTeX references
bibtex 1_presentation

# Pass 3 & 4: Resolve page numbers and frame references
pdflatex -interaction=nonstopmode 1_presentation.tex
pdflatex -interaction=nonstopmode 1_presentation.tex
```

Double-click `1_presentation.pdf` in File Explorer to view the presentation in **Adobe Acrobat**, **Microsoft Edge**, or **Google Chrome**.

---

## 8. Troubleshooting & Windows Tips

1. **Path Quotes for Spaces:**
   - Always wrap file paths containing spaces in double quotes: `"Snatch 1.0\Chain Snatching Videos\Snatch Theft\1.mp4"`.

2. **OpenCV GUI Display in Windows:**
   - If running in a remote terminal or headless environment, pass `--no-display` to `pipeline_runner.py` to disable live OpenCV window rendering.

3. **Windows Long Paths Enabling:**
   - If deep nested paths cause file access warnings, open PowerShell as **Administrator** and run:
     ```powershell
     New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
     ```

4. **Cleaning Output Artifacts:**
   - To clean prior test runs and re-execute fresh benchmarks, delete or rename the `outputs\` directory:
     ```powershell
     Remove-Item -Recurse -Force outputs
     ```
