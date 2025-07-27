# Windows Real-Time Face‑Swapping Livestream Stack

This repository contains a proof‑of‑concept for running a **real‑time face‑swap livestream pipeline** on a Windows workstation with an NVIDIA GPU.  The goal is to let you swap faces live on video using [DeepFaceLive](https://github.com/iperov/DeepFaceLive), [Roop‑Cam](https://github.com/hacksider/roop-cam), and OpenAI’s [Whisper](https://github.com/openai/whisper) for live captions, then stream the result via **OBS Studio**.  A small Flask/Gradio dashboard exposes toggles for choosing models, faces and enabling NSFW mode, and it launches OBS with pre‑configured scenes.

This guide assumes you are running **Windows 10/11** on a machine equipped with a CUDA‑enabled NVIDIA card (RTX 30XX or better is recommended), and that you have administrative rights.  Please read through the entire document before starting.

## 1. Prepare Your System

### 1.1 Install Anaconda/Miniconda

DeepFaceLive is sensitive to Python versions and works best on Python 3.9.  The Plain English guide for installing DeepFaceLive suggests using **Anaconda** to isolate a compatible Python environment【448643498315968†L101-L120】.  Download and install **Anaconda** or **Miniconda** for Windows from the official website.  During installation check the box to *Add Anaconda to your PATH*.

1. Download the [Anaconda installer](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.anaconda.com/free/miniconda/).  
2. Run the installer as administrator and follow the on‑screen instructions.  
3. Open the **Anaconda Prompt** from your Start menu (don’t use Anaconda Navigator)【448643498315968†L101-L117】.

### 1.2 Create the Python Environment

From the Anaconda Prompt, create an environment named `faceswap` with Python 3.9 (DeepFaceLive fails on later versions【448643498315968†L108-L114】):

```cmd
conda create -n faceswap python=3.9
conda activate faceswap
```

### 1.3 Install System Dependencies

Several components require additional packages.  Install **Git**, **ffmpeg**, and **Visual Studio 2022 C++ runtimes** (needed by onnxruntime) if you don’t already have them:

```powershell
# Install Chocolatey package manager (administrator PowerShell)【72765743008919†L144-L152】
Set-ExecutionPolicy Bypass -Scope Process -Force; \
  [System.Net.ServicePointManager]::SecurityProtocol = \
  [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; \
  iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Use chocolatey to install Git and ffmpeg【72765743008919†L154-L158】
choco install git ffmpeg -y

# Install the Microsoft Visual C++ 2022 redistributable (if missing)
choco install vcredist-all -y
```

### 1.4 Install CUDA (Optional for GPU acceleration)

If your GPU supports CUDA and you want GPU‑accelerated face swapping, install the **CUDA Toolkit 11.8** from NVIDIA.  The roop‑cam README recommends CUDA 11.8 for GPU execution and instructs to install the onnxruntime‑gpu package afterwards【625101666220878†L356-L365】.  After installing CUDA:

```cmd
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu==1.15.1
```

This allows roop‑cam to use CUDA.  If you don’t have a GPU, skip this step; roop will run on CPU but will be slower.【625101666220878†L326-L349】

## 2. Clone and Install the Applications

All projects are installed inside the `faceswap` conda environment.

### 2.1 DeepFaceLive

1. Clone the repository:
   ```cmd
   git clone https://github.com/iperov/DeepFaceLive.git
   cd DeepFaceLive
   ```

2. Install the Python dependencies.  DeepFaceLive depends on PyQt6, NumPy, OpenCV and other packages.  According to the installation guide, you may need to install PyQt6, NumPy, opencv and h5py manually【448643498315968†L140-L154】.  Run:

   ```cmd
   pip install PyQt6 --upgrade --force-reinstall
   pip install numpy numexpr onnx onnxruntime
   conda install -c conda-forge opencv h5py -y
   ```

3. Launch DeepFaceLive to download models and verify functionality:
   ```cmd
   python main.py run DeepFaceLive --user-data-dir %USERPROFILE%\DeepFaceLive
   ```
   The first run downloads model files and caches them in the specified `user-data-dir`【448643498315968†L140-L150】.

### 2.2 Roop‑Cam

Roop‑Cam provides real‑time face swap from a single photo.  The README outlines a simple installation for Windows:

1. Download the `windows_run.bat` archive from the roop‑cam GitHub releases page and extract it to a folder **without spaces**【625101666220878†L315-L324】.

2. Double‑click **`windows_run.bat`**, which downloads dependencies and prepares a Python environment【625101666220878†L317-L324】.

3. After it finishes, open the extracted `roop-cam` folder and double‑click **`run-cuda-windows.bat`** to launch roop with CUDA acceleration【625101666220878†L315-L326】.  For CPU‑only execution, run `python run.py`【625101666220878†L346-L350】.

Alternatively you can clone the repository and install manually:

```cmd
git clone https://github.com/hacksider/roop-cam.git
cd roop-cam
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt【625101666220878†L338-L346】
# (Optional) enable CUDA execution provider
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.15.1【625101666220878†L356-L365】
python run.py --execution-provider cuda
```

### 2.3 Whisper

Install Whisper in its own environment or reuse the `faceswap` environment.  The GPUMart tutorial suggests creating a separate conda env and installing PyTorch and Whisper【72765743008919†L116-L139】.  To keep things simple here, we install directly into the `faceswap` environment:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/whisper.git【72765743008919†L160-L165】
```

This installs Whisper with GPU support.  If you encounter CUDA errors, you can install `pip install whisper` for CPU‑only inference.

## 3. Install and Configure OBS Studio

OBS Studio provides video capture and streaming.  Download the Windows installer from the [official OBS website](https://obsproject.com/download)【277180711580014†L24-L33】 and run through the setup wizard.  During installation you can leave optional plugins unchecked【278814849728271†L61-L68】.  After launching OBS for the first time, run the **Auto‑Configuration Wizard** to let OBS pick optimal settings for your hardware【278814849728271†L71-L80】.

### Virtual Camera

OBS includes a built‑in **Virtual Camera** on Windows (found in the Controls pane).  After starting the virtual camera, the output of your OBS scene appears as a webcam device that other applications can see.  We will route the face‑swapped video into OBS via a **Python virtual camera** (`pyvirtualcam`) and then enable the OBS virtual camera so your streaming platform sees the final composite.

## 4. Run the Pipeline

### 4.1 Create the Dashboard

We provide a simple Flask/Gradio dashboard in `python/pipeline_gui.py`.  The GUI lets you:

* Choose between DeepFaceLive and Roop‑Cam for face swapping.
* Upload or select a source face.
* Toggle NSFW content (if you decide to load uncensored models).
* Start or stop the livestream.

The dashboard launches a virtual webcam via `pyvirtualcam`, spawns the chosen face‑swap process and a Whisper transcription thread, and configures OBS via its [WebSocket API](https://github.com/obsproject/obs-websocket) to load a prepared scene with your webcam and transcription overlay.  (The default OBS installation does not enable the WebSocket server; you must install the [obs-websocket plugin](https://github.com/obsproject/obs-websocket) and enable it under *Tools → WebSockets Server Settings*.)

To start the dashboard:

```cmd
conda activate faceswap
cd python
python pipeline_gui.py
```

Open your browser to `http://localhost:7860` to access the dashboard.  Select your options and click **Start Stream**.  The script will launch the appropriate processes and connect them to OBS.

### 4.2 Configure OBS Scene

In OBS create a new scene containing:

1. **Video Capture Device:** your physical webcam.
2. **Display Capture** or **Window Capture:** to display the output of `pyvirtualcam` if desired.
3. **Audio Input Capture:** your microphone.  Whisper reads from this input and sends captions to the dashboard overlay.

You can add a **Browser Source** pointing to `http://localhost:7860/captions` if the dashboard exposes live captions.  Arrange the sources as desired, then start **Virtual Camera** and your streaming platform will see the composite.

## 5. Multi‑GPU and Scalability

If your system has multiple GPUs, you can specify which GPU to use when launching DeepFaceLive and roop by setting the `CUDA_VISIBLE_DEVICES` environment variable before starting the process.  Whisper can be run on a separate GPU by calling `torch.cuda.set_device(index)` at the start of the script.  See the code comments in `pipeline_gui.py` for details.

## 6. NSFW Toggle

The dashboard includes an **NSFW** switch.  When disabled, the application loads only safe‑for‑work models.  When enabled, roop‑cam will download and use uncensored models; please use this responsibly.

## 7. Additional Resources & Add‑Ons

The open‑source deepfake ecosystem evolves quickly.  Several related projects provide complementary functionality or enhancements that you can integrate into your workflow:

| Project | Notable features | Installation hints |
| ------- | ---------------- | ------------------ |
| **VisoMaster** | Powerful face‑swapping and editing tool that supports multiple swapper models, multi‑face masking, expression restoration and real‑time preview【776445749108926†L290-L323】.  It works with DeepFaceLab trained models and offers tensorRT acceleration【776445749108926†L316-L324】. | Windows users can download the automatic installer and run `Start_Portable.bat`【776445749108926†L326-L335】.  Manual installation requires cloning the repo, creating a conda environment, installing CUDA 12.4 and cuDNN, downloading models and running `Start.bat`【776445749108926†L350-L390】. |
| **iRoopDeepFaceCam** | Fork of roop with **multi‑face tracking**: it can track and swap 1–10 faces and includes reset controls【81463164334664†L553-L597】【81463164334664†L611-L617】.  Supplied batch files run CPU or GPU versions, and it supports GFPGAN and other enhancements. | Clone the repository, download the required GFPGAN and inswapper models【81463164334664†L868-L873】, create a virtual environment, install dependencies via `pip install -r requirements.txt`【81463164334664†L876-L904】 and run `python run.py --execution-provider cuda` or use the provided `run-cuda.bat`【81463164334664†L911-L961】. |
| **Wunjo Community Edition** | All‑in‑one creative suite that adds **lip sync**, object/text/background removal, restyling, audio separation, voice cloning and even video generation【386630943437030†L315-L343】【386630943437030†L364-L377】.  Offers portrait animation, highlights extraction and a multilingual interface【386630943437030†L315-L344】【386630943437030†L340-L349】. | Requires Python 3.10 and ffmpeg【386630943437030†L446-L448】.  Official installers for Windows/Ubuntu and Docker images are provided via the project’s website; see the “Launch Project from GitHub” section in their wiki【386630943437030†L446-L452】. |

These projects can replace or augment parts of the pipeline.  For example, VisoMaster provides advanced masking and editing tools, iRoopDeepFaceCam offers robust multi‑face tracking, and Wunjo CE adds lip‑sync and voice‑cloning features.  Review their documentation and ensure your hardware meets each project’s requirements before integration.

### Downloading the add‑ons automatically

This package includes a helper script, `download_addons.py`, which will fetch the repositories above into an `addons` folder for you (shallow clones).  To run it:

```cmd
conda activate faceswap
python download_addons.py
```

Make sure Git is installed (see the system‑dependency section).  The script will clone each project under `addons/` so you don’t have to copy them manually.  Note that the repositories are large and downloading them will take some time.

---

### Disclaimer

Face‑swapping and deepfake technologies raise ethical and legal considerations.  This project is provided **for educational purposes only**.  Do not use it to impersonate, harass or deceive others.  Always comply with applicable laws and respect individuals’ privacy and consent.