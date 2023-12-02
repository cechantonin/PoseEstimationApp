# Installation
1. Using Anaconda or similar software create a Python environment with Python 3.10 and pip.
2. Install dependencies for PyQt5
   ``` bash
   pip install packaging protobuf numpy qimage2ndarray
   ```
3. Install PyQt5
   ``` bash
   pip install pyqt5
   ```
4. Next, install PyTorch. This will download a ~2 GB file.
   ``` bash
   pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117
   ```
5. Install MIM for installation of the OpenMMLab projects.
   ``` bash
   pip install openmim==0.3.3
   ```
6. Install MMCV
   ``` bash
   mim install mmcv-full==1.7.0
   ```
7. Due to mixing different versions of OpenMMLab projects, the dependencies for MMPose and MMTrack muset be installed manually beforehand
   ``` bash
   pip install xtcocotools chumpy json-tricks munkres motmetrics opencv-python
   
   pip install attributee dotty-dict lap mmcls seaborn tqdm
   ```
9. Finally install the main three packages
   ``` bash
   mim install mmdet==2.28.2

   mim install mmtrack==0.14.0 --no-dependencies

   mim install mmpose==0.29.0 --no-dependencies
   ``` 