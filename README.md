# Seam-Carving-in-Videos
Generating a saliency map for each frame of a video that can maintain visual consistency across frames when seam carving is performed.

### Seam Carving in Images ###
The seam carving algorithm, proposed by [Avidan et al.](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html), deals with content-aware image resizing in two steps: it first generates an energy map with an energy function (e.g. magnitude of gradient), then a horizontal or vertical seam, an 8-connected path starting from one of the edges, is calculated dynamically along the lowest cumulative energy. by removing the calculated seams repeatedly, an image can be shrunk, ideally retaining the salient regions without introducing noticeable artefacts.

### Video Resizing ###
Extending this method to video sequences has been a struggle: consistency across frames is difficult to retain, and taking multiple image frames into account greatly increases computational complexity. [Improved Seam Carving for Video Retargeting](http://www.eng.tau.ac.il/~avidan/papers/vidret.pdf) attempts to do this using graph cuts, which is computationally expensive. 

The solution: generating saliency map for each frame of a video that allows for seam carving using dynamic programming while maintaining temporal coherency. The project provides a method for generating saliency maps for each frame of a video that incorporates temporal information. It uses deep learning convolutional neural networks to map these video frames to the corresponding saliency maps generated using the method mentioned previously. 

### Dataset ###
In order to train  network to generate saliency maps that incorporate temporal information regarding the entire video, our training target needs to include the following information:
- Energy of the current frame
- Segmentation of foreground objects in the current frame
- Energy and segmentation of all frames after the current frame

For the segmentation of foreground objects, [DAVIS 2017 Video Segmentation Dataset](https://davischallenge.org/davis2017/code.html#unsupervised) was used, which consists of 90 RGB videos, each frame with a corresponding segmentation mask annotating the subject of the video. Specifically, the 480p version was used for target generation.
<p align="center">
    <img src="./images/DAVIS.png">
</p>
<p align="center">
    <em>Video frames and their corresponding segmentation masks sampled from videos (i)’tractor-sand’, (ii)’breakdance’ and (iii)’bmx-bumps’ in the DAVIS dataset.</em>
</p>

For each video, the energy E<sub>i</sub> of the i<sup>th</sup> frame F<sub>i</sub> is calculated by:
<p align="center">E<sub>i</sub>=∇σ(F<sub>i</sub>)</p>

where σ(F<sub>i</sub>) is the Sobel Operator applied to the current frame. The value of E<sub>i</sub> is normalized to (0,1). 

The segmentation S'<sub>i</sub> is calculated from annotation S<sub>i</sub> of the DAVIS 2017 dataset by
<p align="center">S'<sub>i</sub>(x,y)= 0 if S<sub>i</sub>(x,y)=0</p>
<p align="center">S'<sub>i</sub>(x,y)= 1 otherwise</p>

The saliency map M<sub>i</sub> is then calculated by
<p align="center">M<sub>i</sub>(x,y)=max⁡(E<sub>i</sub>(x,y), S'<sub>i</sub>(x,y))</p>

In order to incorporate temporal information into the target, the dense optical flow from M<sub>i</sub> to M<sub>i+1</sub> is calculated, which gives us (dx/dt,dy/dt) for each pixel in the i<sup>th</sup> frame. Starting from the last frame and propagating to the first frame, each target saliency map T<sub>i</sub> are calculated by
<p align="center">T<sub>i</sub>(x,y)=max⁡(M<sub>i+1</sub>(x+dx,y+dy),M<sub>i</sub>(x,y))</p>

<p align="center">
    <img src="./images/Seam-Carving.jpg">
</p>
<p align="center">
    <em>Video frames and their corresponding target saliency maps sampled from videos (i)’tractor-sand’, (ii)’breakdance’ and (iii)’bmx-bumps’ in the DAVIS dataset. Dynamic programming was used to determine 50 seams with the lowest cumulative energy (marked in red).</em>
</p>

### Neural Networks ###
The network architectures experimented with were inspired from Residual U-Net described in [Road Extraction by Deep Residual U-Net](https://arxiv.org/abs/1711.10684). The implementation for Residual U-Net (without LSTM) was adapted from [this repository](https://github.com/DuFanXin/deep_residual_unet).

<p align="center">
    <img src="./images/Network1.png">
</p>
<p align="center">
    <em>Residual U-Net</em>
</p>

<p align="center">
    <img src="./images/Network2.png">
</p>
<p align="center">
    <em>Residual U-Net with LSTM</em>
</p>

### Results ###
Due to hardware limitations, the video frames had to be resized to 56x32 and sliced into segments of 10 frames each. The netork has been trained on such segments. The results for a randomly sampled sequence of 10 frames from (i)'tractor-sand', used for training, and (ii)'breakdance' and (iii)'bmx-bumps', used only for validation is shown below.
<p align="center">
    <img src="./images/Generated_Maps.png">
</p>




