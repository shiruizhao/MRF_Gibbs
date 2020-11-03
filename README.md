#Examples
## Change Point Model
This model was developed to learn the MCMC technologies. Gibbs sampler and HMC have been learned.
Three different models were implements. Two of them used Gibbs sampling based on Scipy and Pyro, separately, The rest one was based on No-U-Turn Sampling (HMC).

![Change point model](example/output/CPM.png)
## Color Image Segementation
[code](Image denoising using Gibbs sampling and Ising model) developed EM Algorithm for image segmentation.
```
cd example/mrf_example
python image_segmentation.py
```
![zebra](example/data/zebra.jpg)
![zebra mask](example/output/zebra_masked.jpg)
![zebra foregroud](example/output/zebra_foreground.jpg)
![zebra background](example/output/zebra_background.jpg)
## Image Denoising
[code](Image denoising using Gibbs sampling and Ising model) developed Gibbs sampling and Ising model for image denoising.
The denoise function has bugs.
```
cd example/mrf_example
python image_denoising.py
```
![Noise Pic](example/data/2_noise.png)
![Denoise Pic](output/avg_denoise.jpg)
# Hyperspectual Image Segemenation


