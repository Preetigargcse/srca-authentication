# Source images

All patches in `patches/` are crops of test images distributed with scikit-image
(`skimage.data`, version 0.26). Licence statements are quoted from the scikit-image
documentation of each image.

| Source | Domain | Patches | Licence / rights statement |
| --- | --- | --- | --- |
| camera | natural | 40 | CC0 (photographer Lav Varshney) |
| astronaut | natural | 40 | Public domain (NASA) |
| coffee | natural | 40 | CC0 (photographer Rachel Michetti) |
| chelsea | natural | 40 | CC0 (photographer Stefan van der Walt) |
| coins | natural | 40 | No known copyright restrictions (Brooklyn Museum) |
| moon | natural | 40 | No statement in the scikit-image docs; distributed with scikit-image (BSD-3-Clause) |
| hubble_deep_field | natural | 40 | Public domain (NASA) |
| rocket | natural | 40 | Public domain (SpaceX) |
| brick | natural | 40 | CC0 (CC0Textures) |
| grass | natural | 40 | CC0 (DeviantArt, linolafett) |
| gravel | natural | 40 | CC0 (CC0Textures) |
| clock | natural | 40 | Public domain (photographer Stefan van der Walt) |
| colorwheel | natural | 40 | No statement in the scikit-image docs; distributed with scikit-image (BSD-3-Clause) |
| retina | biomedical | 60 | CC0 (Wikimedia Commons, M. Häggström) |
| immunohistochemistry | biomedical | 60 | No known copyright restrictions (CMMI) |
| cell | biomedical | 60 | CC0 |
| shepp_logan_phantom | biomedical | 60 | Synthetic phantom; noise (σ = 2) added in this work |

Preprocessing: RGB images converted to grayscale; retina downsampled by 2; phantom upscaled
by 1.3 with Gaussian noise; random 256 × 256 crop, random 90° rotation and horizontal flip.
