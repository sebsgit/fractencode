# Fractal image encoder

A simple (for now) implementation of fractal image encoding.

# How does it work ?

For given image, we try to find it's self-similarities and estimate the best transform set that generates the image when applied in a feedback loop (input for the next step is the result of previous one).

Current implementation is a simple proof of concept.
Code in this repository is based on white papers by Y.Fisher, S.Menlove and others.
More information and theory here : https://www.bookdepository.com/Fractal-Image-Encoding-Analysis-Yuval-Fisher/9783642083242
