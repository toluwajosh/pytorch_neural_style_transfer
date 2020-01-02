# Notes for Neural Style Transfer with Pytorch

## Credits

This repo was created using the code provided by the official tutorial by pytorch as the base source [[1]](#1), but with many modifications. However, it is necessary to mention this for the sake of attribution and giving the due credits.

Additionally, credits to the original Neural Style Algorithm by Gatys et. al [[2]](#2). Most of the pipeline in this repo is based on the original algorithm.

See a full list of [references below](#references).

## Introduction

Neural-Style, or Neural-Transfer, allows you to take an image and reproduce it with a new artistic style. The algorithm takes three images, an input image, a content-image, and a style-image, and changes the input to resemble the content of the content-image and the artistic style of the style-image[[1]](#1).

## References

<a id="1">[1]</a>  [Neural Transfer Using PyTorch, https://pytorch.org/tutorials/advanced/neural_style_tutorial.html](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). Retrieved on 3rd January, 2020.

<a id="2">[2]</a> [Gatys et, al (2015) Deep Double Descent: A Neural Algorithm of Artistic Style. https://arxiv.org/abs/1508.06576](https://arxiv.org/abs/1508.06576)