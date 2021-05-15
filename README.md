<summary><h2 style="display: inline-block">Table of Contents</h2></summary>
<ol>
    <li>
        <a href="#about-the-project">About</a>
    </li>
    <li>
        <a href="#getting-started">Getting started</a>
    </li>
    <li>
        <a href="#acknowledgements">Acknowledgements</a>
    </li>
</ol>


## About

Simple app that predicts a given image drawn by the user from number of classes. Inspired by Google's QuickDraw.

Check out the demo: https://teo03.github.io/PicPredict/ (best viewable on desktop)

Currently the model is ~89MB and takes a while to download. 

### Getting started

You can download and use the project with more classes and a larger dataset.

1. First, make sure you have PyTorch installed

2. Clone the repo
   ```sh
   git clone https://github.com/Teo03/PicPredict.git
   ```
3. Edit `config.py` and train using the `model.py` notebook
4. Test your predictions with `predict.py` and get a usable model with `export_to_onnx.py`

#### Acknowledgements

* https://quickdraw.withgoogle.com/data
* https://github.com/XJay18/QuickDraw-pytorch
* https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js