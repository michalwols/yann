# Introduction

Yann is a batteries included deep learning framework built in Pytorch.

Inspired by Django and Rails, it aims to automate the tedious steps of a machine learning project 
so that you can focus on the (fun) hard parts. It makes it easy to quickly get a project started 
but also scales with you all the way to production.
 
It could also be viewed as `torch.nn` extended, as it includes common new research modules 
that might be too experimental to be included in torch.



## Getting Started

### Install 

```commandline
pip install yann
```


### Create a Project

```commandline
yann scaffold digit-recognition
```




## Yann Command Line Interface

yann comes with a command line interface that makes it even easier to get started

* `yann run train` - 
* `yann train` - Create a new project.
* `yann serve` - Serve a pretrained model through an API or web UI.
* `yann scaffold` - Print this help message.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
