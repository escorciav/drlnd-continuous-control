[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

## Introduction

For this project, we will use reinforcement learning to program a [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)!

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will work with a Unity environment that contains 20 identical agents, each with its own copy of the environment.

This version is useful to test out algorithms that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience. In particular, we will review a simplified version of [D4PG](https://openreview.net/pdf?id=SyZipzbCb).

#### Our challenge

The barrier for solving the environment is slightly different is to achieve an average score of +30, over 100 consecutive episodes, and over all agents.  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores.
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **Reacher _Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    _Note_ unfortunately, I could only test the project in Linux 64-bit.

2. Place the file in the root folder of this project, and unzip (or decompress) the file.

3. Install all the required dependencies:

    The main requirements of this project are Python==3.6, numpy, matplotlib, jupyter, pytorch and unity-agents. To ease its installation, I recommend the following procedure:

    - [Install miniconda](https://conda.io/docs/user-guide/install/index.html).

      > Feel free to skip this step, if you already have anaconda or miniconda installed in your machine.

      > For Linux 64-bit users, I would recommend trying the step outlined [here](#installation-for-conda-and-linux-x86-64-users)

    - Creating the environment.

      `conda create -n drlnd-continuous-control python=3.6`

    - Activate the environment

      `conda activate drlnd-continuous-control`

    - Installing dependencies.

      `pip install -r requirements.txt`
      

### Installation for conda and Linux x86-64 users

You can use the environment [YAML file](environment_linux_x86-64.yml) provided with repo as follows:

`conda env create -f environment_linux_x86-64.yml`

## Instructions

Launch a jupyter notebook and follow the tutorial in [Continuous_Control.ipynb](Continuous_Control.ipynb) to train your own agent!

> In case you close the shell running the jupyter server, don't forget to activate the environment. `conda activate drlnd-navigation`

## Do you like the project?

Please gimme a ⭐️ in the GitHub banner 😉. I am also open for discussions especially accompany with ☕ or 🍺.
