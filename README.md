# From ML Theory to Practice - Uni Potsdam 2025

[![License: CC-BY 4.0](https://img.shields.io/static/v1.svg?logo=creativecommons&logoColor=white&label=License&message=CC-BY%204.0&color=yellow)](https://creativecommons.org/licenses/by/4.0/)

![The Causal Chambers: (left) the wind tunnel, and (right) the light tunnel with the front panel removed to show its interior.](https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/the_chambers.jpg)

This is the exercise repository for the course "From ML Theory to Practice", imparted at the Universität Potsdam in the fall semester of 2025. The course has been designed by Juan L. Gamella and Simon Bing.

We will use this repository to post the exercises and receive your solutions. To do this, please see [setup](#setup) and [workflow](#workflow) below.

We assume you have a basic understanding of git. If you don't, look at this [quick guide](https://rogerdudler.github.io/git-guide/) and ask us for help!

## Schedule

We will update the table as the course progresses. Click on a project's name to see the notebook and any additional instructions.

| Project                                                                   | Posted     | Due Date            |
|---------------------------------------------------------------------------|:----------:|:-------------------:|
| [Project 1.1: Understanding Linear Models on Synthetic Data](project_11/) | 20.10.2025 | 27.10.2025 at 12:00 |
| [Intermezzo: Collecting data from a Causal Chamber™](intermezzo/)         | 27.10.2025 | NA                  |
| [Project 1.2: Linear models and real-world data](project_12/)             | 27.10.2025 | 3.11.2025  at 12:00 |
| Project 2: Causality, RCTs and two-sample testing                         | tbd        | tbd                 |
| Project 3: Conformal Prediction                                           | tbd        | tbd                 |
| Project 4: Generative Modelling                                           | tbd        | tbd                 |
| Project 5: Controlling a real system: RL and Sim2Real                     | tbd        | tbd                 |


## Setup

Before working on the notebooks, you will need to set up your git and Python environments.

1. **Clone this repository**

    Open your terminal, and type
   ```bash
   git clone git@github.com:causalchamber/potsdam-25-ml-theory-to-practice
   ```

3. **Create a virtual environment to hold the Python dependencies for the course**

   First, make sure you have virtualenv installed:
   ```bash
   pip install virtualenv
   ```
   
   Create a new virtual environment:
   ```bash
   cd potsdam-25-ml-theory-to-practice
   python -m virtualenv venv
   ```

4. **Install the basic dependencies**
   
   Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
   
   Install the requirements:
   ```bash
   pip install -r core_requirements.txt
   ```
   
   To make sure this worked, run:
   ```bash
   jupyter lab
   ```
   This should open Jupyter Lab in your browser at [localhost:8888/lab](http://localhost:8888/lab).

## Workflow

When a new exercise is available, we will post it to the Schedule table above. You can then work on each exercise as follows:

1. Pull the repository to get the notebook and other files:
   ```bash
   git pull origin main
   ```

2. If you need to install additional Python packages for a particular project, we will tell you.

3. Create a new branch for yourself using your university username

    For example, if your username is `jdoe@uni-potsdam.de`, use:
   ```bash
   git checkout -b student/jdoe
   ```

4. You can now begin working inside your `student/<username>` branch, committing your changes as you go:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

5. To submit your exercise, create a final commit with message "SUBMISSION: project xxx". Make sure to push your changes:
   ```bash
   git commit -m "SUBMISSION: Project X.Y"
   git push origin student/<username>
   ```
   If you add additional changes, just add them and redo the commit. We will just look at your last commit _before_ the deadline.
   
   **Important:** If you don't push your changes, we won't see your submission!

## FAQ

We will populate this section as the course progresses. If you have any questions, please reach out to Simon Bing.


## License

The exercises are shared under a [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/). This means you are free to share, copy, distribute, and adapt the material, including for commercial purposes, as long as you give appropriate credit to the original creators: Juan L. Gamella and Simon Bing.
