# From ML Theory to Practice - Exercise Repository

This is the exercise repository for the course "From ML Theory to Practice", being taught at the Universität Potsdam in the fall semester of 2025.

We will use this repository to post the exercises and receive your solutions.

We assume you have a basic understanding of git. If you don't, look at this [quick guide](https://rogerdudler.github.io/git-guide/) and ask us for help!

## Setup

1. **Open your terminal**

2. **Clone this repository:**
   ```bash
   git clone git@github.com:uni-potsdam/ml-theory-to-practice.git
   ```

3. **Enter the repository and create a new branch for yourself using your university username:**
   ```bash
   cd ml-theory-to-practice
   git checkout -b student/<your-potsdam-username>
   ```
   For example, if your username is `jdoe@uni-potsdam.de`, use:
   ```bash
   git checkout -b student/jdoe
   ```

4. **Create a virtual environment to hold the Python dependencies for the course:**
   
   First, make sure you have virtualenv installed:
   ```bash
   pip install virtualenv
   ```
   
   Create a new virtual environment:
   ```bash
   python -m virtualenv venv
   ```

5. **Install the basic dependencies:**
   
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

1. For each exercise, we will post the corresponding notebook in the schedule below. You can then pull the repository to get the notebook and other files:
   ```bash
   git pull origin main
   ```

2. If you need to install additional Python packages, we will tell you.

3. You can begin working inside your `student/<username>` branch, committing your changes as you go:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

4. To submit your exercise, create a final commit with message "SUBMISSION: [exercise name]". Make sure to push your changes:
   ```bash
   git commit -m "SUBMISSION: Exercise 1"
   git push origin student/<username>
   ```
   **Important:** If you don't push your changes, we won't see your submission!

## Schedule

| Project | Posted | Due Date | Page |
|---------|--------|----------|------|
| Exercise 1: Linear Regression | TBD | TBD | TBD |
| Exercise 2: Classification | TBD | TBD | TBD |
| Exercise 3: Neural Networks | TBD | TBD | TBD |
| Exercise 4: Deep Learning | TBD | TBD | TBD |
| Exercise 5: Final Project | TBD | TBD | TBD |

## License

The exercises are shared under a [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/). This means you are free to share, copy, distribute, and adapt the material, including for commercial purposes, as long as you give appropriate credit to the original creators (Juan L. Gamella and Simon Bing).
