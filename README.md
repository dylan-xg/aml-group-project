# Applied Machine Learning Group Project

Facial Recognition with Emotion & Liveness

## Setup

### Python Version

Make sure your python version is >=3.11 and <=3.13.

If you are using Linux, you can use the pyenv package to enable the correct version for this folder. You will need to run `pyenv install 3.13`.

Windows users can figure it out for themselves.

### Automatic Setup

If you have **Make** and know how to use it, you can simply run `make` and it will set everything up for you.

When the packages listed in [requirements.txt](requirements.txt) have been updated, you can run `make install` to easily install them.

### Manual Setup

If you prefer conda you can do it that way.

#### 1. Virtual Environment

- Run `python -m venv .venv`

#### 2. Activate The Environment:

- Linux: `source .venv/bin/activate`
- Windows: `.\.venv\Scripts\Activate.ps1`

#### 3. Install The Requirements

- Run `pip install -r requirements.txt --upgrade pip`

### Additional

- (If using notebook) Select the virtual environment as the kernel.

- Create a `.env` file. Add the values required by [`settings.py`](src/panopticon/settings.py) to it
