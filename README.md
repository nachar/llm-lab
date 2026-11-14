# llm-lab

# Python Environment Setup

Instructions to create and configure the Python virtual environment.

## 1. Create the virtual environment

```bash
python3 -m venv .venv
```

## 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Notes

- Run these commands from the project root directory.
- The `.venv` directory contains the project's isolated Python environment.
- Dependencies and their versions are defined in `requirements.txt`.
- To deactivate the environment, run:

```bash
deactivate
```