# Colab Training — Steps

Goal: train on a Colab GPU and push `weights.joblib` back to GitHub.
**Don't edit `model.py`** (weights are tied to it). Run each step in a Colab cell.

1. **New Colab notebook** → Runtime → Change runtime type → **GPU**.

2. **Clone:**
   ```python
   !git clone https://github.com/DanielBenShabat/Robust-Image-Classification.git
   %cd Robust-Image-Classification
   ```

3. **Get the dataset** (zip with `train/` + `augmentations/` on your Drive):
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   !unzip -q "/content/drive/MyDrive/dataset.zip" -d dataset/
   !ls dataset            # must show: train  augmentations
   ```

4. **Build the split:**
   ```python
   !python submissions/my_team/preprocessing.py
   ```

5. **Train** (GPU auto-detected, ~1h):
   ```python
   !python submissions/my_team/train.py --workers 2
   ```

6. **Check it passes:**
   ```python
   !python check_submission.py my_team
   !python submissions/my_team/eval_robustness.py
   ```

7. **Push weights back** (use a GitHub token; weights are gitignored so force-add):
   ```python
   import getpass; token = getpass.getpass('GitHub token: ')
   !git config user.email "you@example.com"
   !git config user.name "Your Name"
   !git add -f submissions/my_team/weights.joblib
   !git commit -m "Trained weights from Colab"
   !git push https://{token}@github.com/DanielBenShabat/Robust-Image-Classification.git main
   ```

Notes:
- Don't run `pip install -r requirements.txt` (it's CPU torch; Colab already has GPU torch).
- Needs collaborator access + a token with **Contents: Read/write**.
- Then Daniel: delete local `weights.joblib`, `git pull`, `check_submission.py`, submit.
