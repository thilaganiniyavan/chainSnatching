"""Two-stage snatch detection: role-free contact candidates (stage 1) checked by a learned video verifier (stage 2).

Stage 1 (`candidates`) needs only tracked boxes. Stage 2 (`model`, `live`) needs torch and torchvision; the
training command (`train`) also needs scikit-learn (`pip install -e ".[learned]"`).
"""
