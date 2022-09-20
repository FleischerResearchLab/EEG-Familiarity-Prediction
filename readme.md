# Welcome to EEG Familiarity Prediction Repository

This research project is mainly concerned with:

1. Replicate result from this previous paper to build necessary toolset to perform additional research https://pages.ucsd.edu/~desa/EEG_Reveals_Familiarity_2021.pdf
	- [Simulation of Accuracy Based on the Reported Value](https://github.com/FleischerResearchLab/EEG-Familiarity-Prediction/blob/3b08c1a5dd28305b15cd60372e5f43449fdabbb4/AccuracySimulation-Analysis.ipynb)
2. Apply new dataset from this paper to corrobate the findings in the first paper.

---
## File Structures

- `modified-scripts`
	- contains the modified MatLab scripts to facilitate our experiment of strictly enforcing zero shrinkage and only measuring the training set error.
	- `cal_shrinkage.m`
		- Enforces zero shrinkage
	- `check_lda_train_reg_auto.m`
		- Modified bias calculation
	- `lda_apply_prob.m`
		- Modified prediction generation processes
- `Reports.ipynb`
	- A batch report that summarized the current progress of this study.
- `mat_preproc.py`
	- A preprocessing package dedicates to structuring the multi-classes raw data into a trainable dataset in a binary classification setting.
- `LDA_simplest_test.m`
	- A simplified MatLab script directly called the `modified-scripts`. Design to be directly callable from python. See `Reports.ipynb` of how it works.