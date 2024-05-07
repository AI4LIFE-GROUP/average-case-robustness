# Efficient Estimators for Average-Case Robustness

Code repository for the paper 
["Characterizing Data Point Vulnerability via Average-Case Robustness (UAI 2024) by Tessa Han*, Suraj Srinivas*, and Hima Lakkaraju"](https://arxiv.org/abs/2307.13885).


## Summary

Average-case robustness ($p^\mathrm{robust}\_{\sigma}$) is a useful measure of data point vulnerability. However, the naïve approach to compute $p^\mathrm{robust}\_{\sigma}$ (i.e., $p^\mathrm{mc}\_{\sigma}$), which relies on Monte Carlo sampling, is computationally intractable. Thus, in this paper, we develop efficient analytical estimators of average-case robustness ($p^\mathrm{taylor}\_{\sigma}, p^\mathrm{mmse}\_{\sigma}, p^\mathrm{taylor\_mvs}\_{\sigma}$ and $p^\mathrm{mmse\_mvs}\_{\sigma}$). We demonstrate the accuracy and efficiency of these analytical estimators for standard deep learning models and their usefulness for identifying vulnerable data points and quantifying robustness bias of models. 



## Demo

The $p^\mathrm{robust}_{\sigma}$ estimators are implemented in the `estimators` folder. A demonstration of how to use each estimator can be found in `calc_p_robust.py`. To run `calc_p_robust.py`:

* Navigate into the  `average-case-robustness` repo
* Run `$ python calc_p_robust.py`

This will print the following output (numbers may vary):
```
--> load data
Files already downloaded and verified
--> load model
--> calculate p_robust
p_mc: [0.78, 0.88, 0.3, 0.36, 1.0, 1.0, 0.98, 1.0]
p_tay: [1.0, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
p_tay_mvs: [1.0, 0.92, 0.92, 0.94, 1.0, 0.97, 0.88, 0.94]
p_mmse: [0.96, 0.79, 0.13, 0.12, 1.0, 1.0, 1.0, 1.0]
p_mmse_mvs: [0.84, 0.62, 0.2, 0.18, 1.0, 1.0, 0.95, 0.99]
p_softmax: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
complete!
```


## Citation

```
@inproceedings{averagecaserob2024,
    title={Characterizing Data Point Vulnerability via Average Case Robustness},
    author={Han*, Tessa, and Srinivas*, Suraj, and Lakkaraju, Himabindu},
    booktitle={Conference on Uncertainty in Artificial Intelligence (UAI)},
    year={2024}
}
```
