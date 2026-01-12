# Minimalist example of uncertainty quantification with built-in standard deviation
This document demonstrates a minimalist example of how to write a functional CHAP-compatible forecasting model that generates probabilistic predictions using models with built-in uncertainty estimates. The example extends the lagged features tutorial by using a model that provides standard deviation estimates, enabling the generation of multiple samples from the predictive distribution.  

## Setting Up the Environment
Before running this example, you need to have Python installed on your system.

We recommend that you create a virtual environment to isolate the dependencies for this project. This prevents conflicts with other Python projects and keeps your system clean. You can do this using the built-in `venv` module in Python (explained beneath) or by using the tool `uv`. If you are new to virtual environments, you can check out our [guide on virtual environments](https://chap.dhis2.org/tech-intro/virtual-environments/).

If you are on Windows, we assume you are using WSL ([see our terminal setup guide here](https://chap.dhis2.org/tech-intro/terminal/)).

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

This will install the following packages:
- `numpy` - numerical computing
- `pandas` - data manipulation
- `scikit-learn` - machine learning (includes BayesianRidge)
- `joblib` - model serialization

## Running the model without CHAP integration
The example can be run in isolation (e.g. from the command line) using the file isolated_run.py:
```
python isolated_run.py
```

The output data (predictions with uncertainty samples) will be stored in the `./output/` directory.

For details on code files and data, please consult the "minimalist_example_lag" tutorial. The key differences are:

* The train function (in "train.py") uses `BayesianRidge` regression instead of standard linear regression. This model provides built-in uncertainty estimates through its `return_std` parameter:
```python
    model = BayesianRidge()
    X_train = X.to_numpy(dtype=np.float64, copy=True)
    Y_train = Y.to_numpy(dtype=np.float64, copy=True)
    model.fit(X_train, Y_train)
```
Note: The conversion to numpy arrays with explicit `dtype=np.float64` ensures compatibility with numpy 2.x.

* The `predict` function (in `predict.py`) leverages the model's built-in standard deviation to generate multiple probabilistic samples. Instead of a single point prediction, we generate 100 samples from the predictive distribution:
```python
    y_one_pred, std = model.predict(X_input, return_std=True)
    samples = np.random.normal(y_one_pred, std, size=number_of_samples)
```
These samples are stored in columns `sample_0`, `sample_1`, ..., `sample_99`, representing different possible realizations of the forecast given the model's uncertainty.

* The autoregressive prediction loop now incorporates uncertainty at each step:
```python
    prev_disease = historic_df['disease_cases'].iloc[-1]
    for i in range(X.shape[0]):
        X.loc[i, last_disease_col] = prev_disease
        X_input = X.iloc[i:i+1].to_numpy(dtype=np.float64, copy=True)
        y_one_pred, std = model.predict(X_input, return_std=True)
        samples = np.random.normal(y_one_pred, std, size=number_of_samples)
        samples_array[i, :] = samples
        prev_disease = y_one_pred
```

### Understanding uncertainty in predictions

The standard deviation (`std`) returned by BayesianRidge quantifies how uncertain the model is about its prediction. A larger standard deviation means the model is less confident. By sampling from a normal distribution centered on the prediction with this standard deviation, we generate multiple plausible outcomes that reflect this uncertainty. This is essential for probabilistic forecasting, where we want to communicate not just a single "best guess" but a range of possible futures.

## Running the minimalist model as part of CHAP
To run the model in CHAP, we define the model interface in an MLFlow-based yaml specification as we did in the previous examples (in the file "MLproject", which defines :

```yaml
name: min_py_ex

entry_points:
  train:
    parameters:
      train_data: path
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: path
      future_data: path
      model: str
      out_file: path
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"


```

After you have installed chap-core (see here for installation instructions: https://github.com/dhis2-chap/chap-core), you can run this model through CHAP as follows (remember to replace '/path/to/your/model/directory' with your local path):
```
chap evaluate --model-name /path/to/your/model/directory --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf --ignore-environment  --debug
```

