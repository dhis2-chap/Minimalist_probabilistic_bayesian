# Minimalist example of model integration with lagged features 
This document demonstrates a minimalist example of how to write a functional CHAP-compatible forecasting model with lagged features. The example is based on the same simplistic regression as the "minimalist_multiregion" model, but also demonstrates how to include lagged features in your model.  

## Running the model without CHAP integration
The example can be run in isolation (e.g. from the command line) using the file isolated_run.py:
```
python isolated_run.py  
```

For details on code files and data, please consult the "minimalist_multiregion" model. The only differences are that:

* The train function (in "train.py") now adds lagged features to the covariate matrix using the "create_lagged_feature" utility function found in "utils.py": 
```csv
    create_lagged_feature(X, 'mean_temperature', 1)
    create_lagged_feature(X, 'rainfall', 1)
    create_lagged_feature(X, 'disease_cases', 1, df)
```
The first row of `mean_temperature_lag_1` and `rainfall_lag_1` is missing because these lagged features require the previous day’s values, which don’t exist for the very first day in the dataset. Therefore we remove the first row from `X`. Correspondingly, we also remove the first row from the target disease_case values `Y` so that inputs and targets are aligned.
```csv
    X = cut_top_rows(X, 1)
    Y = cut_top_rows(Y, 1)
```

* The `predict` function (in `predict.py`) must create the same lagged features as used during training. Here, however, we have the lagged feature values for the first day in `historic_data_fn`, so we use them to fill in the missing values for the lagged features using the `fill_top_rows_from_historic_last_rows` utility function:
```csv
    fill_top_rows_from_historic_last_rows('mean_temperature', 1, X, historic_df)
        fill_top_rows_from_historic_last_rows('rainfall', 1, X, historic_df)
```

Since each prediction now depends on the previous day’s disease cases, we implement an autoregressive loop that iteratively generates predictions for as many time steps as specified by `future_climate_data`:
```csv
    prev_disease = historic_df['disease_cases'].iloc[-1]
        for i in range(X.shape[0]):
            X.loc[i,last_disease_col] = prev_disease
            y_one_pred = model.predict(X.iloc[i:i+1])
            df.loc[i,'sample_0'] = y_one_pred

            prev_disease = y_one_pred
```

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

