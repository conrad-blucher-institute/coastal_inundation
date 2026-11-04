# =====================================================================================================
# wl_functions.py: every function the water level notebook uses
# =====================================================================================================
#
# The notebook's Setup cell downloads this repo and then loads this file with:
#
#     %run -i {REPO_DIR}/code/wl_functions.py
#
# Loading it only defines functions and settings. Nothing is read or trained until the notebook asks.
#
# To change the code: edit this file, push it to GitHub, then run the Setup cell again. Setup always gets
# the newest version. Changes made to a copy inside Colab are lost when the session ends.
#
# The -i in the command matters. It runs this file as if it were pasted into the notebook, so:
#   - this file can use what the Setup cell creates: COMMIT, DATA_DIR, CONFIG_DIR, OUT_DIR, DRIVE_OK
#   - the notebook's cells can use everything defined here
#   - running Setup again doesn't start a second background saving thread (see "saving in the background")
#
# Sections, in order:
#   1. Imports
#   2. Data loading
#   3. Model evaluation functions
#   4. Functions to create input & output arrays
#   5. Config files and model settings
#   6. Report functions
#   7. Training function


# =====================================================================================================
# 1. Imports
# =====================================================================================================
from datetime import datetime

# Path Handling
import os
import json
import shutil
import hashlib
import subprocess
import tempfile
import traceback
from pathlib import Path

# Plotting
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Data manipulation
import numpy as np
import pandas as pd

# Machine Learning
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Dropout, Normalization
from keras.models import Sequential, load_model
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

# Widgets and HTML for the pick-experiment cells and the report
import ipywidgets as widgets
import html as html_lib
from IPython.display import display, HTML


# =====================================================================================================
# 2. Data loading
# =====================================================================================================
# These functions read the data files that a config lists. Nothing is loaded yet: each experiment loads its own files when it trains. A set of files that has already been loaded is reused, so experiments that share files read them only once.
#
# DATA_ROLES describes each kind of data file. If a new data source has different column names, this is the only place to change.
#
# Data from Port Lavaca / Six Mile Park (main station):
# - Water level - 6 min increments
# - Wind - 6 min increments
#
# Rain (IEM) - hourly increments, which we re-sample into 6 min
#
# Port O'Connor water level - 6 min increments
#
# Re-sampling hourly rain to 6-min data
#
# The rain file comes from the Iowa Environmental Mesonet (IEM) hourly precip tool. (NOT the Accumulator Tool, which records the data slightly differently! Use the hourly precip tool.)
#
# - Timestamps are local Central time (with daylight saving), so they're converted to UTC.
# - Each value is the rain for the hour starting at its timestamp (e.g. the 5 PM value is roughly 4:53–5:53 PM).
#
# 1. Linear interpolation between hourly values is used to give a shape and determines which 6-min bins in the hour get more rain and which get less. (So it's like a gradient!)
# 2. Each hour's 10 bins are then rescaled so they add up to that hour's measured total.
#
# The result ramps smoothly within each hour and keeps every hour total.

# Each kind of data file, recognized by a column only that kind has.
#   time_column: the column with the timestamps
#   timezone:    the timezone the timestamps are in (everything is converted to UTC)
#   columns:     the columns to keep, and the name each one gets in the notebook
DATA_ROLES = {
    'station': {'detect_column': 'wl',
                'time_column': 'DateTime',
                'timezone': 'UTC',  # Port Lavaca lines up with Port O'Connor's UTC (checked by tide timing)
                'columns': {'wl': 'wl', 'uwind': 'uwind', 'vwind': 'vwind'}},
    'poc':     {'detect_column': 'water_level_m',
                'time_column': 'time_utc',
                'timezone': 'UTC',
                'columns': {'water_level_m': 'pc_wl'}},
    'rain':    {'detect_column': 'precip_in',
                'time_column': 'DateTime',
                'timezone': 'America/Chicago',  # IEM hourly precip is local Central time (with daylight saving)
                'columns': {'precip_in': 'precip_in'}},  # turned into 'precip' (mm, 6-min) by hourly_rain_to_6min
}


def find_role(df, file_name):
    """Returns which kind of file this is ('station', 'poc' or 'rain'), based on its columns"""
    matches = []
    for role, spec in DATA_ROLES.items():
        if spec['detect_column'] in df.columns:
            matches.append(role)
    if len(matches) == 0:
        raise ValueError(f"{file_name}: can't tell what kind of data this is. Its columns are {list(df.columns)}. "
                         f"Expected one of these columns: {[spec['detect_column'] for spec in DATA_ROLES.values()]}")
    if len(matches) > 1:
        raise ValueError(f"{file_name}: looks like more than one kind of data ({matches}). Columns: {list(df.columns)}")
    return matches[0]


def hourly_rain_to_6min(s):
    """Spreads hourly rain totals (mm) over 6-min steps. Each hour's 10 steps add up to that hour's measured total.

    1. Linear interpolation between hourly values gives a 'shape': which 6-min steps in the hour get more rain.
       These are not real values yet, just weights (90/10 ... 80/20 ... 70/30 ...).
    2. Each hour's 10 steps are rescaled so they add up to that hour's total.
    """
    shape = s.resample('6min').asfreq().interpolate(method='time')

    # floor: each 6-min step belongs to the hour it falls in
    # i.e. 1:00, 1:06, ... 1:54 all belong to the 1:00 reading, so those 10 values add up to the 1:00 hourly total
    # JULIET WILL INVESTIGATE THIS :)
    hour_label = shape.index.floor('h')

    # add up the shape values of the 10 steps in each hour
    bin_sum = shape.groupby(hour_label).transform('sum')

    # each 6-min step gets the measured total of the hour it belongs to
    hourly_total = pd.Series(s.reindex(hour_label).to_numpy(), index=shape.index)

    # scale the shape so each hour's 10 steps add up to that hour's measured total (an hour with no rain stays 0)
    fine = (shape * hourly_total / bin_sum).where(bin_sum != 0, 0)

    # drop the last hour; only 1 of its 10 steps exists
    fine = fine[fine.index < s.index[-1]]
    return fine


def read_data_file(path):
    """Reads one data file into a DataFrame with a UTC DateTime index and the notebook's column names"""
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    role = find_role(df, path.name)
    spec = DATA_ROLES[role]

    needed = [spec['time_column']] + list(spec['columns'])
    missing = []
    for column in needed:
        if column not in df.columns:
            missing.append(column)
    if missing:
        raise ValueError(f"{path.name} looks like '{role}' data but has no column(s) {missing}. Columns: {list(df.columns)}")

    # Timestamps -> UTC
    timestamps = pd.to_datetime(df[spec['time_column']], format='mixed')
    if spec['timezone'] == 'UTC':
        timestamps = timestamps.dt.tz_localize('UTC')
    else:
        # ambiguous/nonexistent = NaT: drops the repeated 1 AM hour in November (spring's 2 AM is already missing)
        timestamps = timestamps.dt.tz_localize(spec['timezone'], ambiguous='NaT', nonexistent='NaT')
        timestamps = timestamps.dt.tz_convert('UTC')

    out = df[list(spec['columns'])].rename(columns=spec['columns'])
    out.index = pd.DatetimeIndex(timestamps, name='DateTime')
    out = out[out.index.notna()]                  # drop blank rows + the dropped DST hour
    out = out[~out.index.duplicated()]            # drop duplicate timestamps (keeps the first)
    out = out.sort_index()

    if role == 'rain':
        precip_mm = out['precip_in'] * 25.4       # inches -> mm
        out = hourly_rain_to_6min(precip_mm).to_frame('precip')
    return out


def file_fingerprint(path):
    """Short fingerprint of a file's contents. Two files with the same fingerprint have identical data."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


_loaded_files = {}  # file name -> DataFrame, so each file is only read once
_loaded_data = {}   # (file names) -> DataFrame, so each combination is only joined once


def load_file(name):
    """Reads one data file from DATA_DIR (once; later calls reuse it)"""
    if name not in _loaded_files:
        _loaded_files[name] = read_data_file(Path(DATA_DIR) / name)
    return _loaded_files[name]


def load_split(file_names):
    """Reads the listed data files from DATA_DIR and joins them side by side on time"""
    key = tuple(file_names)
    if key in _loaded_data:
        return _loaded_data[key]

    dfs = []
    columns_so_far = []
    for name in file_names:
        df = load_file(name)
        for column in df.columns:
            if column in columns_so_far:
                raise ValueError(f"Two data files both have a '{column}' column ({', '.join(file_names)}). "
                                 f"List only one file of each kind (one station, one rain, one POC).")
            columns_so_far.append(column)
        dfs.append(df)

    combined = pd.concat(dfs, axis=1)
    _loaded_data[key] = combined
    return combined


# =====================================================================================================
# 3. Model evaluation functions
# =====================================================================================================
# - Calculate model loss
# - Calculate Central Frequency, Mean Squared Error, Mean Absolute Error, and R-squared
# - Plot time series of model predictions vs observations
# - Plot model loss

def calculate_central_frequency_percentage(testing_label_array, predictions, cm):
  """Find the percentage of predictions with an absolute error of less than or equal to a given number of centimeters (cm)

  Args:
        testing_label_array (array): Testing labels

        predictions (array): Model predictions

        cm (int): Number of centimeters

  Returns:
        (float): central frequency (CF) percentage
  """
  errors = np.abs(np.ravel(testing_label_array) - np.ravel(predictions))
  return float(np.mean(errors <= cm / 100) * 100)


def evaluate_model(model, input_array, label_array, verbose = True):
  """Calculates loss, makes predictions, and calculates Central Frequency (CF),
  Mean Squared Error (MSE), Root Mean Squared Error(RMSE), Mean Absolute Error (MAE),
  Median Absolute Error, and R-squared (R2)

  Args:
        model (tf.keras.model): The trained model

        input_array (array): Inputs (this notebook only passes the validation set)

        label_array (array): Labels

        verbose (bool): print the metrics

  Returns:
        (dict, array): metrics, predictions
  """
  test_loss = model.evaluate(input_array, label_array, batch_size = len(input_array), verbose = 0)
  predictions = model.predict(input_array, batch_size = len(input_array), verbose = 0).ravel()

  # Calculate evaluation metrics
  metrics = {'loss': float(test_loss),
             'cf_15cm': calculate_central_frequency_percentage(label_array, predictions, 15),
             'cf_5cm': calculate_central_frequency_percentage(label_array, predictions, 5),
             'cf_1cm': calculate_central_frequency_percentage(label_array, predictions, 1),
             'mse': float(mean_squared_error(label_array, predictions)),
             'rmse': float(root_mean_squared_error(label_array, predictions)),
             'mae': float(mean_absolute_error(label_array, predictions)),
             'medae': float(median_absolute_error(label_array, predictions)),
             'r2': float(r2_score(label_array, predictions))}

  if not verbose:
    return metrics, predictions

  print("Loss:", metrics['loss'])
  print("Central Frequency Percentage 15cm:", metrics['cf_15cm'])
  print("Central Frequency Percentage 5cm:", metrics['cf_5cm'])
  print("Central Frequency Percentage 1cm:", metrics['cf_1cm'])
  print("Mean Squared Error:", metrics['mse'])
  print("Root Mean Squared Error:", metrics['rmse'])
  print("Mean Absolute Error:", metrics['mae'])
  print("Median Absolute Error:", metrics['medae'])
  print("R-squared:", metrics['r2'])

  return metrics, predictions


def plot_model_predictions(testing_labels, predictions, title, x_label, y_label, plot_file_name, legend_location = 'best', show = True):
  """Plots model predictions and compares to labels to evaluate the model performance

  Args:
        testing_labels (list): list of labels

        predictions (list): list of model predictions

        title (string): plot title

        x_label (string): x-axis label

        y_label (string): y-axis label

        plot_file_name (string): file name for plot. Must have appropriate image extension (eg. 'png', 'pdf', 'svg', ...)

        legend_location (string): 'best' (Axes only), 'upper right', 'upper left', 'lower left', 'lower right', 'right',
                                  'center left', 'center right', 'lower center', 'upper center', 'center'. Defaults to 'best'
  """
  fig, ax = plt.subplots(1, figsize=(25, 10))

  # Plot the observations and model predictions on the same plot
  plt.plot(testing_labels, label='Observed')
  plt.plot(predictions, label='Predicted')

  plt.title(title, fontsize=30)

  ax.set_xlabel(x_label, fontsize=22)
  ax.set_ylabel(y_label, fontsize=22)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.legend(fontsize = 20, loc = legend_location)

  plt.savefig(plot_file_name, bbox_inches='tight')

  if show:
    plt.show()

  plt.close()


def plot_model_loss(model_history, plot_file_name, legend_location = 'best', show = True):
  """Plots model training loss and validation loss on the same plot

  Args:
        model_history (keras.src.callbacks.History): model history

        plot_file_name (string): file name for plot. Must have appropriate image extension (eg. 'png', 'pdf', 'svg', ...)

        legend_location (string): 'best' (Axes only), 'upper right', 'upper left', 'lower left', 'lower right', 'right',
                                  'center left', 'center right', 'lower center', 'upper center', 'center'. Defaults to 'best'
  """
  fig, ax = plt.subplots(1, figsize=(25, 10))

  # Plot training & validation loss values
  plt.plot(model_history.history['loss'])
  plt.plot(model_history.history['val_loss'])

  lowest_val_loss = min(model_history.history['val_loss'])
  best_epoch = model_history.history['val_loss'].index(lowest_val_loss)

  ax.axvline(x = best_epoch, color='r', linestyle='--')

  plt.title('Model loss', fontsize=30)

  ax.set_xlabel('Epoch', fontsize=22)
  ax.set_ylabel('Loss', fontsize=22)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.legend(['Train', 'Validation', 'Lowest Validation Loss'], fontsize = 20, loc = legend_location)

  plt.savefig(plot_file_name, bbox_inches='tight')

  if show:
    plt.show()

  plt.close()


def plot_interactive_time_series(dfs, title, x_label, y_label, plot_file_name, mode = 'lines', show = True):
  """Plots list of pandas dataframes datetime data on the same interactive Plotly chart.
  Saves the resulting plot in the user-specified path.

  Args:
        dfs (list): list of pandas dataframes. Must contain columns: 'data', 'dash_style', 'line_color', and 'legend_label'

        title (string): plot title

        x_label (string): x-axis label

        y_label (string): y-axis label

        plot_file_name (string): file name for plot. Must have appropriate image extension ('.html')

        mode (string): Choose between 'markers', 'lines', and 'lines+markers'. Defaults to 'lines'

        show (bool): Whether or not to display the plot on the screen. If the plot contains a large amount of data it
                     is recommended to set show = False. The code may crash if you do not
  """
  fig = go.Figure()

  for df in dfs:
    fig.add_trace(go.Scatter(x = df.index, y = df['data'], mode = mode, line = dict(dash=df['dash_style'].iloc[0], color=df['line_color'].iloc[0]), name = df['legend_label'].iloc[0]))

  fig.update_layout(
      title=go.layout.Title(
                    text = title,
                    font=dict(
                        family = 'Arial',
                        size = 20,
                        color = 'black'
                    ),
                    xref = 'paper',
                    x = 0.5
      ),

      xaxis=dict(title = x_label),
      yaxis=dict(title = y_label),
      hovermode='x unified'
  )

  fig.update_traces(connectgaps=False)

  fig.write_html(plot_file_name, include_plotlyjs='cdn')

  if show is True:
    fig.show()


# =====================================================================================================
# 4. Functions to create input & output arrays
# =====================================================================================================
# Used by build_arrays in the next section.

def lag_column_name(column, lag):
    """Name of a lagged column. Lag 0 is the original column; negative = past, positive = future.
    e.g. ('wl', -240) -> 'wl_t-240', ('wl', 120) -> 'wl_t+120'"""
    if lag == 0:
        return column
    return f'{column}_t{"" if lag < 0 else "+"}{lag}'


def fill_short_gaps(s, max_steps):
    """Linearly fills gaps of at most max_steps missing values, only when there's a real value on both sides.
    Longer gaps (and gaps at the start/end) are left as NaN, so dropna still removes them."""
    if max_steps <= 0:
        return s
    missing = s.isna()
    gap_id = (missing != missing.shift()).cumsum()           # every run of NaN / non-NaN gets its own id
    gap_len = missing.groupby(gap_id).transform('sum')       # length of the NaN run each row is in (0 if not NaN)
    filled = s.interpolate(method='time', limit_area='inside')
    return filled.where(~missing | (gap_len <= max_steps))   # undo the fill for long gaps


def extract_input_output_arrays(df, job_specification, gap_fill_steps = 0):
  """
  Builds the input array, target array, and datetime index for one dataset. The original df is not modified.

  Only the columns this run uses are kept BEFORE dropping NaNs, so a column the run doesn't use
  (e.g. rain in a rain-off run) can never remove rows.

  Args:
        df (pd.DataFrame): DataFrame containing dataset. Not modified.

        job_specification (dict): Dictionary containing the job specification (see make_job_spec)

        gap_fill_steps (int): fill gaps of up to this many 6-min steps in the gap-fill columns (0 = off)

  Returns:
      dict: input array, target array, datetime index, and the ordered list of input column names
  """
  input_specifications = job_specification['input_specifications']
  target_column = job_specification['target_column']

  # 1. Keep only the columns this run uses
  base_columns = list(dict.fromkeys(spec['column'] for spec in input_specifications))
  absent = [c for c in base_columns if c not in df.columns]
  if absent:
    raise KeyError(f"Column(s) {absent} not in the data. Columns found: {list(df.columns)}")
  df = df[base_columns].copy()

  # 2. Optionally fill short gaps (water level / wind only; rain is never filled)
  for column in job_specification.get('gap_fill_columns', []):
    df[column] = fill_short_gaps(df[column], gap_fill_steps)

  # 3. Build every input column (and the target) as one float32 array, straight from the shifted columns
  #    NOTE: shift's sign is the opposite of ours: shift(240) looks 240 steps into the PAST
  df = df.astype('float32')
  input_columns, input_data = [], []
  target_data = None
  for spec in input_specifications:
    start, finish = spec['lag_range']
    for lag in range(start, finish + 1):  # include upper bound of range
      name = lag_column_name(spec['column'], lag)
      values = df[spec['column']].shift(-lag).to_numpy()
      if name == target_column:
        target_data = values
      elif name not in input_columns:
        input_columns.append(name)
        input_data.append(values)
  input_array = np.column_stack(input_data)

  # 4. Drop rows where any input or the target is missing
  #    (dates at the beginning/end of the dataset or near data gaps don't have every input)
  keep = ~np.isnan(input_array).any(axis=1) & ~np.isnan(target_data)

  return {'input_array': input_array[keep],
          'target_array': target_data[keep],
          'datetime_index': df.index[keep],
          'input_columns': input_columns}


# =====================================================================================================
# 5. Config files and model settings
# =====================================================================================================
# Reads and checks the config files, turns a config into model inputs (make_job_spec), and builds the model.

# Column names in the notebook (set by DATA_ROLES in the data loading section)
WL_COL, UWIND_COL, VWIND_COL, POC_COL = 'wl', 'uwind', 'vwind', 'pc_wl'
RAIN_COL = 'precip'
STEPS_PER_HOUR = 10  # data is in 6-minute steps

# A 6-min rain value depends on its own hour's total AND the next hour's (through the interpolation shape),
# and IEM hours end at ~:53. So at time t, rain is only fully reported up to about 2 hours before t.
RAIN_REPORT_DELAY_HRS = 2
# Perfect prog rain never goes further than this many hours after the issue time
RAIN_PERFPROG_MAX_HRS = 18

# ---- What a config file must contain ----
REQUIRED_SETTINGS = ['location', 'train_files', 'val_files', 'test_files', 'lead_hrs', 'rain', 'rain_future_hrs',
                     'poc', 'train_gap_fill', 'val_gap_fill', 'test_gap_fill', 'normalize', 'lookback_hrs', 'layers', 'neurons',
                     'dropout', 'activation', 'epochs',
                     'patience', 'min_delta', 'n_models', 'seed']
# Lists of data files (testing results only appear in the separate VAL+TEST report)
FILE_SETTINGS = ['train_files', 'val_files', 'test_files']
# These can be one value for every lead time, or one value per lead time ("0.15;0.1;0.1")
PER_LEAD_SETTINGS = ['lookback_hrs', 'layers', 'neurons', 'dropout', 'activation', 'epochs', 'patience', 'min_delta']
WHOLE_NUMBER_SETTINGS = ['rain_future_hrs', 'train_gap_fill', 'val_gap_fill', 'test_gap_fill', 'lookback_hrs', 'layers', 'neurons', 'epochs',
                         'patience', 'n_models', 'seed']
DECIMAL_SETTINGS = ['dropout', 'min_delta']
CHOICES = {'rain': ['off', 'reported', 'to_now', 'perfprog'],
           'poc': ['yes', 'no'],
           'normalize': ['yes', 'no'],
           'activation': ['relu', 'tanh', 'sigmoid']}


def split_list(text):
    """'a; b;c' -> ['a', 'b', 'c']"""
    pieces = []
    for piece in text.split(';'):
        piece = piece.strip()
        if piece != '':
            pieces.append(piece)
    return pieces


def to_number(text, whole):
    """Turns text into a number. Returns None if it isn't one (or isn't whole when whole=True)."""
    try:
        number = float(text)
    except ValueError:
        return None
    if whole:
        if not number.is_integer():
            return None
        return int(number)
    return number


def read_config_file(path):
    """Reads and checks one config file. Returns the experiment, or raises ValueError listing every problem."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip().str.lower()
    if 'setting' not in df.columns or 'value' not in df.columns:
        raise ValueError(f"needs the columns 'setting' and 'value' (found {list(df.columns)})")

    # 1. Read the settings as text
    problems = []
    raw = {}
    for _, row in df.iterrows():
        name = row['setting'].strip()
        value = row['value'].strip()
        if name == '':
            continue
        if name in raw:
            problems.append(f"'{name}' is listed twice")
        raw[name] = value

    for name in raw:
        if name not in REQUIRED_SETTINGS:
            problems.append(f"'{name}' is not a known setting")
    for name in REQUIRED_SETTINGS:
        if name not in raw:
            problems.append(f"'{name}' is missing")
        elif raw[name] == '':
            problems.append(f"'{name}' has no value")
    if problems:
        raise ValueError("; ".join(problems))

    # 2. Lead times
    leads = []
    for text in split_list(raw['lead_hrs']):
        lead = to_number(text, whole=True)
        if lead is None or lead <= 0:
            problems.append(f"lead_hrs: '{text}' is not a whole number above 0")
        elif lead in leads:
            problems.append(f"lead_hrs: {lead} is listed twice")
        else:
            leads.append(lead)
    if len(leads) == 0:
        raise ValueError("; ".join(problems + ["lead_hrs has no valid lead times"]))

    # 3. Every other setting -> a list with one value per lead time
    per_lead = {}
    for name in REQUIRED_SETTINGS:
        if name in ['location', 'lead_hrs'] or name in FILE_SETTINGS:
            continue
        texts = split_list(raw[name])
        if len(texts) != 1 and name not in PER_LEAD_SETTINGS:
            problems.append(f"{name}: needs exactly one value (found {len(texts)})")
            continue
        if len(texts) != 1 and len(texts) != len(leads):
            problems.append(f"{name}: has {len(texts)} values but there are {len(leads)} lead times "
                            f"(give 1 value, or 1 per lead time)")
            continue

        values = []
        for text in texts:
            if name in WHOLE_NUMBER_SETTINGS:
                value = to_number(text, whole=True)
                if value is None:
                    problems.append(f"{name}: '{text}' is not a whole number")
            elif name in DECIMAL_SETTINGS:
                value = to_number(text, whole=False)
                if value is None:
                    problems.append(f"{name}: '{text}' is not a number")
            else:
                value = text.lower()
                if value not in CHOICES[name]:
                    problems.append(f"{name}: '{text}' must be one of {', '.join(CHOICES[name])}")
            values.append(value)

        if len(values) == 1:
            values = values * len(leads)
        per_lead[name] = values

    # 4. Data files must exist
    for name in FILE_SETTINGS:
        files = split_list(raw[name])
        for file_name in files:
            if not (Path(DATA_DIR) / file_name).exists():
                problems.append(f"{name}: '{file_name}' is not in the data folder")
    if problems:
        raise ValueError("; ".join(problems))

    # 5. Settings that depend on each other
    rain = per_lead['rain'][0]
    future = per_lead['rain_future_hrs'][0]
    if rain == 'perfprog':
        if future < 1 or future > RAIN_PERFPROG_MAX_HRS:
            problems.append(f"rain_future_hrs must be 1 to {RAIN_PERFPROG_MAX_HRS} when rain is perfprog")
    elif future != 0:
        problems.append("rain_future_hrs must be 0 unless rain is perfprog")
    for name in ['train_gap_fill', 'val_gap_fill', 'test_gap_fill']:
        if per_lead[name][0] < 0:
            problems.append(f"{name} can't be negative")
    if per_lead['n_models'][0] < 1:
        problems.append("n_models must be at least 1")
    for i in range(len(leads)):
        if per_lead['lookback_hrs'][i] < 1:
            problems.append("lookback_hrs must be at least 1")
        if rain == 'reported' and per_lead['lookback_hrs'][i] <= RAIN_REPORT_DELAY_HRS:
            problems.append(f"lookback_hrs must be more than {RAIN_REPORT_DELAY_HRS} when rain is 'reported'")
        if per_lead['dropout'][i] < 0 or per_lead['dropout'][i] >= 1:
            problems.append("dropout must be from 0 up to (not including) 1")
        if per_lead['min_delta'][i] < 0:
            problems.append("min_delta can't be negative")
    if problems:
        raise ValueError("; ".join(sorted(set(problems))))

    # 6. One fully filled-in settings dict per lead time
    lead_cfgs = []
    for i in range(len(leads)):
        cfg = {'experiment': path.stem,
               'location': raw['location'],
               'lead_hrs': leads[i],
               'train_files': split_list(raw['train_files']),
               'val_files': split_list(raw['val_files']),
               'test_files': split_list(raw['test_files'])}
        for name in per_lead:
            cfg[name] = per_lead[name][i]
        cfg['rain_end_hrs'] = rain_end_hrs(cfg)
        lead_cfgs.append(cfg)

    return {'name': path.stem,
            'file': path.name,
            'fingerprint': file_fingerprint(path),
            'settings': raw,
            'location': raw['location'],
            'train_files': split_list(raw['train_files']),
            'val_files': split_list(raw['val_files']),
            'test_files': split_list(raw['test_files']),
            'leads': lead_cfgs}


def load_configs(config_dir):
    """Reads every config file in the folder.
    Returns ({experiment name: experiment} for good files, {file name: error message} for bad ones)"""
    good = {}
    bad = {}
    for path in sorted(Path(config_dir).glob("*.csv")):
        try:
            experiment = read_config_file(path)
            good[experiment['name']] = experiment
        except Exception as e:
            bad[path.name] = str(e)
    return good, bad


def rain_end_hrs(cfg):
    """Last rain input, in hours relative to the issue time (negative = before it). None when rain is off."""
    if cfg['rain'] == 'off':
        return None
    if cfg['rain'] == 'reported':
        return -RAIN_REPORT_DELAY_HRS
    if cfg['rain'] == 'to_now':
        return 0
    # perfprog: observed rain after the issue time, but not past the lead time or the cap
    return min(cfg['rain_future_hrs'], cfg['lead_hrs'], RAIN_PERFPROG_MAX_HRS)


def describe_rain(cfg):
    """Rain setting in words, for the report"""
    if cfg['rain'] == 'off':
        return "Not used"
    start = f"{cfg['lookback_hrs']} h before issue time"
    end_hrs = cfg['rain_end_hrs']
    if end_hrs < 0:
        end = f"{-end_hrs} h before issue time (only rain that would already be reported)"
    elif end_hrs == 0:
        end = "the issue time"
    else:
        end = f"{end_hrs} h after issue time (perfect prognosis: observed future rain)"
    return f"From {start} to {end}"


def describe_rain_setting(cfg):
    """An experiment's rain setting in words (the same for all its lead times), for the report"""
    if cfg['rain'] == 'off':
        return "Not used"
    if cfg['rain'] == 'reported':
        return f"Past rain only, stopping {RAIN_REPORT_DELAY_HRS} h before issue time (only rain that would already be reported)"
    if cfg['rain'] == 'to_now':
        return "Past rain up to the issue time"
    return (f"Perfect prognosis: observed rain up to {cfg['rain_future_hrs']} h after issue time, "
            f"but never past the lead time")


def input_windows(cfg):
    """Each input's time window, in hours relative to the issue time (negative = before it).
    e.g. lookback 24, lead 6, perfprog 18 -> water level -24..0, wind -24..+6, rain -24..+6"""
    back = -cfg['lookback_hrs']
    windows = [{'column': WL_COL, 'label': 'Water level', 'start_hrs': back, 'end_hrs': 0},
               {'column': UWIND_COL, 'label': 'Wind u (east-west)', 'start_hrs': back, 'end_hrs': cfg['lead_hrs']},
               {'column': VWIND_COL, 'label': 'Wind v (north-south)', 'start_hrs': back, 'end_hrs': cfg['lead_hrs']}]
    if cfg['rain'] != 'off':
        windows.append({'column': RAIN_COL, 'label': 'Rain', 'start_hrs': back, 'end_hrs': cfg['rain_end_hrs']})
    if cfg['poc'] == 'yes':
        windows.append({'column': POC_COL, 'label': "Port O'Connor water level", 'start_hrs': back, 'end_hrs': 0})
    return windows


def make_job_spec(cfg):
    """Builds the job_specification dict for one lead time (hours -> 6-min steps)"""
    lead = cfg['lead_hrs'] * STEPS_PER_HOUR
    back = -cfg['lookback_hrs'] * STEPS_PER_HOUR

    # Water level and wind are always inputs
    input_specifications = [{'column': WL_COL, 'lag_range': (back, 0)},
                            {'column': WL_COL, 'lag_range': (lead, lead)},  # this one becomes the target column
                            {'column': UWIND_COL, 'lag_range': (back, lead)},
                            {'column': VWIND_COL, 'lag_range': (back, lead)}]
    gap_fill_columns = [WL_COL, UWIND_COL, VWIND_COL]

    if cfg['rain'] != 'off':
        rain_end = cfg['rain_end_hrs'] * STEPS_PER_HOUR
        input_specifications.append({'column': RAIN_COL, 'lag_range': (back, rain_end)})

    if cfg['poc'] == 'yes':
        input_specifications.append({'column': POC_COL, 'lag_range': (back, 0)})
        gap_fill_columns.append(POC_COL)

    return {'target_column': f'{WL_COL}_t+{lead}',
            'input_specifications': input_specifications,
            'gap_fill_columns': gap_fill_columns}


def make_run_name(cfg, model_num):
    """e.g. pl_poc_rain_12h_m2 = experiment pl_poc_rain, 12 h lead, model 2"""
    return f"{cfg['experiment']}_{cfg['lead_hrs']}h_m{model_num}"


def build_model(cfg, x_train):
    """Same MLP as before, with the sizes coming from the config"""
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))  # fixes the input size, so a loaded model can check it

    # Scaling comes from the training rows only, and is saved inside the model file
    if cfg['normalize'] == 'yes':
        normalizer = Normalization()
        normalizer.adapt(x_train)
        model.add(normalizer)

    # Hidden layer(s), each followed by a dropout layer
    for _ in range(cfg['layers']):
        model.add(Dense(units = cfg['neurons'], activation = cfg['activation']))
        model.add(Dropout(rate = cfg['dropout']))

    # Output Layer outputs one value (water level at the lead time)
    model.add(Dense(1, kernel_initializer = 'normal'))

    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model


def build_arrays(df, cfg, gap_fill_hrs):
    """Input/target arrays for one lead time (see extract_input_output_arrays)"""
    return extract_input_output_arrays(df, make_job_spec(cfg), gap_fill_hrs * STEPS_PER_HOUR)


def check_experiment_data(experiment):
    """Loads an experiment's data and checks every column it needs is there, so problems show up before training"""
    for split in FILE_SETTINGS:
        df = load_split(experiment[split])
        for cfg in experiment['leads']:
            for spec in make_job_spec(cfg)['input_specifications']:
                if spec['column'] not in df.columns:
                    raise ValueError(f"{experiment['name']}: the {split} have no '{spec['column']}' data "
                                     f"(files: {', '.join(experiment[split])}). Columns found: {list(df.columns)}")


def try_download(path):
    """Sends a file to the browser's downloads. A failed download never stops training."""
    try:
        from google.colab import files
        files.download(path)
    except Exception as e:
        print(f"Download failed for {os.path.basename(path)} ({e}). The file is still at {path}")


def append_csv(row, csv_path):
    """Adds one row to a CSV, creating it if needed (columns can differ between rows)"""
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
    else:
        old = pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(csv_path, index=False)


def load_run(model_path):
    """Loads a saved model + its config, rebuilds the VALIDATION inputs exactly as in training,
    and evaluates. The testing set is never touched.

    Returns a dict with: model, cfg, run_name, x_val, y_val, predictions, datetime_index, input_columns, metrics"""
    model_path = Path(str(model_path).strip())
    if not model_path.name.endswith('_model.keras') or not model_path.exists():
        raise FileNotFoundError(f"No saved model at '{model_path}'. Pick one from the list, or paste the full path "
                                f"to a file ending in _model.keras")
    run_name = model_path.name[:-len('_model.keras')]
    saved = json.loads((model_path.parent / f"{run_name}_config.json").read_text())
    cfg = saved['cfg']

    # Warn if a data file changed since training
    for file_name, fingerprint in saved['data_fingerprints'].items():
        path = Path(DATA_DIR) / file_name
        if not path.exists():
            raise FileNotFoundError(f"{file_name} (used to train this model) is not in the data folder any more")
        if file_fingerprint(path) != fingerprint:
            print(f"Note: {file_name} has changed since this model was trained, so results can differ from the saved ones.")

    model = load_model(model_path)
    val = build_arrays(load_split(cfg['val_files']), cfg, cfg['val_gap_fill'])
    if val['input_columns'] != saved['input_columns']:
        raise ValueError("The inputs rebuilt now don't match the ones this model was trained on.")

    print(f"Loaded {run_name}: {val['input_array'].shape[0]:,} validation rows x {val['input_array'].shape[1]:,} inputs\n")
    metrics, predictions = evaluate_model(model, val['input_array'], val['target_array'])
    return {'model': model, 'cfg': cfg, 'run_name': run_name,
            'x_val': val['input_array'], 'y_val': val['target_array'], 'predictions': predictions,
            'datetime_index': val['datetime_index'], 'input_columns': val['input_columns'], 'metrics': metrics}


# =====================================================================================================
# 6. Report functions
# =====================================================================================================
# Builds the one report for a session: <session>_report.html, saved in the session folder. Used at the end of training and by the Build the report cells at the bottom of the notebook.

METRIC_COLUMNS = [  # (key, label, unit, scale, decimals, lower_is_better)
  ('rmse', 'RMSE', 'cm', 100, 1, True),
  ('mae', 'MAE', 'cm', 100, 1, True),
  ('medae', 'Median abs. error', 'cm', 100, 1, True),
  ('cf_15cm', 'Within ±15 cm', '%', 1, 1, False),
  ('cf_5cm', 'Within ±5 cm', '%', 1, 1, False),
  ('cf_1cm', 'Within ±1 cm', '%', 1, 1, False),
  ('r2', 'R²', '', 1, 3, False),
]

# Names of the data columns, as shown in the report
INPUT_LABELS = {WL_COL: 'Water level', UWIND_COL: 'Wind u', VWIND_COL: 'Wind v', RAIN_COL: 'Rain', POC_COL: "Port O'Connor water level"}
INPUT_UNITS = {WL_COL: 'm', UWIND_COL: 'm/s', VWIND_COL: 'm/s', RAIN_COL: 'mm per 6 min', POC_COL: 'm'}

# 'val' = validation set, 'test' = testing set
SET_NAMES = {'val': 'Validation', 'test': 'TESTING'}
SET_FILES = {'val': 'val_files', 'test': 'test_files'}


def esc(value):
    """Makes any value safe to put in HTML"""
    return html_lib.escape(str(value))


def no_nan(values):
    """Replaces NaN with None, so the list can be written as JSON"""
    out = []
    for v in values:
        if pd.isna(v):
            out.append(None)
        else:
            out.append(v)
    return out


def number(row, key):
    """A number from a results row, or NaN if the row doesn't have it (sessions trained with an older notebook)"""
    value = row.get(key)
    if value is None or pd.isna(value):
        return float('nan')
    return float(value)


def format_count(value):
    """12345 -> '12,345'; missing -> '—'"""
    if pd.isna(value):
        return '—'
    return f"{int(value):,}"


def format_sig(value):
    """3 significant digits; missing -> '—'"""
    if pd.isna(value):
        return '—'
    return f"{value:.3g}"


def format_metric(value, scale, decimals):
    if pd.isna(value):
        return '—'
    return f"{value * scale:.{decimals}f}"


def hours_text(hours):
    """-24 -> '24 h before', 0 -> 'issue time', 6 -> '6 h after'"""
    if hours < 0:
        return f"{-hours:g} h before"
    if hours == 0:
        return "issue time"
    return f"{hours:g} h after"


def rounded_list(series, decimals):
    """Numbers rounded for the report file (keeps it small). Missing values become None."""
    values = []
    for v in series:
        if pd.isna(v):
            values.append(None)
        else:
            values.append(round(float(v), decimals))
    return values


def whole_mm_list(series):
    """Metres -> whole millimetres (small numbers keep the report file small). Missing values become None."""
    values = []
    for v in series:
        if pd.isna(v):
            values.append(None)
        else:
            values.append(int(round(float(v) * 1000)))
    return values


# ---------------------------------------------------------------- reading a session folder

def collect_models(folder, include_test):
    """Every model in a session (or experiment) folder: its results, its saved config, and its folder.
    Testing results are only read when include_test is True."""
    models = []
    for results_path in sorted(Path(folder).rglob('*_results.csv')):
        if results_path.name == 'session_results.csv' or results_path.name.endswith('_test_results.csv'):
            continue
        run_name = results_path.name[:-len('_results.csv')]
        config_path = results_path.parent / f"{run_name}_config.json"
        saved = None
        if config_path.exists():
            saved = json.loads(config_path.read_text())
        test_row = None
        test_path = results_path.parent / f"{run_name}_test_results.csv"
        if include_test and test_path.exists():
            test_row = pd.read_csv(test_path).iloc[0].to_dict()
        row = pd.read_csv(results_path).iloc[0].to_dict()
        models.append({'run_name': run_name, 'run_dir': results_path.parent, 'row': row, 'test_row': test_row,
                       'saved': saved})
    return models


def result_rows(lead_models, which):
    """Results rows of the models that finished: which='val' (validation) or 'test' (testing)"""
    rows = []
    for m in lead_models:
        if which == 'val':
            if m['row']['status'] == 'ok':
                rows.append(m['row'])
        else:
            if m['test_row'] is not None:
                rows.append(m['test_row'])
    return rows


def first_saved(lead_models):
    """The saved config of the first model that has one (they're the same for every model of a lead)"""
    for m in lead_models:
        if m['saved'] is not None:
            return m['saved']
    return None


def summarize_lead(lead_models, which):
    """Mean and standard deviation of each metric over the models of one lead time"""
    rows = result_rows(lead_models, which)
    summary = {'n_ok': len(rows), 'n_total': len(lead_models), 'mean': {}, 'std': {}}
    for key, *_ in METRIC_COLUMNS:
        values = []
        for row in rows:
            values.append(float(row[key]))
        if len(values) == 0:
            summary['mean'][key] = float('nan')
            summary['std'][key] = float('nan')
        else:
            summary['mean'][key] = float(np.mean(values))
            summary['std'][key] = float(np.std(values))
    total_seconds = 0
    for row in result_rows(lead_models, 'val'):
        total_seconds += number(row, 'train_seconds')  # NaN for older sessions, shown as '—'
    summary['train_minutes'] = total_seconds / 60
    return summary


# ---------------------------------------------------------------- data series (stored once, shared by every chart)

def add_series(store, grids, file_name, column):
    """Puts one column of a data file into the report's shared series store (only once) and returns its id"""
    series_id = f"{file_name}|{column}"
    if series_id not in store:
        s = load_file(file_name)[column]
        grid = pd.date_range(s.index.min(), s.index.max(), freq='6min')
        store[series_id] = {'start_ms': int(grid[0].value // 10**6), 'step_ms': 6 * 60 * 1000,
                            'y': rounded_list(s.reindex(grid), 3)}
        grids[series_id] = grid
    return series_id


def find_file_with(file_names, column):
    """The file (of a list) that has this column, or None"""
    for name in file_names:
        if column in load_file(name).columns:
            return name
    return None


def time_series_for_set(e, which, store, grids):
    """Observed water level, the mean prediction of the n models for each lead, each model's difference from that
    mean, and the input series, for the validation ('val') or testing ('test') set. None if there's nothing to show."""
    cfg = e['cfg']
    files_key = SET_FILES[which]
    if files_key not in cfg:
        return None, ""
    files = cfg[files_key]
    station_file = find_file_with(files, WL_COL)
    if station_file is None:
        return None, ""
    observed_id = add_series(store, grids, station_file, WL_COL)
    grid = grids[observed_id]

    preds = {}
    models = {}
    first_time = None
    last_time = None
    for lead in e['leads']:
        model_preds = []
        for m in e['models_by_lead'][lead]:
            if which == 'val':
                path = m['run_dir'] / f"{m['run_name']}_predictions.csv"
            else:
                path = m['run_dir'] / f"{m['run_name']}_test_predictions.csv"
            if m['row']['status'] != 'ok' or not path.exists():
                continue
            df = pd.read_csv(path, parse_dates=['valid_time_utc']).set_index('valid_time_utc')
            model_preds.append(df['predicted_m'].reindex(grid))
            if first_time is None or df.index.min() < first_time:
                first_time = df.index.min()
            if last_time is None or df.index.max() > last_time:
                last_time = df.index.max()
        if not model_preds:
            continue
        mean = pd.concat(model_preds, axis=1).mean(axis=1)
        preds[str(lead)] = rounded_list(mean, 3)
        offsets = []
        for p in model_preds:
            offsets.append(whole_mm_list(p - mean))
        models[str(lead)] = offsets
    if not preds:
        return None, ""

    inputs = []
    for window in input_windows(cfg):
        input_file = find_file_with(files, window['column'])
        if input_file is not None:
            inputs.append({'name': f"{INPUT_LABELS[window['column']]} ({INPUT_UNITS[window['column']]})",
                           'id': add_series(store, grids, input_file, window['column'])})

    period = f" (forecasts for {first_time:%Y-%m-%d} to {last_time:%Y-%m-%d} UTC)"
    return {'observed': observed_id, 'preds': preds, 'models': models, 'inputs': inputs}, period


def find_gaps(values, grid):
    """Every run of missing values: list of [start ms, end ms], plus the longest run in hours"""
    gaps = []
    longest = 0
    start = None
    for i in range(len(values)):
        if values[i] is None:
            if start is None:
                start = i
        elif start is not None:
            gaps.append([int(grid[start].value // 10**6), int(grid[i - 1].value // 10**6)])
            longest = max(longest, i - start)
            start = None
    if start is not None:
        gaps.append([int(grid[start].value // 10**6), int(grid[len(values) - 1].value // 10**6)])
        longest = max(longest, len(values) - start)
    return gaps, longest / STEPS_PER_HOUR


def data_files_section(experiments, include_test, store, grids):
    """A table describing every data file the experiments used, and the data for its plots and gap timelines"""
    uses = {}  # file name -> list of 'training' / 'validation' / 'testing'
    for e in experiments:
        for files_key, label in [('train_files', 'training'), ('val_files', 'validation'), ('test_files', 'testing')]:
            if label == 'testing' and not include_test:
                continue
            for name in e['cfg'].get(files_key, []):
                if name not in uses:
                    uses[name] = []
                if label not in uses[name]:
                    uses[name].append(label)

    body = ""
    files_data = []
    for i, name in enumerate(uses):
        columns = []
        for column in load_file(name).columns:
            series_id = add_series(store, grids, name, column)
            values = store[series_id]['y']
            gaps, longest_hrs = find_gaps(values, grids[series_id])
            present = []
            for v in values:
                if v is not None:
                    present.append(v)
            missing_pct = 100 * (len(values) - len(present)) / len(values)
            grid = grids[series_id]
            if present:
                low, avg, high = f"{min(present):.3f}", f"{np.mean(present):.3f}", f"{max(present):.3f}"
            else:
                low, avg, high = '—', '—', '—'
            body += (f"<tr><th scope='row'><a href='#file-{i}'>{esc(name)}</a></th><td>{esc(', '.join(uses[name]))}</td>"
                     f"<td>{esc(INPUT_LABELS.get(column, column))} ({esc(INPUT_UNITS.get(column, ''))})</td>"
                     f"<td>{grid[0]:%Y-%m-%d %H:%M}</td><td>{grid[-1]:%Y-%m-%d %H:%M}</td>"
                     f"<td>{missing_pct:.2f}</td><td>{len(gaps):,}</td><td>{longest_hrs:g}</td>"
                     f"<td>{low}</td><td>{avg}</td><td>{high}</td></tr>")
            columns.append({'label': f"{INPUT_LABELS.get(column, column)} ({INPUT_UNITS.get(column, '')})",
                            'id': series_id, 'gaps': gaps})
        files_data.append({'name': name, 'uses': uses[name], 'columns': columns})

    head = ("<th scope='col'>File</th><th scope='col'>Used for</th><th scope='col'>Column</th><th scope='col'>From (UTC)</th>"
            "<th scope='col'>To (UTC)</th><th scope='col'>Missing (%)</th><th scope='col'>Gaps</th>"
            "<th scope='col'>Longest gap (h)</th><th scope='col'>Min</th><th scope='col'>Mean</th><th scope='col'>Max</th>")
    blocks = ""
    for i, f in enumerate(files_data):
        blocks += f"""
    <details class='datafile' id='file-{i}' data-file='{i}'><summary>{esc(f['name'])} <span class='pm'>({esc(', '.join(f['uses']))})</span></summary>
      <div class='chart'><h3>Missing data (each bar is a gap)</h3><div class='plot' id='gaps-{i}'></div></div>
      <div class='filecharts' id='filecharts-{i}'></div>
    </details>"""
    return table_html(head, body) + blocks, files_data


# ---------------------------------------------------------------- per-lead data for the charts

def loss_data(lead_models):
    """Training/validation loss per epoch for each finished model of one lead"""
    curves = []
    for m in lead_models:
        history_path = m['run_dir'] / f"{m['run_name']}_history.csv"
        if m['row']['status'] != 'ok' or not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        loss = []
        for v in history['loss']:
            loss.append(float(f"{v:.4g}"))
        val = []
        for v in history['val_loss']:
            val.append(float(f"{v:.4g}"))
        seconds = []
        if 'epoch_seconds' in history.columns:  # older sessions don't have per-epoch times
            for v in history['epoch_seconds']:
                seconds.append(float(f"{v:.4g}"))
        curves.append({'name': f"model {int(m['row']['model_num'])}", 'loss': loss, 'val': val, 'seconds': seconds})
    return curves


def split_column_name(name):
    """'precip_t+180' -> ('precip', 180); 'wl' -> ('wl', 0)"""
    if '_t' in name:
        base, lag_text = name.rsplit('_t', 1)
        return base, int(lag_text)
    return name, 0


def input_vector_data(lead_models):
    """Column names + one example row of the inputs of one lead, with each column's group (for the filters)"""
    saved = first_saved(lead_models)
    if saved is None or 'example_input' not in saved:
        return None
    groups = []
    group_of_column = []
    for name in saved['input_columns']:
        base, lag = split_column_name(name)
        if lag <= 0:
            when = 'past'
        else:
            when = 'future'
        label = f"{INPUT_LABELS.get(base, base)}, {when}"
        if label not in groups:
            groups.append(label)
        group_of_column.append(groups.index(label))

    counts = []
    for _ in groups:
        counts.append(0)
    for g in group_of_column:
        counts[g] += 1
    group_list = []
    for i, label in enumerate(groups):
        group_list.append({'label': f"{label} ({counts[i]} values)"})

    return {'columns': saved['input_columns'], 'values': saved['example_input']['values'],
            'issue_time': saved['example_input']['issue_time_utc'], 'groups': group_list, 'group_of': group_of_column}


# ---------------------------------------------------------------- HTML pieces

def shade(value, lo, hi, lower_is_better):
    """0 = best in this table, 1 = worst; used to tint the metric cells"""
    if hi == lo or pd.isna(value):
        return 0.0
    t = (value - lo) / (hi - lo)
    if lower_is_better:
        return t
    return 1 - t


def metric_header_cells():
    cells = ""
    for key, label, unit, *_ in METRIC_COLUMNS:
        if unit:
            label = f"{label} ({unit})"
        cells += f"<th scope='col'>{esc(label)}</th>"
    return cells


def table_html(head, body):
    return f"<div class='tablewrap'><table class='resizable'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def set_cell(which):
    if which == 'test':
        return "<td class='settag testtag'>TEST</td>"
    return "<td class='settag'>Validation</td>"


def metric_cells(summary, ranges, which):
    """The metric cells of one table row: mean (± std when there are several models)"""
    cells = ""
    for key, _, _, scale, dec, lower in METRIC_COLUMNS:
        mean = summary['mean'][key]
        tint = shade(mean, ranges[which][key][0], ranges[which][key][1], lower)
        text = format_metric(mean, scale, dec)
        if summary['n_ok'] > 1:
            text += f" <span class='pm'>± {format_metric(summary['std'][key], scale, dec)}</span>"
        cells += f"<td style='--s:{tint:.3f}'>{text}</td>"
    return cells


def average_table(e, sets):
    """One row per lead (and per set): mean ± std over the n models"""
    ranges = {}
    for which in sets:
        ranges[which] = {}
        for key, *_ in METRIC_COLUMNS:
            means = []
            for lead in e['leads']:
                value = e['summaries'][which][lead]['mean'][key]
                if not pd.isna(value):
                    means.append(value)
            if means:
                ranges[which][key] = (min(means), max(means))
            else:
                ranges[which][key] = (0, 0)

    body = ""
    for lead in e['leads']:
        for which in sets:
            s = e['summaries'][which][lead]
            cells = f"<th scope='row'>{lead} h</th>"
            if len(sets) > 1:
                cells += set_cell(which)
            cells += metric_cells(s, ranges, which)
            cells += f"<td>{s['n_ok']} of {s['n_total']}</td>"
            val_rows = result_rows(e['models_by_lead'][lead], 'val')
            if which == 'test' or len(val_rows) == 0:
                cells += "<td></td><td></td><td></td><td></td><td></td>"
            else:
                first = val_rows[0]
                cells += (f"<td>{format_count(number(first, 'train_rows'))}</td><td>{format_count(number(first, 'val_rows'))}</td>"
                          f"<td>{format_count(number(first, 'test_rows'))}</td><td>{format_count(number(first, 'n_inputs'))}</td>"
                          f"<td>{format_metric(s['train_minutes'], 1, 1)}</td>")
            row_class = ''
            if which == 'test':
                row_class = " class='testrow'"
            body += f"<tr{row_class}>{cells}</tr>"

    head = "<th scope='col'>Lead</th>"
    if len(sets) > 1:
        head += "<th scope='col'>Set</th>"
    head += (metric_header_cells() +
             "<th scope='col'>Models</th><th scope='col'>Training rows</th><th scope='col'>Validation rows</th>"
             "<th scope='col'>Testing rows</th><th scope='col'>Inputs</th><th scope='col'>Training time, all models (min)</th>")
    return table_html(head, body)


def per_model_table(lead_models, summaries, sets):
    """One row per model of a lead (metrics, losses, timings), plus the mean row. With testing: a TEST row under each."""
    head = "<th scope='col'>Model</th>"
    if len(sets) > 1:
        head += "<th scope='col'>Set</th>"
    head += ("<th scope='col'>Seed</th><th scope='col'>Best epoch</th><th scope='col'>Best val loss</th>"
             "<th scope='col'>Train loss at best</th>" + metric_header_cells() +
             "<th scope='col'>Training time (min)</th><th scope='col'>s / epoch</th>")

    body = ""
    for m in lead_models:
        row = m['row']
        for which in sets:
            cells = f"<th scope='row'>{int(row['model_num'])}</th>"
            if len(sets) > 1:
                cells += set_cell(which)
            if which == 'val':
                cells += f"<td>{int(row['seed_used'])}</td>"
                if row['status'] != 'ok':
                    cells += f"<td colspan='{4 + len(METRIC_COLUMNS) + 1}' class='failed'>{esc(row['status'])}</td>"
                else:
                    cells += f"<td>{int(row['best_epoch']):,} of {int(row['epochs_trained']):,}</td>"
                    cells += f"<td>{format_sig(number(row, 'best_val_loss'))}</td><td>{format_sig(number(row, 'train_loss_at_best'))}</td>"
                    for key, _, _, scale, dec, _ in METRIC_COLUMNS:
                        cells += f"<td>{format_metric(float(row[key]), scale, dec)}</td>"
                    cells += (f"<td>{format_metric(number(row, 'train_seconds'), 1 / 60, 1)}</td>"
                              f"<td>{format_metric(number(row, 'seconds_per_epoch'), 1, 3)}</td>")
                body += f"<tr>{cells}</tr>"
            else:
                cells += "<td></td><td></td><td></td><td></td>"
                if m['test_row'] is None:
                    cells += f"<td colspan='{len(METRIC_COLUMNS) + 2}' class='failed'>No testing results saved</td>"
                else:
                    for key, _, _, scale, dec, _ in METRIC_COLUMNS:
                        cells += f"<td>{format_metric(number(m['test_row'], key), scale, dec)}</td>"
                    cells += "<td></td><td></td>"
                body += f"<tr class='testrow'>{cells}</tr>"

    for which in sets:
        cells = "<th scope='row'>Mean</th>"
        if len(sets) > 1:
            cells += set_cell(which)
        cells += "<td></td><td></td><td></td><td></td>"
        for key, _, _, scale, dec, _ in METRIC_COLUMNS:
            cells += f"<td>{format_metric(summaries[which]['mean'][key], scale, dec)}</td>"
        cells += "<td></td><td></td>"
        body += f"<tr class='meanrow'>{cells}</tr>"
    return table_html(head, body)


def model_structure_block(lead_models):
    """The model's layers as a row of boxes: input -> hidden layer(s) -> output, with sizes and parameter counts"""
    saved = first_saved(lead_models)
    if saved is None or 'architecture' not in saved:
        return "<p class='note'>Model structure not saved for this session (trained with an older notebook).</p>"
    boxes = ""
    for i, layer in enumerate(saved['architecture']['layers']):
        if i > 0:
            boxes += "<span class='arrow'>→</span>"
        params = ""
        if layer['params'] > 0:
            params = f"<span class='pm'>{layer['params']:,} parameters</span>"
        boxes += (f"<div class='layer'><b>{esc(layer['type'])}</b><span>{esc(layer['detail'])}</span>"
                  f"<span class='pm'>output {esc(layer['output'])}</span>{params}</div>")
    total = saved['architecture']['total_params']
    return (f"<div class='modelgraph'>{boxes}</div>"
            f"<p class='note'>Total: <b>{total:,}</b> trainable parameters. Every model of this lead has the same structure; only the seed differs.</p>")


def config_block(saved):
    """The experiment's config exactly as written in its file, plus fingerprints of the files used"""
    rows = ""
    for name, value in saved['experiment_settings'].items():
        rows += f"<tr><th scope='row'>{esc(name)}</th><td>{esc(value)}</td></tr>"
    rows += f"<tr><th scope='row'>config file</th><td>{esc(saved['config_file'])} " \
            f"<span class='pm'>(fingerprint {esc(saved['config_fingerprint'])})</span></td></tr>"
    for file_name, fingerprint in saved['data_fingerprints'].items():
        rows += f"<tr><th scope='row'>data file</th><td>{esc(file_name)} <span class='pm'>(fingerprint {esc(fingerprint)})</span></td></tr>"
    rows += f"<tr><th scope='row'>repo version</th><td>{esc(saved['commit'])}</td></tr>"
    return f"<table class='config'><tbody>{rows}</tbody></table>"


def windows_list(lead, cfg):
    """The input windows of one lead, as an HTML list item"""
    parts = []
    for window in input_windows(cfg):
        parts.append(f"{esc(window['label'])}: {hours_text(window['start_hrs'])} → {hours_text(window['end_hrs'])}")
    return f"<li><b>{lead} h lead</b>: {'; '.join(parts)}</li>"


def comparison_table(experiments, which, key, scale, decimals):
    """Experiments as rows, lead times as columns, mean of one metric"""
    all_leads = []
    for e in experiments:
        for lead in e['leads']:
            if lead not in all_leads:
                all_leads.append(lead)
    all_leads.sort()

    head = "<th scope='col'>Experiment</th><th scope='col'>Location</th><th scope='col'>Rain</th><th scope='col'>POC</th>"
    for lead in all_leads:
        head += f"<th scope='col'>{lead} h</th>"
    body = ""
    for e in experiments:
        cfg = e['cfg']
        cells = f"<th scope='row'><a href='#{e['id']}'>{esc(e['name'])}</a></th>"
        cells += f"<td>{esc(cfg['location'])}</td><td>{esc(cfg['rain'])}</td><td>{esc(cfg['poc'])}</td>"
        for lead in all_leads:
            if lead in e['summaries'][which]:
                cells += f"<td>{format_metric(e['summaries'][which][lead]['mean'][key], scale, decimals)}</td>"
            else:
                cells += "<td></td>"
        body += f"<tr>{cells}</tr>"
    return table_html(head, body)


def time_series_block(eid, which, n_models, windows_html):
    """The time series heading, explanation and (empty) plot box of one set"""
    if which == 'val':
        tag = "<b class='valtag'>Validation set.</b>"
        title = "Time series: validation set"
        anchor = f"{eid}-ts"
    else:
        tag = "<b class='valtag testtag'>TESTING set.</b>"
        title = "Time series: TESTING set"
        anchor = f"{eid}-ts-test"
    return f"""
    <h3 id='{anchor}'>{title}</h3>
    <p class='note'>{tag} Observed water level and, for each lead, the <b>mean prediction of the {n_models} models</b>, plotted at the time each forecast is for.
    Also in the legend, turned off: each lead's min–max of the {n_models} models, each single model, and the input series (right-hand axis). Click a legend name to show or hide it.
    In the plot's toolbar (top right), the <b>▭ Input window</b> button turns on a mode that, as you hover a lead's line, shades the stretch of each input that forecast was made from.</p>
    <details><summary>Input windows per lead (relative to the issue time = forecast time − lead)</summary><ul class='windows'>{windows_html}</ul></details>
    <button type='button' class='tsbutton' data-exp='{eid}' data-set='{which}'>Show time series</button>
    <div class='chart tall' hidden><h3>{title}</h3><div class='plot' id='ts-{eid}-{which}'></div></div>"""


# ---------------------------------------------------------------- the report

def build_session_report(session_dir, experiment_name=None, include_test=False):
    """Writes a report into the session folder and returns its path.
    experiment_name=None: every experiment in the session; experiment_name given: only that experiment.
    include_test=False: validation only (<session>_report.html), the one to share.
    include_test=True: validation AND testing (<session>_report_VAL+TEST.html), only on request."""
    session_dir = Path(session_dir)
    if experiment_name is None:
        models = collect_models(session_dir, include_test)
        title = f"Session {session_dir.name}"
        file_stem = f"{session_dir.name}_report"
    else:
        models = collect_models(session_dir / experiment_name, include_test)
        title = f"Session {session_dir.name}: {experiment_name}"
        file_stem = f"{session_dir.name}_{experiment_name}_report"
    sets = ['val']
    if include_test:
        sets.append('test')
        title += " (VALIDATION + TESTING)"
        file_stem += "_VAL+TEST"
    out_path = session_dir / f"{file_stem}.html"
    if not models:
        raise FileNotFoundError(f"No models found in {session_dir}. Pick a session folder.")

    # Group the models: experiment -> lead -> models
    grouped = {}
    for m in models:
        experiment = m['row']['experiment']
        lead = int(m['row']['lead_hrs'])
        if experiment not in grouped:
            grouped[experiment] = {}
        if lead not in grouped[experiment]:
            grouped[experiment][lead] = []
        grouped[experiment][lead].append(m)

    experiments = []
    for i, name in enumerate(sorted(grouped)):
        models_by_lead = grouped[name]
        leads = sorted(models_by_lead)
        summaries = {}
        for which in sets:
            summaries[which] = {}
        saved = None
        for lead in leads:
            models_by_lead[lead].sort(key=lambda m: int(m['row']['model_num']))
            for which in sets:
                summaries[which][lead] = summarize_lead(models_by_lead[lead], which)
            if saved is None:
                saved = first_saved(models_by_lead[lead])
        experiments.append({'id': f"e{i}", 'name': name, 'leads': leads, 'models_by_lead': models_by_lead,
                            'summaries': summaries, 'saved': saved, 'cfg': saved['cfg']})

    # ---- data series, stored once and shared (several experiments often use the same files) ----
    store = {}
    grids = {}
    try:
        data_files_html, files_data = data_files_section(experiments, include_test, store, grids)
        data_ok = True
    except Exception as e:
        print(f"  (data files left out of the report: {e})")
        data_files_html = f"<p class='note'>The data files couldn't be read when this report was made ({esc(e)}).</p>"
        files_data = []
        data_ok = False

    # ---- page data for the charts ----
    chart_data = []
    for e in experiments:
        mean = {}
        std = {}
        for which in sets:
            mean[which] = {}
            std[which] = {}
            for key, *_ in METRIC_COLUMNS:
                means = []
                stds = []
                for lead in e['leads']:
                    means.append(e['summaries'][which][lead]['mean'][key])
                    stds.append(e['summaries'][which][lead]['std'][key])
                mean[which][key] = no_nan(means)
                std[which][key] = no_nan(stds)
        losses = {}
        vectors = {}
        windows = {}
        for lead in e['leads']:
            losses[str(lead)] = loss_data(e['models_by_lead'][lead])
            vectors[str(lead)] = input_vector_data(e['models_by_lead'][lead])
            windows[str(lead)] = input_windows(first_saved(e['models_by_lead'][lead])['cfg'])
        ts = {}
        e['periods'] = {}
        for which in sets:
            ts[which] = None
            e['periods'][which] = ""
            if data_ok:
                ts[which], e['periods'][which] = time_series_for_set(e, which, store, grids)
        chart_data.append({'id': e['id'], 'leads': e['leads'], 'sets': sets, 'mean': mean, 'std': std,
                           'loss': losses, 'vectors': vectors, 'windows': windows, 'ts': ts})

    # ---- HTML ----
    nav = ("<li><a href='#overview'>Overview</a></li>"
           "<li><a href='#compare'>Compare experiments</a><ul>"
           "<li><a href='#compare-rmse'>RMSE</a></li><li><a href='#compare-cf'>Within ±15 cm</a></li></ul></li>"
           "<li><a href='#datafiles'>Data files</a></li>")
    sections = ""
    for e in experiments:
        cfg = e['cfg']
        eid = e['id']
        n_models = int(cfg['n_models'])

        lead_nav = ""
        for lead in e['leads']:
            lead_nav += f"<li><a href='#{eid}-lead-{lead}'>{lead} h</a></li>"
        test_nav = ""
        if include_test:
            test_nav = f"<li><a href='#{eid}-ts-test'>Time series (TEST)</a></li>"
        nav += (f"<li><a href='#{eid}'>{esc(e['name'])}</a><ul>"
                f"<li><a href='#{eid}-config'>Config used</a></li>"
                f"<li><a href='#{eid}-inputs'>Input vector</a></li>"
                f"<li><a href='#{eid}-average'>Average metrics</a></li>"
                f"<li><a href='#{eid}-charts'>Error vs lead</a></li>"
                f"<li><a href='#{eid}-ts'>Time series</a></li>{test_nav}"
                f"<li><a href='#{eid}-errors'>Errors</a></li>"
                f"<li><a href='#{eid}-models'>Each model</a><ul>{lead_nav}</ul></li>"
                f"</ul></li>")

        if 'test_files' not in cfg:
            testing_text = "not listed (trained with an older notebook)"
        elif include_test:
            testing_text = '; '.join(cfg['test_files']) + e['periods']['test'] + ": results shown in this report"
        else:
            testing_text = '; '.join(cfg['test_files']) + ": results are NOT in this report (see the VAL+TEST report)"

        inputs = ["water level", "wind (u and v)"]
        if cfg['poc'] == 'yes':
            inputs.append("Port O'Connor water level")
        if cfg['rain'] != 'off':
            inputs.append("rain")

        lead_options = ""
        windows_html = ""
        lead_details = ""
        for lead in e['leads']:
            lead_options += f"<option value='{lead}'>{lead} h lead</option>"
            windows_html += windows_list(lead, first_saved(e['models_by_lead'][lead])['cfg'])
            lead_summaries = {}
            for which in sets:
                lead_summaries[which] = e['summaries'][which][lead]
            lead_details += f"""
        <details class='lead' id='{eid}-lead-{lead}' data-exp='{eid}' data-lead='{lead}'>
          <summary>{lead} h lead: each model</summary>
          <p class='note'><b class='valtag'>Validation set</b> unless marked TEST. Loss is the mean squared error (m²).</p>
          <h4>Model structure</h4>
          {model_structure_block(e['models_by_lead'][lead])}
          {per_model_table(e['models_by_lead'][lead], lead_summaries, sets)}
          <div class='charts'>
            <div class='chart'><h3>Loss: training (dashed) and validation</h3><div class='plot' id='loss-{eid}-{lead}'></div></div>
            <div class='chart'><h3>Loss, log scale</h3><div class='plot' id='losslog-{eid}-{lead}'></div></div>
            <div class='chart'><h3>Seconds per epoch</h3><div class='plot' id='epochtime-{eid}-{lead}'></div></div>
          </div>
        </details>"""

        set_options = ""
        for which in sets:
            set_options += f"<option value='{which}'>{SET_NAMES[which]}</option>"
        model_options = "<option value='mean'>Mean of the models</option>"
        for k in range(1, n_models + 1):
            model_options += f"<option value='{k - 1}'>Model {k}</option>"

        ts_blocks = time_series_block(eid, 'val', n_models, windows_html)
        if include_test:
            ts_blocks += time_series_block(eid, 'test', n_models, windows_html)

        average_title = f"Validation set: average of the {n_models} models per lead (mean ± standard deviation)"
        if include_test:
            average_title = f"Validation and TESTING sets: average of the {n_models} models per lead (mean ± standard deviation)"

        sections += f"""
  <section id='{eid}' class='experiment'>
    <h2>{esc(e['name'])} <span class='sub'>{esc(cfg['location'])}</span></h2>
    <dl class='facts'>
      <dt>Scored on</dt><dd><b class='valtag'>Validation set</b> {esc('; '.join(cfg['val_files']))}{esc(e['periods']['val'])}</dd>
      <dt>Trained on</dt><dd>{esc('; '.join(cfg['train_files']))}</dd>
      <dt>Testing set</dt><dd>{esc(testing_text)}</dd>
      <dt>Inputs</dt><dd>{esc(', '.join(inputs))}</dd>
      <dt>Rain</dt><dd>{esc(cfg['rain'])}: {esc(describe_rain_setting(cfg))}</dd>
      <dt>Models per lead</dt><dd>{n_models}, seeds {int(cfg['seed'])} to {int(cfg['seed']) + n_models - 1}. Each model's seed is set for Python (<code>random</code>), NumPy and TensorFlow, all three at once (<code>keras.utils.set_random_seed</code>).</dd>
    </dl>
    <details id='{eid}-config' open><summary>Config used</summary>{config_block(e['saved'])}</details>

    <details id='{eid}-inputs' class='inputs' data-exp='{eid}'><summary>Input vector (what the model receives)</summary>
      <p class='note'>One example row from the validation set, with the column names. <code>_t-240</code> means 240 steps of 6 min (24 h) before the issue time; <code>_t+60</code> means 6 h after. Use the boxes to show or hide groups of columns.</p>
      <div class='vectortools'>
        <label>Lead <select class='vlead'>{lead_options}</select></label>
        <label>Filter <input type='search' class='vfilter' placeholder='e.g. precip_t+'></label>
        <span class='vcount'></span>
      </div>
      <div class='vgroups'></div>
      <div class='vector'></div>
    </details>

    <h3 id='{eid}-average'>{average_title}</h3>
    {average_table(e, sets)}
    <div class='charts' id='{eid}-charts'>
      <div class='chart'><h3>Error size (cm)</h3><div class='plot' id='err-{eid}'></div></div>
      <div class='chart'><h3>Forecasts within a tolerance (%)</h3><div class='plot' id='cf-{eid}'></div></div>
      <div class='chart'><h3>R²</h3><div class='plot' id='r2-{eid}'></div></div>
    </div>
    {ts_blocks}

    <h3 id='{eid}-errors'>Errors</h3>
    <details class='errors' data-exp='{eid}'><summary>Error plots and the worst forecasts</summary>
      <p class='note'>Error = predicted − observed. Pick the lead, the set and the model (or the mean of the models).</p>
      <div class='vectortools'>
        <label>Lead <select class='elead'>{lead_options}</select></label>
        <label>Set <select class='eset'>{set_options}</select></label>
        <label>Model <select class='emodel'>{model_options}</select></label>
        <span class='ecount'></span>
      </div>
      <div class='chart wide'><h3>Error over time (cm), at the time each forecast is for</h3><div class='plot' id='etime-{eid}'></div></div>
      <div class='charts'>
        <div class='chart'><h3>How often each error size happens (cm)</h3><div class='plot' id='ehist-{eid}'></div></div>
        <div class='chart'><h3>Observed vs predicted (m)</h3><div class='plot' id='escatter-{eid}'></div></div>
      </div>
      <h3>The 20 worst forecasts</h3>
      <div class='tablewrap'><table class='worst'><thead><tr><th scope='col'>Issued (UTC)</th><th scope='col'>For (UTC)</th>
        <th scope='col'>Observed (m)</th><th scope='col'>Predicted (m)</th><th scope='col'>Error (cm)</th></tr></thead><tbody></tbody></table></div>
    </details>

    <h3 id='{eid}-models'>Each model</h3>
    {lead_details}
  </section>"""

    commits = []
    experiment_names = []
    for e in experiments:
        if e['saved']['commit'] not in commits:
            commits.append(e['saved']['commit'])
        experiment_names.append(e['name'])

    compare = (f"<h3 id='compare-rmse'>Validation set: RMSE (cm), mean of the models</h3>"
               f"{comparison_table(experiments, 'val', 'rmse', 100, 1)}"
               f"<h3 id='compare-cf'>Validation set: within ±15 cm (%), mean of the models</h3>"
               f"{comparison_table(experiments, 'val', 'cf_15cm', 1, 1)}")
    if include_test:
        compare += (f"<h3><span class='valtag testtag'>TESTING set</span> RMSE (cm), mean of the models</h3>"
                    f"{comparison_table(experiments, 'test', 'rmse', 100, 1)}"
                    f"<h3><span class='valtag testtag'>TESTING set</span> within ±15 cm (%), mean of the models</h3>"
                    f"{comparison_table(experiments, 'test', 'cf_15cm', 1, 1)}")

    if include_test:
        banner = ("<div class='banner danger'><b>⚠ This report includes TESTING results.</b> The testing set is the only data "
                  "that played no part in training or in choosing when to stop. Once testing results influence which settings or "
                  "experiments you pick, they stop being an independent check. Use them for final confirmation only, "
                  "and share the validation-only report by default. Rows marked <b>TEST</b> are testing results; "
                  "everything else is the validation set.</div>")
    else:
        banner = ("<div class='banner'><b>Everything in this report is on the VALIDATION set.</b> These are the "
                  "<code>val_files</code> listed under \"Scored on\" in each experiment. The validation set is also used for "
                  "early stopping, so these numbers are slightly optimistic. The testing results are in a separate "
                  "VAL+TEST report, available on request.</div>")

    page = REPORT_TEMPLATE
    replacements = {
        '__TITLE__': esc(title),
        '__NAV__': nav,
        '__EXPERIMENTS__': esc(", ".join(experiment_names)),
        '__COMMITS__': esc(", ".join(commits)),
        '__GENERATED__': datetime.now().strftime("%Y-%m-%d %H:%M"),
        '__BANNER__': banner,
        '__COMPARE__': compare,
        '__DATA_FILES__': data_files_html,
        '__SECTIONS__': sections,
        '__RAIN_DELAY__': f"{RAIN_REPORT_DELAY_HRS:g}",
        '__RAIN_CAP__': f"{RAIN_PERFPROG_MAX_HRS:g}",
        '__DATA__': json.dumps({'experiments': chart_data, 'series': store, 'files': files_data},
                               separators=(',', ':'), allow_nan=False),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)

    out_path.write_text(page, encoding='utf-8')
    return out_path


REPORT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--paper:#F2F5F6;--panel:#FFFFFF;--ink:#14324A;--ink-2:#4D6577;--rule:#D3DDE2;--sea:#1F6F7A;--sea-soft:#D8EBEE;--sand:#F3E3BC;--bad:#B3261E;--bad-soft:#F9DEDC;
  --font:"Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root{--paper:#0E1E2B;--panel:#132838;--ink:#E2EBEF;--ink-2:#9DB2BF;--rule:#244257;--sea:#6FC1CC;--sea-soft:#1B4150;--sand:#5C4C25;--bad:#F2B8B5;--bad-soft:#5A1D1A}}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:16px}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);font-size:16px;line-height:1.5;font-variant-numeric:tabular-nums}
a{color:var(--sea)}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr)}
body.navhidden .layout{grid-template-columns:minmax(0,1fr)}
body.navhidden nav{display:none}
#navtoggle{position:fixed;top:8px;left:8px;z-index:10;padding:4px 10px;font-size:13px}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:48px 16px 32px;border-right:1px solid var(--rule)}
nav ul{list-style:none;margin:0;padding:0} nav li{margin:1px 0}
nav ul ul{padding-left:14px} nav ul ul a{font-size:13px;padding:2px 8px}
nav a{display:block;padding:4px 8px;border-radius:4px;color:var(--ink-2);text-decoration:none;font-size:14px;overflow-wrap:anywhere}
nav a:hover{background:var(--sea-soft);color:var(--ink)}
main{padding:48px 32px 80px;min-width:0}
h1{font-size:36px;line-height:1.1;margin:0 0 20px}
h2{font-size:28px;margin:0 0 12px} h2 .sub{font-size:18px;font-weight:500;color:var(--ink-2);margin-left:8px}
h3{font-size:17px;margin:28px 0 10px}
.experiment{border-top:2px solid var(--rule);padding-top:32px;margin-top:56px}
.facts{display:grid;grid-template-columns:max-content 1fr;gap:6px 20px;margin:0 0 12px}
.facts dt{color:var(--ink-2)} .facts dd{margin:0}
.banner{background:var(--sand);border-left:6px solid var(--sea);border-radius:6px;padding:14px 18px;margin:0 0 16px;font-size:17px;max-width:110ch}
.banner.danger{background:var(--bad-soft);border-left-color:var(--bad);font-size:18px}
.valtag{background:var(--sand);color:var(--ink);padding:1px 6px;border-radius:3px}
.testtag{background:var(--bad-soft);color:var(--bad);font-weight:700}
.settag{text-align:left}
tr.testrow>*{background:var(--bad-soft)}
.note{color:var(--ink-2);max-width:100ch;font-size:15px}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%}
th,td{padding:8px 12px;text-align:right;white-space:nowrap;font-size:14px;overflow:hidden;text-overflow:ellipsis}
thead th{font-weight:600;font-size:13px;color:var(--ink-2);border-bottom:1px solid var(--rule);vertical-align:bottom;position:relative}
.grip{position:absolute;top:0;right:0;bottom:0;width:7px;cursor:col-resize;border-right:2px solid transparent}
.grip:hover{border-right-color:var(--sea)}
tbody th{text-align:left;font-weight:600} tbody tr+tr>*{border-top:1px solid var(--rule)}
td[style]{background:color-mix(in srgb,var(--sea-soft) calc((1 - var(--s)) * 100%),var(--sand) calc(var(--s) * 100%))}
.pm{color:var(--ink-2);font-size:12px}
.meanrow>*{font-weight:700;border-top:2px solid var(--rule)!important}
.failed{color:var(--bad);text-align:left;white-space:normal}
table.config{width:100%;background:var(--panel);border:1px solid var(--rule);border-radius:6px}
table.config th{width:200px}
table.config td{text-align:left;white-space:normal;overflow-wrap:anywhere}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-top:16px}
.chart{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 8px 4px;margin-top:12px}
.chart h3{font-size:15px;margin:0 8px 4px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.chart .plot{height:280px}
.chart.tall .plot{height:600px}
.chart.wide .plot{height:300px}
.chart.collapsed .plot{display:none}
button{font:inherit;padding:6px 14px;border-radius:4px;border:1px solid var(--sea);background:var(--panel);color:var(--sea);cursor:pointer}
button.mini{padding:1px 8px;font-size:12px;font-weight:500}
details{margin:12px 0} summary{cursor:pointer;font-weight:600;padding:4px 0}
.vectortools{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:8px 0}
.vectortools input,.vectortools select{font:inherit;padding:3px 6px}
.vcount,.ecount{color:var(--ink-2);font-size:14px}
.vgroups{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0;font-size:14px}
.vector{display:flex;flex-wrap:wrap;gap:4px;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:8px;max-height:520px;overflow:auto}
.vcell{display:flex;flex-direction:column;border:1px solid var(--rule);border-radius:3px;padding:2px 6px;font-size:12px;min-width:84px}
.vcell b{font-weight:600;color:var(--ink-2)} .vcell span{font-family:ui-monospace,Consolas,monospace}
.windows li{margin:4px 0;font-size:14px}
.filecharts .chart .plot{height:220px}
h4{font-size:15px;margin:16px 0 6px}
.modelgraph{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:6px 0}
.layer{display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:6px 10px;font-size:13px;min-width:120px}
.layer b{color:var(--sea)}
.arrow{color:var(--ink-2);font-size:18px}
.glossary dt{font-weight:600;margin-top:14px} .glossary dd{margin:2px 0 0;max-width:80ch;color:var(--ink-2)}
@media (max-width:860px){.layout{grid-template-columns:1fr}
  nav{position:static;height:auto;border-right:0;border-bottom:1px solid var(--rule);padding:48px 16px 16px}
  main{padding:24px 16px 60px}}
</style></head>
<body>
<button type="button" id="navtoggle">☰ Contents</button>
<div class="layout">
<nav aria-label="Sections"><ul>__NAV__<li><a href="#info">More info</a><ul>
  <li><a href="#glossary">What the metrics mean</a></li><li><a href="#info-inputs">How the inputs work</a></li>
  <li><a href="#info-rain">How rain works</a></li><li><a href="#info-training">How training works</a></li></ul></li></ul></nav>
<main>
<header id="overview">
  <h1>__TITLE__</h1>
  __BANNER__
  <dl class="facts">
    <dt>Experiments</dt><dd>__EXPERIMENTS__</dd>
    <dt>Repo version</dt><dd>__COMMITS__</dd>
    <dt>Report made</dt><dd>__GENERATED__</dd>
  </dl>
  <p class="note">Each lead time trains several models with different seeds; tables show their average (± standard deviation), and each experiment lists every model further down. Drag the edge of a table's column header to change its width. Every chart has a Hide/Show button.</p>
</header>

<section id="compare">
  <h2>Compare experiments</h2>
  __COMPARE__
</section>

<section id="datafiles" class="experiment">
  <h2>Data files</h2>
  <p class="note">Every data file the experiments in this report used, after loading: UTC times, rain spread to 6-minute steps (see "How rain works"). Missing = 6-minute steps with no value between the file's first and last time. Open a file below to plot it and see where its gaps are.</p>
  __DATA_FILES__
</section>

__SECTIONS__

<section id="info" class="experiment">
  <h2>More info</h2>
  <h3 id="glossary">What the metrics mean</h3>
  <dl class="glossary">
    <dt>RMSE (root mean squared error)</dt><dd>The typical size of an error. Large misses count extra, so a few big errors raise it a lot.</dd>
    <dt>MAE (mean absolute error)</dt><dd>The average size of an error, with every miss counted equally.</dd>
    <dt>Median absolute error</dt><dd>Half of all forecasts are closer than this.</dd>
    <dt>Within ±15 / ±5 / ±1 cm (central frequency)</dt><dd>The share of forecasts within that distance of the observed water level. The dashed line on the chart marks 90% at ±15 cm, the target commonly used for operational water level forecasts.</dd>
    <dt>R²</dt><dd>How much of the water level's variation the model explains. 1 is perfect, 0 is no better than always predicting the average.</dd>
    <dt>± (standard deviation)</dt><dd>How much the models of one lead time differ from each other. They differ only in their random seed.</dd>
    <dt>Lead time</dt><dd>How far ahead the forecast is. A 12 h lead forecast issued at 06:00 is for 18:00.</dd>
    <dt>Loss</dt><dd>Mean squared error in m², what the model minimizes while training. "Best val loss" is the lowest validation loss; that epoch's weights are the ones kept.</dd>
  </dl>

  <h3 id="info-inputs">How the inputs work</h3>
  <dl class="glossary">
    <dt>Times</dt><dd>Everything is in UTC, in 6-minute steps. A forecast is <em>issued</em> at a time t and is <em>for</em> t + lead. The input windows are measured from the issue time.</dd>
    <dt>Water level and Port O'Connor water level</dt><dd>Past only: from <code>lookback_hrs</code> before the issue time up to the issue time.</dd>
    <dt>Wind (u and v)</dt><dd>From <code>lookback_hrs</code> before the issue time up to the forecast time (lead hours after the issue time). The future part is <b>observed</b> wind standing in for a perfect wind forecast, so wind is always perfect prognosis.</dd>
    <dt>Gap fill</dt><dd>Optional, set separately for each data set (<code>train_gap_fill</code>, <code>val_gap_fill</code>, <code>test_gap_fill</code>). Fills gaps in water level, wind and Port O'Connor of up to that many hours by straight-line interpolation, only when there's a real value on both sides. Rain is never filled.</dd>
    <dt>Normalization</dt><dd>Optional (<code>normalize</code>). Rescales each input to mean 0 and standard deviation 1, using the training rows only. The scaling is saved inside the model. The target stays in meters, and the inputs shown in this report are the original values.</dd>
    <dt>Which forecasts are used</dt><dd>A forecast is only used when every one of its inputs and its target exist. Missing data near a gap removes every forecast that would use it, so an experiment with more inputs (e.g. rain) can have fewer rows. Experiments are then not scored on exactly the same forecasts; compare the row counts in the tables.</dd>
  </dl>

  <h3 id="info-rain">How rain works</h3>
  <dl class="glossary">
    <dt>Source</dt><dd>Hourly totals from the Iowa Environmental Mesonet (IEM) hourly precip tool, in inches, converted to mm. IEM timestamps are local Central time with daylight saving; they're converted to UTC, and the repeated 1 AM hour in November is dropped.</dd>
    <dt>Hourly to 6 minutes</dt><dd>Straight lines between the hourly values give a <em>shape</em>: which 6-minute steps in an hour get more rain and which get less. Each hour's 10 steps are then scaled so they add up to that hour's measured total. So every hourly total is kept exactly, and rain ramps smoothly within the hour. Each 6-minute step belongs to the hour it falls in (1:00 to 1:54 belong to the 1:00 value). Which stretch of time IEM's 1:00 value really covers is still being checked.</dd>
    <dt>Report delay</dt><dd>A 6-minute value also depends on the <em>next</em> hour's total (through the shape), and IEM hours end at about :53. So rain at a given time is only final about __RAIN_DELAY__ h later.</dd>
    <dt>Rain settings</dt><dd>
      <code>off</code>: no rain input.<br>
      <code>reported</code>: past rain up to __RAIN_DELAY__ h before the issue time, i.e. only rain that would already be published in real time.<br>
      <code>to_now</code>: past rain up to the issue time.<br>
      <code>perfprog</code>: perfect prognosis. Observed rain after the issue time stands in for a perfect rain forecast. It goes up to <code>rain_future_hrs</code> after the issue time, but never past the lead time and never past __RAIN_CAP__ h. With <code>rain_future_hrs</code> = 18: 6 h lead → 6 h of future rain, 12 h → 12 h, 18 h → 18 h, 24 h → 18 h, 48 h → 18 h.</dd>
    <dt>Why not past the lead time?</dt><dd>Rain that falls after the time a forecast is for can't change the water level at that time. It would only add inputs that are pure noise.</dd>
    <dt>Missing rain</dt><dd>Never filled. A missing hour removes every forecast whose rain window includes it.</dd>
  </dl>

  <h3 id="info-training">How training works</h3>
  <dl class="glossary">
    <dt>Model</dt><dd>A small neural network (MLP): the hidden layers, neurons, dropout and activation from the config, and one output (the water level at the forecast time). Trained with the Adam optimizer on mean squared error, using all training rows in every step.</dd>
    <dt>Early stopping</dt><dd>Training stops when the validation loss hasn't improved by at least <code>min_delta</code> for <code>patience</code> epochs (or at <code>epochs</code>). The weights from the epoch with the lowest validation loss are kept.</dd>
    <dt>Validation vs testing</dt><dd>The validation set picks when training stops, so the validation metrics are slightly optimistic. The testing set plays no part in training. Each model is evaluated on it after training, but those results only appear in the separate VAL+TEST report.</dd>
    <dt>Several models per lead</dt><dd>The n models of a lead are identical except for their random seed (starting weights and dropout). The tables show the mean ± standard deviation of their metrics. The time series shows the mean of their predictions. Note: the mean prediction usually scores a little better than the average model.</dd>
    <dt>Seeds</dt><dd>Each model's seed is set for Python (<code>random</code>), NumPy and TensorFlow, all three at once (<code>keras.utils.set_random_seed</code>). Model 1 uses the config's <code>seed</code>, model 2 uses seed + 1, and so on. The same seed gives the same model, except that a GPU can still give tiny differences between runs.</dd>
  </dl>
</section>
</main></div>
<script>
const DATA = __DATA__;
const css = getComputedStyle(document.documentElement);
const ink = css.getPropertyValue('--ink').trim();
const ink2 = css.getPropertyValue('--ink-2').trim();
const rule = css.getPropertyValue('--rule').trim();
const bad = css.getPropertyValue('--bad').trim();
const COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9', '#8C6D31', '#7F7F7F'];
const INPUT_COLORS = ['#8C6D31', '#7F7F7F', '#56B4E9', '#D55E00', '#CC79A7'];
const plotConfig = {displayModeBar: false, responsive: true};
const HOUR_MS = 3600 * 1000;
const SET_NAMES = {val: 'Validation', test: 'TESTING'};

// ---------------------------------------------------------------- page layout
document.getElementById('navtoggle').addEventListener('click', function () {
  document.body.classList.toggle('navhidden');
  window.dispatchEvent(new Event('resize'));  // lets the charts use the new width
});

// Every chart gets a Hide/Show button
function addHideButton(chart) {
  const title = chart.querySelector('h3');
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'mini';
  button.textContent = 'Hide';
  button.addEventListener('click', function () {
    chart.classList.toggle('collapsed');
    if (chart.classList.contains('collapsed')) {
      button.textContent = 'Show';
    } else {
      button.textContent = 'Hide';
      window.dispatchEvent(new Event('resize'));
    }
  });
  title.appendChild(button);
}
for (const chart of document.querySelectorAll('.chart')) { addHideButton(chart); }

// Table columns: drag the right edge of a header to resize
function makeResizable(table) {
  for (const th of table.querySelectorAll('thead th')) {
    const grip = document.createElement('span');
    grip.className = 'grip';
    th.appendChild(grip);
    grip.addEventListener('mousedown', function (event) {
      event.preventDefault();
      if (table.style.tableLayout !== 'fixed') {
        // the first time: freeze every column at its current width
        for (const h of table.querySelectorAll('thead th')) { h.style.width = h.offsetWidth + 'px'; }
        table.style.width = table.offsetWidth + 'px';
        table.style.tableLayout = 'fixed';
      }
      const startX = event.clientX;
      const startWidth = th.offsetWidth;
      const startTableWidth = table.offsetWidth;
      function move(e) {
        const width = Math.max(40, startWidth + e.clientX - startX);
        th.style.width = width + 'px';
        table.style.width = (startTableWidth + width - startWidth) + 'px';
      }
      function up() {
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
      }
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    });
  }
}
for (const table of document.querySelectorAll('table.resizable')) { makeResizable(table); }

// ---------------------------------------------------------------- helpers
function baseLayout(xTitle, yTitle) {
  return {margin: {l: 60, r: 12, t: 8, b: 40}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
          font: {family: 'Public Sans, system-ui, sans-serif', color: ink, size: 12},
          legend: {orientation: 'h', y: -0.25}, hovermode: 'x unified',
          xaxis: {title: {text: xTitle}, gridcolor: rule, zeroline: false},
          yaxis: {title: {text: yTitle}, gridcolor: rule, zeroline: false, exponentformat: 'none'}};
}

function times(list, factor) {
  const out = [];
  for (const v of list) {
    if (v === null) { out.push(null); } else { out.push(v * factor); }
  }
  return out;
}

function findExperiment(id) {
  for (const exp of DATA.experiments) {
    if (exp.id === id) { return exp; }
  }
  return null;
}

// UTC time <-> the text Plotly uses on date axes (no timezone, so the browser doesn't convert to local time)
function toAxisText(ms) { return new Date(ms).toISOString().slice(0, 16).replace('T', ' '); }
function fromAxisText(text) {
  let t = String(text).replace(' ', 'T');
  if (t.length === 10) { t = t + 'T00:00'; }
  return Date.parse(t.slice(0, 16) + ':00Z');
}

// The time axis of a stored series (made once, then reused)
const xCache = {};
function seriesX(id) {
  if (xCache[id] === undefined) {
    const s = DATA.series[id];
    const x = new Array(s.y.length);
    for (let i = 0; i < s.y.length; i++) { x[i] = toAxisText(s.start_ms + i * s.step_ms); }
    xCache[id] = x;
  }
  return xCache[id];
}

// One model's predictions = the mean + that model's difference from it (stored in whole mm)
function modelPrediction(mean, offsets) {
  const out = new Array(mean.length);
  for (let i = 0; i < mean.length; i++) {
    if (mean[i] === null || offsets[i] === null) { out[i] = null; } else { out[i] = mean[i] + offsets[i] / 1000; }
  }
  return out;
}

// -24 -> '-24 h', 6 -> '+6 h'
function signedHours(h) {
  if (h > 0) { return '+' + h + ' h'; }
  return h + ' h';
}

// Log-axis ticks at 1, 2, 5 × 10^k, labelled with full numbers (0.001, 0.002, 0.005, ...)
function logTicks(curves) {
  let low = Infinity;
  let high = 0;
  for (const c of curves) {
    for (const v of c.loss.concat(c.val)) {
      if (v > 0 && v < low) { low = v; }
      if (v > high) { high = v; }
    }
  }
  const values = [];
  const labels = [];
  if (high === 0) { return {values: values, labels: labels}; }
  for (let k = Math.floor(Math.log10(low)); k <= Math.ceil(Math.log10(high)); k++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, k);
      values.push(v);
      labels.push(String(Number(v.toPrecision(1))));
    }
  }
  return {values: values, labels: labels};
}

// ---------------------------------------------------------------- error-vs-lead charts (testing = dashed lines)
function leadLine(exp, which, name, key, factor, color) {
  let dash = 'solid';
  let label = name;
  if (which === 'test') { dash = 'dash'; label = name + ' (TEST)'; }
  return {x: exp.leads, y: times(exp.mean[which][key], factor), name: label, mode: 'lines+markers',
          line: {color: color, width: 2, dash: dash}, marker: {size: 7},
          error_y: {type: 'data', array: times(exp.std[which][key], factor), visible: true, thickness: 1}};
}

function leadLines(exp, items, factor) {
  const traces = [];
  for (const which of exp.sets) {
    for (const item of items) { traces.push(leadLine(exp, which, item[0], item[1], factor, item[2])); }
  }
  return traces;
}

for (const exp of DATA.experiments) {
  let layout = baseLayout('Lead time (h)', 'cm');
  layout.xaxis.tickvals = exp.leads;
  Plotly.newPlot('err-' + exp.id, leadLines(exp, [['RMSE', 'rmse', COLORS[0]], ['MAE', 'mae', COLORS[1]], ['Median', 'medae', COLORS[2]]], 100),
                 layout, plotConfig);

  layout = baseLayout('Lead time (h)', '%');
  layout.xaxis.tickvals = exp.leads;
  layout.yaxis.range = [0, 100];
  layout.shapes = [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 90, y1: 90, line: {color: ink2, width: 1, dash: 'dash'}}];
  Plotly.newPlot('cf-' + exp.id, leadLines(exp, [['±15 cm', 'cf_15cm', COLORS[0]], ['±5 cm', 'cf_5cm', COLORS[1]], ['±1 cm', 'cf_1cm', COLORS[2]]], 1),
                 layout, plotConfig);

  layout = baseLayout('Lead time (h)', 'R²');
  layout.xaxis.tickvals = exp.leads;
  Plotly.newPlot('r2-' + exp.id, leadLines(exp, [['R²', 'r2', COLORS[3]]], 1), layout, plotConfig);
}

// ---------------------------------------------------------------- input vector viewer
for (const box of document.querySelectorAll('details.inputs')) {
  const exp = findExperiment(box.dataset.exp);
  const leadSelect = box.querySelector('.vlead');
  const filterBox = box.querySelector('.vfilter');
  const groupsBox = box.querySelector('.vgroups');
  const vectorBox = box.querySelector('.vector');
  const countBox = box.querySelector('.vcount');
  let hiddenGroups = {};

  function draw() {
    const v = exp.vectors[leadSelect.value];
    vectorBox.textContent = '';
    if (v === null) { vectorBox.textContent = 'No example saved for this lead.'; return; }
    const filter = filterBox.value.trim().toLowerCase();
    let shown = 0;
    const cells = document.createDocumentFragment();
    for (let i = 0; i < v.columns.length; i++) {
      if (hiddenGroups[v.groups[v.group_of[i]].label]) { continue; }
      if (filter !== '' && v.columns[i].toLowerCase().indexOf(filter) === -1) { continue; }
      const cell = document.createElement('div');
      cell.className = 'vcell';
      const name = document.createElement('b');
      name.textContent = v.columns[i];
      const value = document.createElement('span');
      value.textContent = v.values[i];
      cell.appendChild(name);
      cell.appendChild(value);
      cells.appendChild(cell);
      shown += 1;
    }
    vectorBox.appendChild(cells);
    countBox.textContent = 'Showing ' + shown + ' of ' + v.columns.length + ' inputs. Example issue time: ' + v.issue_time + ' UTC';
  }

  function drawGroups() {
    const v = exp.vectors[leadSelect.value];
    groupsBox.textContent = '';
    if (v === null) { return; }
    for (const group of v.groups) {
      const label = document.createElement('label');
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.checked = !hiddenGroups[group.label];
      check.addEventListener('change', function () {
        hiddenGroups[group.label] = !check.checked;
        draw();
      });
      label.appendChild(check);
      label.appendChild(document.createTextNode(' ' + group.label));
      groupsBox.appendChild(label);
    }
  }

  leadSelect.addEventListener('change', function () { drawGroups(); draw(); });
  filterBox.addEventListener('input', draw);
  box.addEventListener('toggle', function () {
    if (box.open && !box.dataset.drawn) { box.dataset.drawn = 'yes'; drawGroups(); draw(); }
  });
}

// ---------------------------------------------------------------- time series (+ the input-window hover tool)
const WINDOW_ICON = {width: 1000, height: 1000, path: 'M100 250 H900 V750 H100 Z M180 330 V670 H820 V330 Z'};

function drawTimeSeries(exp, which, box) {
  const ts = exp.ts[which];
  const x = seriesX(ts.observed);
  const traces = [{type: 'scattergl', mode: 'lines', name: 'Observed', x: x, y: DATA.series[ts.observed].y,
                   line: {color: ink, width: 1.6}, hovertemplate: '%{y:.3f} m<extra>Observed</extra>'}];
  let i = 0;
  for (const lead of exp.leads) {
    const mean = ts.preds[String(lead)];
    if (mean === undefined) { continue; }
    const color = COLORS[i % COLORS.length];
    const group = 'lead' + lead;
    const groupTitle = {text: lead + ' h lead'};
    const name = lead + ' h: mean of models';
    traces.push({type: 'scattergl', mode: 'lines', name: name, x: x, y: mean, meta: lead, legendgroup: group,
                 legendgrouptitle: groupTitle, line: {color: color, width: 1.2},
                 hovertemplate: '%{y:.3f} m<extra>' + name + '</extra>'});

    // min-max of the models: the lowest and highest prediction at each time, as one trace (two lines)
    const models = ts.models[String(lead)];
    const low = new Array(mean.length);
    const high = new Array(mean.length);
    for (let t = 0; t < mean.length; t++) {
      low[t] = null;
      high[t] = null;
      for (const offsets of models) {
        if (mean[t] === null || offsets[t] === null) { continue; }
        const v = mean[t] + offsets[t] / 1000;
        if (low[t] === null || v < low[t]) { low[t] = v; }
        if (high[t] === null || v > high[t]) { high[t] = v; }
      }
    }
    traces.push({type: 'scattergl', mode: 'lines', name: lead + ' h: min–max of models', x: x.concat([null], x),
                 y: low.concat([null], high), meta: lead, legendgroup: group, visible: 'legendonly',
                 line: {color: color, width: 0.8, dash: 'dot'}, hovertemplate: '%{y:.3f} m<extra>' + lead + ' h min/max</extra>'});

    for (let k = 0; k < models.length; k++) {
      const modelName = lead + ' h: model ' + (k + 1);
      traces.push({type: 'scattergl', mode: 'lines', name: modelName, x: x, y: modelPrediction(mean, models[k]),
                   meta: lead, legendgroup: group, visible: 'legendonly', line: {color: color, width: 0.8},
                   opacity: 0.7, hovertemplate: '%{y:.3f} m<extra>' + modelName + '</extra>'});
    }
    i += 1;
  }
  for (let j = 0; j < ts.inputs.length; j++) {
    const input = ts.inputs[j];
    traces.push({type: 'scattergl', mode: 'lines', name: 'Input: ' + input.name, x: seriesX(input.id),
                 y: DATA.series[input.id].y, yaxis: 'y2', visible: 'legendonly', legendgroup: 'inputs',
                 legendgrouptitle: {text: 'Inputs'}, line: {color: INPUT_COLORS[j % INPUT_COLORS.length], width: 1, dash: 'dot'},
                 hovertemplate: '%{y:.3f}<extra>' + input.name + '</extra>'});
  }

  const layout = baseLayout('Time (UTC)', 'Water level (m)');
  layout.xaxis.type = 'date';
  layout.yaxis2 = {title: {text: 'Inputs'}, overlaying: 'y', side: 'right', showgrid: false, zeroline: false};
  layout.margin.r = 60;
  layout.margin.t = 10;
  layout.legend = {orientation: 'v', x: 1.06, y: 1, groupclick: 'toggleitem'};

  const status = document.createElement('p');
  status.className = 'note';
  box.parentNode.parentNode.insertBefore(status, box.parentNode);

  let windowMode = false;
  const windowButton = {
    name: 'Input window: shade the data used by the forecast you hover',
    icon: WINDOW_ICON,
    click: function () {
      windowMode = !windowMode;
      if (windowMode) {
        Plotly.relayout(box, {hovermode: 'closest'});
        status.textContent = 'Input window: ON. Hover a lead line.';
      } else {
        Plotly.relayout(box, {hovermode: 'x unified', shapes: [], annotations: []});
        status.textContent = '';
      }
    }
  };

  Plotly.newPlot(box, traces, layout, {responsive: true, displaylogo: false, modeBarButtonsToAdd: [windowButton]});

  box.on('plotly_hover', function (event) {
    if (!windowMode) { return; }
    let point = null;
    for (const p of event.points) {
      if (p.data.meta !== undefined) { point = p; break; }
    }
    if (point === null) { return; }
    const lead = point.data.meta;
    const validMs = fromAxisText(point.x);
    const issueMs = validMs - lead * HOUR_MS;
    const windows = exp.windows[String(lead)];
    const shapes = [];
    const annotations = [];
    const bandHeight = 0.8 / windows.length;
    for (let k = 0; k < windows.length; k++) {
      const w = windows[k];
      const top = 1 - k * bandHeight;
      shapes.push({type: 'rect', xref: 'x', yref: 'paper', x0: toAxisText(issueMs + w.start_hrs * HOUR_MS),
                   x1: toAxisText(issueMs + w.end_hrs * HOUR_MS), y0: top - bandHeight, y1: top,
                   fillcolor: INPUT_COLORS[k % INPUT_COLORS.length], opacity: 0.2, line: {width: 0}, layer: 'below'});
      annotations.push({xref: 'x', yref: 'paper', x: toAxisText(issueMs + w.start_hrs * HOUR_MS), y: top,
                        xanchor: 'left', yanchor: 'top', showarrow: false, font: {size: 11, color: ink2},
                        text: w.label + ' (' + signedHours(w.start_hrs) + ' to ' + signedHours(w.end_hrs) + ')'});
    }
    shapes.push({type: 'line', xref: 'x', yref: 'paper', x0: toAxisText(issueMs), x1: toAxisText(issueMs), y0: 0, y1: 1,
                 line: {color: ink2, width: 1, dash: 'dash'}});
    shapes.push({type: 'line', xref: 'x', yref: 'paper', x0: toAxisText(validMs), x1: toAxisText(validMs), y0: 0, y1: 1,
                 line: {color: ink, width: 1.5}});
    annotations.push({xref: 'x', yref: 'paper', x: toAxisText(issueMs), y: 0, yanchor: 'bottom', showarrow: false,
                      text: 'issued', font: {size: 11, color: ink2}});
    annotations.push({xref: 'x', yref: 'paper', x: toAxisText(validMs), y: 0, yanchor: 'bottom', showarrow: false,
                      text: 'forecast for (' + lead + ' h later)', font: {size: 11, color: ink}});
    Plotly.relayout(box, {shapes: shapes, annotations: annotations});
  });
}

for (const button of document.querySelectorAll('.tsbutton')) {
  button.addEventListener('click', function () {
    const exp = findExperiment(button.dataset.exp);
    const which = button.dataset.set;
    const box = document.getElementById('ts-' + exp.id + '-' + which);
    const chart = box.parentNode;
    if (exp.ts[which] === null) { button.textContent = 'Nothing to show for this set'; button.disabled = true; return; }
    button.remove();
    chart.hidden = false;
    drawTimeSeries(exp, which, box);
  });
}

// ---------------------------------------------------------------- error plots and the worst forecasts
for (const box of document.querySelectorAll('details.errors')) {
  const exp = findExperiment(box.dataset.exp);
  const leadSelect = box.querySelector('.elead');
  const setSelect = box.querySelector('.eset');
  const modelSelect = box.querySelector('.emodel');
  const countBox = box.querySelector('.ecount');
  const worstBody = box.querySelector('table.worst tbody');

  function draw() {
    const which = setSelect.value;
    const lead = Number(leadSelect.value);
    const ts = exp.ts[which];
    worstBody.textContent = '';
    if (ts === null || ts.preds[String(lead)] === undefined) {
      countBox.textContent = 'Nothing to show for this lead and set.';
      return;
    }
    const x = seriesX(ts.observed);
    const observed = DATA.series[ts.observed].y;
    let predicted = ts.preds[String(lead)];
    let label = 'mean of the models';
    if (modelSelect.value !== 'mean') {
      const k = Number(modelSelect.value);
      if (k >= ts.models[String(lead)].length) { countBox.textContent = 'That model has no results.'; return; }
      predicted = modelPrediction(predicted, ts.models[String(lead)][k]);
      label = 'model ' + (k + 1);
    }

    const validTimes = [];
    const errors = [];
    const obsList = [];
    const predList = [];
    const index = [];
    for (let i = 0; i < x.length; i++) {
      if (observed[i] === null || predicted[i] === null) { continue; }
      validTimes.push(x[i]);
      errors.push((predicted[i] - observed[i]) * 100);
      obsList.push(observed[i]);
      predList.push(predicted[i]);
      index.push(i);
    }
    let setText = 'validation set';
    if (which === 'test') { setText = 'TESTING set'; }
    countBox.textContent = errors.length + ' forecasts, ' + lead + ' h lead, ' + label + ', ' + setText;

    let layout = baseLayout('Time the forecast is for (UTC)', 'Error (cm)');
    layout.xaxis.type = 'date';
    layout.showlegend = false;
    Plotly.newPlot('etime-' + exp.id, [{type: 'scattergl', mode: 'markers', x: validTimes, y: errors, marker: {size: 2, color: COLORS[0]},
                                        hovertemplate: '%{y:.1f} cm<extra></extra>'}], layout, plotConfig);

    layout = baseLayout('Error (cm)', 'Number of forecasts');
    layout.showlegend = false;
    layout.hovermode = 'closest';
    Plotly.newPlot('ehist-' + exp.id, [{type: 'histogram', x: errors, nbinsx: 80, marker: {color: COLORS[0]}}], layout, plotConfig);

    let low = Infinity;
    let high = -Infinity;
    for (const v of obsList.concat(predList)) {
      if (v < low) { low = v; }
      if (v > high) { high = v; }
    }
    layout = baseLayout('Observed (m)', 'Predicted (m)');
    layout.showlegend = false;
    layout.hovermode = 'closest';
    Plotly.newPlot('escatter-' + exp.id, [
      {type: 'scattergl', mode: 'markers', x: obsList, y: predList, marker: {size: 2, color: COLORS[0], opacity: 0.3},
       hovertemplate: 'observed %{x:.3f} m<br>predicted %{y:.3f} m<extra></extra>'},
      {type: 'scatter', mode: 'lines', x: [low, high], y: [low, high], line: {color: bad, dash: 'dash', width: 1}, hoverinfo: 'skip'}
    ], layout, plotConfig);

    // The 20 largest misses
    const order = [];
    for (let i = 0; i < errors.length; i++) { order.push(i); }
    order.sort(function (a, b) { return Math.abs(errors[b]) - Math.abs(errors[a]); });
    for (let r = 0; r < Math.min(20, order.length); r++) {
      const i = order[r];
      const validMs = fromAxisText(validTimes[i]);
      const tr = document.createElement('tr');
      const cells = [toAxisText(validMs - lead * HOUR_MS), validTimes[i], obsList[i].toFixed(3), predList[i].toFixed(3), errors[i].toFixed(1)];
      for (const text of cells) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.appendChild(td);
      }
      worstBody.appendChild(tr);
    }
  }

  leadSelect.addEventListener('change', draw);
  setSelect.addEventListener('change', draw);
  modelSelect.addEventListener('change', draw);
  box.addEventListener('toggle', function () {
    if (box.open && !box.dataset.drawn) { box.dataset.drawn = 'yes'; draw(); }
  });
}

// ---------------------------------------------------------------- data files: plots and gap timelines
for (const box of document.querySelectorAll('details.datafile')) {
  box.addEventListener('toggle', function () {
    if (!box.open || box.dataset.drawn) { return; }
    box.dataset.drawn = 'yes';
    const i = Number(box.dataset.file);
    const file = DATA.files[i];

    // Gap timeline: one row per column, a thick bar for every gap
    const gapTraces = [];
    for (let c = 0; c < file.columns.length; c++) {
      const column = file.columns[c];
      const x = [];
      const y = [];
      for (const gap of column.gaps) {
        x.push(toAxisText(gap[0]), toAxisText(gap[1] + 6 * 60 * 1000), null);
        y.push(column.label, column.label, null);
      }
      if (x.length === 0) { x.push(null); y.push(column.label); }
      gapTraces.push({type: 'scatter', mode: 'lines', x: x, y: y, name: column.label, line: {color: bad, width: 14},
                      hovertemplate: '%{x}<extra>gap</extra>'});
    }
    const gapLayout = baseLayout('Time (UTC)', '');
    gapLayout.xaxis.type = 'date';
    gapLayout.showlegend = false;
    gapLayout.hovermode = 'closest';
    gapLayout.margin.l = 200;
    Plotly.newPlot('gaps-' + i, gapTraces, gapLayout, plotConfig);

    // One chart per column
    const holder = document.getElementById('filecharts-' + i);
    for (let c = 0; c < file.columns.length; c++) {
      const column = file.columns[c];
      const chart = document.createElement('div');
      chart.className = 'chart';
      const title = document.createElement('h3');
      title.textContent = column.label;
      const plot = document.createElement('div');
      plot.className = 'plot';
      chart.appendChild(title);
      chart.appendChild(plot);
      holder.appendChild(chart);
      addHideButton(chart);
      const layout = baseLayout('Time (UTC)', column.label);
      layout.xaxis.type = 'date';
      layout.showlegend = false;
      Plotly.newPlot(plot, [{type: 'scattergl', mode: 'lines', x: seriesX(column.id), y: DATA.series[column.id].y,
                             line: {color: COLORS[c % COLORS.length], width: 1}, hovertemplate: '%{y:.3f}<extra></extra>'}],
                     layout, {responsive: true, displaylogo: false});
    }
  });
}

// ---------------------------------------------------------------- loss curves (drawn when a lead is opened)
for (const details of document.querySelectorAll('details.lead')) {
  details.addEventListener('toggle', function () {
    if (!details.open || details.dataset.drawn) { return; }
    details.dataset.drawn = 'yes';
    const exp = findExperiment(details.dataset.exp);
    const curves = exp.loss[details.dataset.lead];
    const traces = [];
    for (let i = 0; i < curves.length; i++) {
      const color = COLORS[i % COLORS.length];
      traces.push({y: curves[i].val, name: curves[i].name + ' validation', mode: 'lines', line: {color: color, width: 1.6}});
      traces.push({y: curves[i].loss, name: curves[i].name + ' training', mode: 'lines', line: {color: color, width: 1, dash: 'dash'}});
    }
    const id = exp.id + '-' + details.dataset.lead;
    Plotly.newPlot('loss-' + id, traces, baseLayout('Epoch', 'Loss (MSE)'), plotConfig);

    const logLayout = baseLayout('Epoch', 'Loss (MSE)');
    const ticks = logTicks(curves);
    logLayout.yaxis.type = 'log';
    logLayout.yaxis.tickvals = ticks.values;
    logLayout.yaxis.ticktext = ticks.labels;
    Plotly.newPlot('losslog-' + id, traces, logLayout, plotConfig);

    const timeTraces = [];
    for (let i = 0; i < curves.length; i++) {
      if (curves[i].seconds.length === 0) { continue; }
      timeTraces.push({y: curves[i].seconds, name: curves[i].name, mode: 'lines', line: {color: COLORS[i % COLORS.length], width: 1}});
    }
    const timeBox = document.getElementById('epochtime-' + id);
    if (timeTraces.length === 0) {
      timeBox.textContent = 'Per-epoch times not saved for this session (trained with an older notebook).';
    } else {
      Plotly.newPlot(timeBox, timeTraces, baseLayout('Epoch', 'Seconds'), plotConfig);
    }
  });
}
</script></body></html>
"""


# =====================================================================================================
# 7. Training function
# =====================================================================================================
# Trains, evaluates and saves one model. The notebook's training cell calls this once per model.

import time
import queue
import threading

# ---------------------------------------------------------------- saving in the background
# Saving (zips, HTML pages, copies to Drive) happens in a second thread, so training doesn't wait for it.
# Matplotlib isn't safe to use from two threads, so the PNG plots are still made by the training loop itself.

# Zips and HTML copies are also kept here, in the Colab session (visible in the Files panel)
if Path('/content').exists():
    LOCAL_SAVE_DIR = Path('/content/wl_saves')
else:
    LOCAL_SAVE_DIR = Path(tempfile.gettempdir()) / 'wl_saves'
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)


def save_worker():
    """Runs in the background thread: does each queued save, one at a time"""
    while True:
        name, function, args = save_queue.get()
        try:
            function(*args)
        except Exception:
            traceback.print_exc()
            print(f"⚠ Background save failed: {name}. Training continues.")
        save_queue.task_done()


if 'save_thread' not in globals() or not save_thread.is_alive():
    save_queue = queue.Queue()
    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()


def save_in_background(name, function, *args):
    """Queues a save for the background thread"""
    save_queue.put((name, function, args))


def wait_for_saves():
    """Waits until every queued save is finished"""
    save_queue.join()


def save_everywhere(file_path, session_id, download=False):
    """Keeps a copy of a finished file (zip or HTML) in the Colab session and in Drive.
    download=True also tries a browser download (only done at the very end: Colab holds downloads back while a cell runs)"""
    file_path = Path(file_path)
    local_copy = LOCAL_SAVE_DIR / session_id / file_path.name
    os.makedirs(local_copy.parent, exist_ok=True)
    if file_path != local_copy:
        shutil.copy(file_path, local_copy)
    if DRIVE_OK:
        drive_copy = Path(OUT_DIR) / 'saves' / session_id / file_path.name
        os.makedirs(drive_copy.parent, exist_ok=True)
        if file_path != drive_copy:
            shutil.copy(file_path, drive_copy)
    if download:
        try_download(str(local_copy))


def zip_and_save(folder, zip_name, session_id, download=False):
    """Zips a folder (in the Colab session), then copies it to Drive (and downloads it when download=True)"""
    zip_base = LOCAL_SAVE_DIR / session_id / zip_name
    os.makedirs(zip_base.parent, exist_ok=True)
    zip_path = shutil.make_archive(str(zip_base), 'zip', folder)
    save_everywhere(zip_path, session_id, download)


def report_and_save(session_dir, experiment_name, session_id):
    """Builds a report (one experiment, or the whole session when experiment_name is None) and saves it everywhere"""
    report_path = build_session_report(session_dir, experiment_name)
    save_everywhere(report_path, session_id)
    print(f"  Saved report: {report_path.name}")


# ---------------------------------------------------------------- one model

class EpochTimer(keras.callbacks.Callback):
    """Records how many seconds each epoch took (training + validation)"""
    def on_train_begin(self, logs=None):
        self.seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(round(time.perf_counter() - self.epoch_start, 4))


def describe_model(model, n_inputs):
    """The model's structure, layer by layer (type, size, activation, dropout, output shape, parameters), for the report"""
    layers = [{'type': 'Input', 'detail': f"{n_inputs:,} values", 'output': f"({n_inputs},)", 'params': 0}]
    total_params = 0
    for layer in model.layers:
        config = layer.get_config()
        if 'units' in config:
            detail = f"{config['units']} neurons, {config['activation']}"
        elif 'rate' in config:
            detail = f"drops {config['rate']:g} of values while training"
        elif layer.__class__.__name__ == 'Normalization':
            detail = "rescales each input using the training rows"
        else:
            detail = ''
        try:
            output = str(tuple(layer.output.shape))
        except Exception:
            output = ''
        # Trainable only: the scaling values in a Normalization layer aren't learned
        params = 0
        for weight in layer.trainable_weights:
            params += int(np.prod(weight.shape))
        total_params += params
        layers.append({'type': layer.__class__.__name__, 'detail': detail, 'output': output, 'params': params})
    return {'layers': layers, 'total_params': total_params}


def save_predictions_csv(path, datetime_index, lead_hrs, observed, predicted):
    """One row per forecast: when it was issued, the time it's for, observed and predicted water level"""
    pd.DataFrame({'issue_time_utc': datetime_index,
                  'valid_time_utc': datetime_index + pd.Timedelta(hours=lead_hrs),
                  'observed_m': np.round(observed, 4),
                  'predicted_m': np.round(predicted, 4)}).to_csv(path, index=False)


def save_checkpoint_html(path, run_name, datetime_index, lead_hrs, observed, predicted, metrics, session_id):
    """Small interactive observed-vs-predicted page for one model, saved in the session folder and in the
    Colab session (kept only until the final report exists)"""
    valid_time = datetime_index + pd.Timedelta(hours=lead_hrs)
    predictions_df = pd.DataFrame({'data': predicted}, index = valid_time)
    predictions_df['dash_style'] = 'dash'
    predictions_df['legend_label'] = 'Predicted'
    predictions_df['line_color'] = 'orange'
    observed_df = pd.DataFrame({'data': observed}, index = valid_time)
    observed_df['dash_style'] = 'solid'
    observed_df['legend_label'] = 'Observed'
    observed_df['line_color'] = 'blue'
    title = f"{run_name}: RMSE {metrics['rmse'] * 100:.1f} cm, within ±15 cm {metrics['cf_15cm']:.1f}%"
    plot_interactive_time_series([predictions_df, observed_df], title, 'Valid time (UTC)',
                                 'Elevation in meters (NAVD88)', path, show = False)
    local_copy = LOCAL_SAVE_DIR / session_id / 'checkpoints' / Path(path).name
    os.makedirs(local_copy.parent, exist_ok=True)
    if Path(path) != local_copy:
        shutil.copy(path, local_copy)


def save_test_results(model, test, cfg, run_name, run_dir, row):
    """Evaluates a model on the TESTING set and saves the results in their own files. They are never printed, and only
    the VAL+TEST report shows them, so nobody sees testing results by accident while still choosing settings."""
    if len(test['target_array']) == 0:
        return
    metrics, predictions = evaluate_model(model, test['input_array'], test['target_array'], verbose = False)
    save_predictions_csv(run_dir / f"{run_name}_test_predictions.csv", test['datetime_index'], cfg['lead_hrs'],
                         test['target_array'], predictions)
    test_row = {'session': row['session'], 'experiment': row['experiment'], 'run_name': run_name,
                'lead_hrs': cfg['lead_hrs'], 'model_num': row['model_num'], 'test_rows': len(test['target_array']),
                'test_files': ';'.join(cfg['test_files'])}
    for name in metrics:
        test_row[name] = metrics[name]
    pd.DataFrame([test_row]).to_csv(run_dir / f"{run_name}_test_results.csv", index=False)
    append_csv(test_row, f"{OUT_DIR}/all_test_results.csv")


def train_one_model(experiment, cfg, model_num, train, val, test, session_id, session_dir, data_fingerprints):
    """Trains, evaluates and saves one model. Returns its results row (status 'ok' or 'FAILED: ...').
    The testing set is evaluated too, but its results go in separate files (see save_test_results)."""
    run_name = make_run_name(cfg, model_num)
    seed = cfg['seed'] + model_num - 1
    run_dir = Path(session_dir) / experiment['name'] / f"{cfg['lead_hrs']}h" / f"m{model_num}"
    os.makedirs(run_dir, exist_ok=True)

    # Everything about this model goes in its results row, so a result can never be mixed up with another config
    row = {'session': session_id, 'experiment': experiment['name'], 'run_name': run_name,
           'location': cfg['location'], 'lead_hrs': cfg['lead_hrs'], 'model_num': model_num, 'seed_used': seed}
    for name in cfg:
        if name not in row:
            value = cfg[name]
            if isinstance(value, list):
                value = ';'.join(value)
            row[name] = value
    row['config_file'] = experiment['file']
    row['config_fingerprint'] = experiment['fingerprint']
    row['data_fingerprints'] = json.dumps(data_fingerprints)
    row['commit'] = COMMIT

    try:
        x_train = train['input_array']
        y_train = train['target_array']
        x_val = val['input_array']
        y_val = val['target_array']
        row['train_rows'] = len(x_train)
        row['val_rows'] = len(x_val)
        row['test_rows'] = len(test['target_array'])
        row['n_inputs'] = x_train.shape[1]

        # One example input vector (the middle validation row), shown in the report
        example_index = len(x_val) // 2
        example_values = []
        for value in x_val[example_index]:
            example_values.append(round(float(value), 4))
        example = {'issue_time_utc': str(val['datetime_index'][example_index]), 'values': example_values}

        # Save everything needed to rebuild this model's inputs later (used by load_run and the report)
        saved = {'run_name': run_name, 'session': session_id, 'commit': COMMIT, 'model_num': model_num,
                 'seed': seed, 'cfg': cfg, 'experiment_settings': experiment['settings'],
                 'config_file': experiment['file'], 'config_fingerprint': experiment['fingerprint'],
                 'data_fingerprints': data_fingerprints, 'input_columns': train['input_columns'],
                 'example_input': example}

        keras.backend.clear_session()      # fresh model every time
        keras.utils.set_random_seed(seed)  # sets the Python (random), NumPy and TensorFlow seeds: same seed = repeatable results

        model = build_model(cfg, x_train)
        saved['architecture'] = describe_model(model, x_train.shape[1])
        with open(run_dir / f"{run_name}_config.json", 'w') as fh:
            json.dump(saved, fh, indent=2)

        epoch_timer = EpochTimer()
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = cfg['patience'], min_delta = cfg['min_delta'],
                                       restore_best_weights = True)
        # Checkpoint to the local disk while training (fast), then copy the best model to the run folder once
        local_model_file = os.path.join(tempfile.gettempdir(), f"tmp_{run_name}.keras")
        checkpoint = ModelCheckpoint(local_model_file, monitor = 'val_loss', save_best_only = True, mode = 'min')

        start = time.perf_counter()
        model_history = model.fit(x_train, y_train,
                                  validation_data = (x_val, y_val),
                                  epochs = cfg['epochs'],
                                  batch_size = len(x_train),
                                  validation_batch_size = len(x_val),
                                  callbacks = [early_stopping, checkpoint, epoch_timer],
                                  verbose = 0)
        train_seconds = time.perf_counter() - start
        shutil.copy(local_model_file, run_dir / f"{run_name}_model.keras")
        history = pd.DataFrame(model_history.history)
        history['epoch_seconds'] = epoch_timer.seconds
        history.to_csv(run_dir / f"{run_name}_history.csv", index_label='epoch')

        # ---- Metrics on the validation set ----
        metrics, predictions = evaluate_model(model, x_val, y_val, verbose = False)

        # ---- Testing set: saved separately, never printed ----
        save_test_results(model, test, cfg, run_name, run_dir, row)

        # ---- Plots and predictions ----
        start_saving = time.perf_counter()
        plot_model_loss(model_history, run_dir / f"{run_name}_loss.png", legend_location = 'upper right', show = False)
        plot_model_predictions(y_val, predictions, f'{run_name}: Observed vs Predicted', 'Index',
                               'Elevation in meters (NAVD88)', run_dir / f"{run_name}_obsvspred.png", show = False)
        save_predictions_csv(run_dir / f"{run_name}_predictions.csv", val['datetime_index'], cfg['lead_hrs'],
                             y_val, predictions)
        checkpoint_dir = Path(session_dir) / 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_in_background(f"checkpoint page {run_name}", save_checkpoint_html, checkpoint_dir / f"{run_name}.html",
                           run_name, val['datetime_index'], cfg['lead_hrs'], y_val, predictions, metrics, session_id)
        saving_seconds = time.perf_counter() - start_saving

        row['status'] = 'ok'
        row['best_epoch'] = int(np.argmin(model_history.history['val_loss']))
        row['epochs_trained'] = len(model_history.history['loss'])
        row['best_val_loss'] = float(model_history.history['val_loss'][row['best_epoch']])
        row['train_loss_at_best'] = float(model_history.history['loss'][row['best_epoch']])
        row['train_seconds'] = round(train_seconds, 1)
        row['seconds_per_epoch'] = round(train_seconds / row['epochs_trained'], 4)
        row['saving_seconds'] = round(saving_seconds, 1)
        for name in metrics:
            row[name] = metrics[name]
        print(f"  {run_name}: RMSE {metrics['rmse'] * 100:.1f} cm | within ±15 cm {metrics['cf_15cm']:.1f}% | "
              f"best epoch {row['best_epoch']} of {row['epochs_trained']} | "
              f"trained in {train_seconds / 60:.1f} min ({row['seconds_per_epoch']:.3f} s/epoch), plots {saving_seconds:.0f} s")
        append_csv(row, f"{OUT_DIR}/all_results.csv")  # add to the master CSV right away

    except Exception as e:
        # One broken model shouldn't stop the others
        traceback.print_exc()
        print(f"⚠ {run_name} FAILED, moving on.")
        row['status'] = f"FAILED: {e}"
        append_csv(row, f"{OUT_DIR}/failed_runs.csv")

    pd.DataFrame([row]).to_csv(run_dir / f"{run_name}_results.csv", index=False)
    return row
