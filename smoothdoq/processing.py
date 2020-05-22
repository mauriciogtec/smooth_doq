import numpy as np
import pandas as pd
import json


def parse_raw_counts(
        raw_string: str,
        dim: str,
        zero_spike: bool=True,
        trim_lower_inf: bool=True,
        trim_upper_inf: bool=True,
        subtract: int=0,
        use_log_bins: bool=True,
        min_bins: int=1,
        min_counts: int=50,
        min_counts_to_bins_ratio: int=2) -> pd.DataFrame:
        
    s = pd.Series(raw_string[:-4].split(","))
    s = s.str.slice(3)
    s = s.str.split(":")

    # extract counts and bins
    lower = s.apply(lambda x: np.float32(x[0]))
    upper = s.apply(lambda x: np.float32(x[1][:-2]))
    count = s.apply(lambda x: int(np.float32(x[2])))
    df = pd.DataFrame({'lower': lower,
                       'upper': upper,
                       'counts': count})

    flags = []

    if subtract > 0:
        df['counts'] = np.maximum(df['counts'] - subtract, 0)
        if df['counts'].sum() == 0:
            flags.append('zero_sum')

    if zero_spike and dim != "TIME":
        if (np.argmax(df['counts'].values) == 0):
            flags.append("zero_spike")

    if trim_lower_inf:
        if np.isinf(df['lower'].values[0]):
            flags.append("trim_ninf")
            df['lower'].values[0] = df['upper'].values[0]
        
    if trim_upper_inf:
        if np.isinf(df['upper'].values[-1]):
            flags.append("trim_inf")
            df['upper'].values[-1] = df['lower'].values[-1]

    if use_log_bins:
        df['loglower'] = np.log10(df['lower'])
        df['logupper'] = np.log10(df['upper'])
        df.pop("lower")
        df.pop("upper")

    if df.shape[0] <= min_bins:
        flags.append("min_bins")

    N = df.counts.sum() 
    
    if N < min_counts:
        flags.append("min_counts")

    if min_bins * min_counts_to_bins_ratio > N:
        flags.append("bin_counts_ratio")

    return df, flags

