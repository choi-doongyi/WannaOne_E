def make_rolling_zscore(
    path="data/test_set_1_snapshot_features.csv",
    baseline_fraction=0.30,
    window=5,
):
    import pandas as pd

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_names = ["rms", "kurtosis", "peak", "crest_factor"]

    feature_cols = [
        f"channel_{channel}_{feature}"
        for channel in range(1, 9)
        for feature in feature_names
    ]

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"데이터에 없는 컬럼: {missing_cols}")

    train_end = int(len(df) * baseline_fraction)
    baseline = df.loc[:train_end - 1, feature_cols]

    mean = baseline.mean()
    std = baseline.std(ddof=1)

    if std.isna().any() or std.eq(0).any():
        raise ValueError("베이스라인 표준편차를 확인하세요.")

    z_scores = df[feature_cols].sub(mean).div(std)
    rolling_z = z_scores.rolling(
        window=window,
        min_periods=window,
    ).median()

    rolling_z.columns = [
        f"{col}_rolling_zscore" for col in feature_cols
    ]

    result = pd.concat([df[["timestamp"]], rolling_z], axis=1)
    return result, train_end


result, train_end = make_rolling_zscore()
print("베이스라인 행 수:", train_end)