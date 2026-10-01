import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.common import ensure_dir, finish_plot, setup_plot_font
from src.config import FFT_COMPARE_SEGMENTS, FFT_DIR, FFT_FREQ_BANDS, FS, GAP_THRESHOLD, MODEL_READY_DATASETS, MOVING_ZSCORE_SUMMARY, RAW_DATASETS

def calculate_fft(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    n = len(signal)
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1 / FS)
    amplitude = np.abs(fft_values) / n * 2
    if len(amplitude) > 0:
        amplitude[0] = 0
    return frequencies, amplitude

def calculate_band_energy(frequencies, amplitude, band_start, band_end):
    mask = (frequencies >= band_start) & (frequencies < band_end)
    return 0 if mask.sum() == 0 else np.sum(amplitude[mask] ** 2)

def run():
    setup_plot_font()
    ensure_dir(FFT_DIR)
    degradation_summary = pd.read_csv(MOVING_ZSCORE_SUMMARY, parse_dates=["degradation_start"])
    band_results, summary_results = [], []
    for dataset_name in RAW_DATASETS:
        print("\n" + "=" * 80)
        print(dataset_name)
        print("=" * 80)
        degradation_row = degradation_summary[degradation_summary["dataset"] == dataset_name]
        if degradation_row.empty:
            print("열화 시작 결과 없음")
            continue
        degradation_time = degradation_row["degradation_start"].iloc[0]
        raw_df = pd.read_csv(RAW_DATASETS[dataset_name], parse_dates=["timestamp"])
        raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)
        channels = [col for col in raw_df.columns if col.startswith("channel_")]
        raw_df["time_diff"] = raw_df["timestamp"].diff()
        raw_df["segment"] = (raw_df["time_diff"] > GAP_THRESHOLD).cumsum()
        segment_info = raw_df.groupby("segment").agg(start_time=("timestamp", "first"), end_time=("timestamp", "last"), n_samples=("timestamp", "size")).reset_index()
        main_sample_size = segment_info["n_samples"].mode().iloc[0]
        valid_segments = segment_info[segment_info["n_samples"] == main_sample_size].copy()
        before_segments = valid_segments[valid_segments["start_time"] < degradation_time].tail(FFT_COMPARE_SEGMENTS)
        after_segments = valid_segments[valid_segments["start_time"] >= degradation_time].head(FFT_COMPARE_SEGMENTS)
        if before_segments.empty or after_segments.empty:
            print("FFT 비교용 segment 부족 → 건너뜀")
            continue
        dataset_dir = ensure_dir(FFT_DIR / dataset_name)
        channel_before_spectra, channel_after_spectra = {}, {}
        frequencies_ref = None
        for channel in channels:
            before_fft_list, after_fft_list = [], []
            for segment_id in before_segments["segment"]:
                sig = raw_df.loc[raw_df["segment"] == segment_id, channel].values
                frequencies, amp = calculate_fft(sig)
                frequencies_ref = frequencies
                before_fft_list.append(amp)
            for segment_id in after_segments["segment"]:
                sig = raw_df.loc[raw_df["segment"] == segment_id, channel].values
                _, amp = calculate_fft(sig)
                after_fft_list.append(amp)
            before_fft_mean = np.mean(before_fft_list, axis=0)
            after_fft_mean = np.mean(after_fft_list, axis=0)
            channel_before_spectra[channel] = before_fft_mean
            channel_after_spectra[channel] = after_fft_mean
            plt.figure(figsize=(14, 5))
            plt.plot(frequencies_ref, before_fft_mean, label="열화 전")
            plt.plot(frequencies_ref, after_fft_mean, label="열화 후")
            plt.title(f"{dataset_name} - {channel} FFT 비교")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.xlim(0, FS / 2)
            plt.legend()
            finish_plot(dataset_dir / f"{channel}_fft_before_after.png")
            for band_start, band_end in FFT_FREQ_BANDS:
                before_energy = calculate_band_energy(frequencies_ref, before_fft_mean, band_start, band_end)
                after_energy = calculate_band_energy(frequencies_ref, after_fft_mean, band_start, band_end)
                band_results.append({"dataset": dataset_name, "channel": channel, "band_start": band_start, "band_end": band_end, "before_energy": before_energy, "after_energy": after_energy, "energy_ratio": after_energy / before_energy if before_energy > 0 else np.nan})
        all_before = np.mean(np.array(list(channel_before_spectra.values())), axis=0)
        all_after = np.mean(np.array(list(channel_after_spectra.values())), axis=0)
        plt.figure(figsize=(14, 5))
        plt.plot(frequencies_ref, all_before, label="열화 전 평균")
        plt.plot(frequencies_ref, all_after, label="열화 후 평균")
        plt.title(f"{dataset_name} 전체 채널 평균 FFT 비교")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.xlim(0, FS / 2)
        plt.legend()
        finish_plot(dataset_dir / "all_channels_fft_before_after.png")
        feature_df = pd.read_csv(MODEL_READY_DATASETS[dataset_name], parse_dates=["start_time", "end_time"])
        feature_df = feature_df.sort_values("start_time").reset_index(drop=True)
        spectral_cols = [col for col in feature_df.columns if col.endswith("_spectral_energy")]
        if spectral_cols:
            feature_df["mean_spectral_energy"] = feature_df[spectral_cols].mean(axis=1)
            plt.figure(figsize=(14, 5))
            plt.plot(feature_df["start_time"], feature_df["mean_spectral_energy"], label="전체 채널 평균 Spectral Energy")
            plt.axvline(x=degradation_time, linestyle="--", label="열화 시작")
            plt.title(f"{dataset_name} 평균 Spectral Energy 추세")
            plt.xlabel("Time")
            plt.ylabel("Mean Spectral Energy")
            plt.legend()
            finish_plot(dataset_dir / "mean_spectral_energy_trend.png")
        current_band_df = pd.DataFrame([row for row in band_results if row["dataset"] == dataset_name])
        current_band_df.to_csv(dataset_dir / "frequency_band_energy.csv", index=False)
        band_summary = current_band_df.groupby(["band_start", "band_end"], as_index=False).agg(mean_before_energy=("before_energy", "mean"), mean_after_energy=("after_energy", "mean"), mean_energy_ratio=("energy_ratio", "mean"))
        band_summary["band"] = band_summary["band_start"].astype(str) + "-" + band_summary["band_end"].astype(str) + " Hz"
        band_summary.to_csv(dataset_dir / "frequency_band_summary.csv", index=False)
        plt.figure(figsize=(12, 6))
        plt.bar(band_summary["band"], band_summary["mean_energy_ratio"])
        plt.axhline(y=1, linestyle="--", label="변화 없음")
        plt.title(f"{dataset_name} 주파수 대역별 에너지 변화율")
        plt.xlabel("Frequency Band")
        plt.ylabel("After / Before Energy Ratio")
        plt.xticks(rotation=45)
        plt.legend()
        finish_plot(dataset_dir / "frequency_band_energy_ratio.png")
        valid_summary = band_summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_energy_ratio"])
        if valid_summary.empty:
            top_band_name, top_energy_ratio = None, None
        else:
            top_band = valid_summary.sort_values("mean_energy_ratio", ascending=False).iloc[0]
            top_band_name, top_energy_ratio = top_band["band"], top_band["mean_energy_ratio"]
        summary_results.append({"dataset": dataset_name, "degradation_start": degradation_time, "sample_size": main_sample_size, "before_segment_count": len(before_segments), "after_segment_count": len(after_segments), "top_frequency_band": top_band_name, "top_energy_ratio": top_energy_ratio})
    pd.DataFrame(band_results).to_csv(FFT_DIR / "all_frequency_band_energy.csv", index=False)
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(FFT_DIR / "fft_analysis_summary.csv", index=False)
    print("\n" + "=" * 80)
    print("FFT 선택과제 최종 결과")
    print("=" * 80)
    print(summary_df)

if __name__ == "__main__":
    run()
