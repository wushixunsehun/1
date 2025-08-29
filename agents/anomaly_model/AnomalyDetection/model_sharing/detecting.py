import os
import json
import argparse
import tsfel
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils import load_cluster, special_distance, ConfigHandler
from TransformerAndMoe import TransformerModel, model_predict



def normalized(metrics, df):
    # Preprocess the data, including filling in nan values; Processing the difference between indicators containing total; Z-score standardization
    for i in metrics:
        df[i].replace('N/A', 'NaN', inplace=True)
        df[i] = df[i].interpolate(method='linear', limit_direction='forward', axis=0)
        if 'total' in i:
            df[i] = df[i].diff()
            df[i] = df[i].fillna(0.0)
        metric_df = df[i].astype(float)
        metric_df = metric_df.sort_values()
        df_5percent = int(metric_df.shape[0] * 0.05)
        df_selected = metric_df[df_5percent: metric_df.shape[0] - df_5percent]

        df_mean = np.mean(df_selected)
        df_std = np.std(df_selected)
        if df_std == 0:
            df_std = 1
        df[i] = (df[i].astype(float) - df_mean) / df_std

        df[i] = df[i].clip(lower=-5, upper=5)
        if df[i].isnull().values.any():
            df[i] = df[i].interpolate(method='linear', limit_direction='forward', axis=0)
    return df


def tsfel_match(used_data, tsfel_cfg, metrics, cluster_num, center_subfolder, center_file):
    """
    Matches a MTS segment to a cluster using TSFEL features and distance calculation.

    Args:
        used_data: MTS segment to be matched.
        tsfel_cfg: Configuration for TSFEL feature extraction.
        metrics: List of candidate metrics.
        cluster_num: Number of clusters.
        center_subfolder: Path to the folder containing cluster centers.
        center_file: File name of the cluster center.

    Returns:
        int: The matched cluster label.
    """
    extract_feature = []
    temp_list = []
    for i in range(used_data.shape[1]):
        temp_df = used_data[:, i]
        X = tsfel.time_series_features_extractor(tsfel_cfg, temp_df)
        if len(extract_feature) == 0:
            for j in X.columns:
                extract_feature.append("_".join(j.split("_")[1:]))
        temp_list.append(X.reset_index(drop=True))
    tsfel_df = pd.concat(temp_list, axis=0)
    tsfel_df.columns = extract_feature
    tsfel_df.index = metrics

    FFT_index = -1
    features_list = tsfel_df.columns.tolist()
    for i in range(len(features_list)):
        if features_list[i].split("_")[0] == "FFT mean coefficient":
            FFT_index = i
            break
    feature_indices = [
        i for i in features_list if i.split("_")[0] == "FFT mean coefficient"
    ]
    new_column_list = [i for i in features_list if not i in feature_indices]
    new_column_list.insert(FFT_index, "FFT mean coefficient")

    average_values = tsfel_df[feature_indices].mean(axis=1)

    tsfel_df["FFT mean coefficient"] = average_values
    result = tsfel_df.reindex(columns=new_column_list)

    used_tsfel_data = result.fillna(0).values

    label_list = list(range(cluster_num))
    distance_list = []
    for detect_label in label_list:
        center_path = f"{center_subfolder}/{detect_label}/{center_file}"
        distance = special_distance(center_path, used_tsfel_data, metrics)
        distance_list.append(distance)
    node_judge = pd.DataFrame({"label": label_list, "distance": distance_list})
    node_judge.sort_values(by="distance", inplace=True, ascending=True)
    match_label = node_judge["label"].iloc[0]

    return match_label


def fit_point_threshold(score_list, window_size=120, ignore_peak=3, k_sigma=5, period_seg=20, ratio=0.4):
    thresholds_point_max = []
    thresholds_point_min = []
    alarm_point = []
    
    start = 0
    end = len(score_list)
    end -= 1        

    temp_max=0
    temp_min=0
    for j in range(end-start):
        start_in = max(start + j  - window_size, start)
        end_in = start + j + 1
        temp_df = score_list[start_in:end_in]
        if len(temp_df) > ignore_peak * 4:
            temp_sorted = sorted(temp_df)
            temp_selected = temp_sorted[ignore_peak: len(temp_sorted) - ignore_peak]
        else:
            temp_selected = temp_df
        if len(temp_selected) == 0:
            print(f'start_in: {start_in} end_in: {end_in} j {j}')

        temp_mean = np.mean(temp_selected)
        temp_std = np.std(temp_selected)

        temp_max = temp_mean + k_sigma * temp_std
        temp_min = temp_mean - k_sigma * temp_std
        if len(temp_df) > ignore_peak * 2:
            if score_list[end_in-1] > temp_max or score_list[end_in-1] < temp_min:
                alarm_point.append(start + j)

        thresholds_point_max.append(temp_max)
        thresholds_point_min.append(temp_min)

    alarm_point.sort()
    predict_list = []
    period_num = len(score_list) // period_seg
    for i in range(period_num):
        select = [j for j in alarm_point if i*period_seg <= j <= (i+1)*period_seg]
        if len(select) >= int(period_seg * ratio):
            predict_list.append(i)
    
    predict_list.sort()
    return [i*period_seg for i in predict_list]


def run_detect(node_df=None):
    # loading
    parser = argparse.ArgumentParser(
        description='Detection'
    )
    
    cfg_file = f"config.yml"
    cfg_para = {"dataset": None, "rawdata_dir": None, "config_file": cfg_file}
    config = ConfigHandler(cfg_para).config

    tsfel_cfg = tsfel.get_features_by_domain()

    weight_file = config.metric_weight_file
    model_folder = f"{config.model_dir}"
    center_subfolder = config.center_feature

    clusternum = len(os.listdir(center_subfolder))

    with open(config.metric_file, "r", encoding="utf-8") as f:
        # loading metric 
        metrics = json.load(f)

    input_dim = len(metrics)
    nhead = config.nhead
    dim_feedforward = (input_dim // nhead ) * nhead if (input_dim // nhead ) * nhead % 2==0 else (input_dim // nhead + 1) * nhead
    num_layers = config.num_layers
    num_experts = config.num_experts
    k = config.k
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading data
    print("loading data")
    if node_df is None:
        node_df = pd.read_csv(config.data_file, index_col=0, header=0)

    if config.preprocess:
        print("preprocessing data")
        node_df = normalized(metrics, node_df)

    test_data = node_df[metrics].values
    
    total_score_weight = np.array([])

    used_data = test_data[:240]

    print(f"pattern match start")
    label = 0
    label = tsfel_match(
        used_data, tsfel_cfg, metrics, clusternum, center_subfolder, config.center_file
    )
    print(f"pattern match end")
    print(f"load model...")
    model_save_path = os.path.join(model_folder, str(label))

    model = TransformerModel(
        input_dim=input_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        device=device,
        num_experts=num_experts,
        k=k,
    ).to(device)
    
    model.load_state_dict(
        torch.load(
            os.path.join(model_save_path, config.model_name),
            map_location=torch.device(device),
        )
    )
    model.eval()
    print('reconstruct start')
    reconstructed, original = model_predict(model, test_data)

    weight_path = f"{center_subfolder}/{label}/{weight_file}"

    x_all_processed = test_data[-len(reconstructed) :]
    with open(weight_path, "r", encoding="utf-8") as f:
        weight_dict = json.load(f)
    reconstruct_error = np.square(x_all_processed - reconstructed)
    weight_list = list(weight_dict.values())
    all_score = np.sum(reconstruct_error * weight_list, axis=1).reshape(-1, 1)

    if total_score_weight.size == 0:
        total_score_weight = all_score
    else:
        total_score_weight = np.concatenate((total_score_weight, all_score), axis=0)

    score_list = total_score_weight.tolist()

    alarms_list = fit_point_threshold(score_list, window_size=config.window_size, k_sigma=config.k_sigma, period_seg=config.period_seg, ratio=config.ratio)   
    print(f'Anomaly : {alarms_list}')
    pd.DataFrame({"alarms": alarms_list}).to_csv(f'{config.result_dir}/result.csv', index=False)
    return alarms_list


if __name__ == "__main__":
    
    # Two way to start detection:

    # # 1. Load data and perform detecting
    # # data
    node_df = pd.read_csv("/home/tanxh/mas/agents/anomaly_model/AnomalyDetection/data/Dataset1/Node1.csv", index_col=0, header=0)

    # # detect
    run_detect(node_df)

    # 2. Or read the data to be detected according to the configuration file config.yml
    # run_detect()

