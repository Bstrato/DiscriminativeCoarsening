import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gc
import time
import numpy as np


def safe_divide(numerator, denominator, default=0.0):
    """Safe division that returns default value for division by zero or NaN"""
    if pd.isna(denominator) or denominator == 0:
        return default
    if pd.isna(numerator):
        return default
    return numerator / denominator


def safe_normalize(value, max_val, default=0.0):
    """Safe normalization with NaN checking"""
    if pd.isna(value) or pd.isna(max_val) or max_val == 0:
        return default
    return min(value / max_val, 1.0)


def clean_features(features_array):
    """Clean feature array to remove any NaN or inf values"""
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
    mask = np.isfinite(features_array)
    features_array[~mask] = 0.0
    return features_array


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')

torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('medium')

hosp_path = "/nfs/hpc/share/bayitaas/stra/MIMIC/physionet.org/files/mimiciv/3.1/hosp"
icu_path = "/nfs/hpc/share/bayitaas/stra/MIMIC/physionet.org/files/mimiciv/3.1/icu"

start_time = time.time()
print('Loading MIMIC-IV datasets with real medical features...')

print('Loading core datasets...')
icu = pd.read_csv(f'{icu_path}/icustays.csv.gz')
patients = pd.read_csv(f'{hosp_path}/patients.csv.gz')
admissions = pd.read_csv(f'{hosp_path}/admissions.csv.gz')
diagnoses = pd.read_csv(f'{hosp_path}/diagnoses_icd.csv.gz')
d_items = pd.read_csv(f'{icu_path}/d_items.csv.gz')

print(f'ICU stays: {len(icu):,} rows')
print(f'Patients: {len(patients):,} rows')
print(f'Admissions: {len(admissions):,} rows')
print(f'Diagnoses: {len(diagnoses):,} rows')
print(f'Item definitions: {len(d_items):,} rows')

valid_stay_ids = set(icu['stay_id'])
print(f'Valid stay IDs: {len(valid_stay_ids):,}')

chunk_size = 100000
print(f'Using chunk size: {chunk_size:,}')

print('Loading chart events in chunks...')
chartevents_list = []
chunk_count = 0

for chunk in pd.read_csv(f'{icu_path}/chartevents.csv.gz', chunksize=chunk_size):
    chunk_count += 1
    print(f'Processing chunk {chunk_count}: {len(chunk):,} rows')

    chunk = chunk.dropna(subset=['stay_id'])
    chunk = chunk[chunk['stay_id'].isin(valid_stay_ids)]

    if len(chunk) > 0:
        chartevents_list.append(chunk)

    if chunk_count >= 20:
        print(f'Stopping at chunk {chunk_count} to manage memory')
        break

    if torch.cuda.is_available() and chunk_count % 5 == 0:
        torch.cuda.empty_cache()

if chartevents_list:
    chartevents = pd.concat(chartevents_list, ignore_index=True)
    print(f'Combined chart events: {len(chartevents):,} rows')
else:
    chartevents = pd.DataFrame()

del chartevents_list
gc.collect()

print('Loading input events in chunks...')
inputevents_list = []
chunk_count = 0

for chunk in pd.read_csv(f'{icu_path}/inputevents.csv.gz', chunksize=chunk_size):
    chunk_count += 1
    print(f'Processing input chunk {chunk_count}: {len(chunk):,} rows')

    chunk = chunk.dropna(subset=['stay_id'])
    chunk = chunk[chunk['stay_id'].isin(valid_stay_ids)]

    if len(chunk) > 0:
        inputevents_list.append(chunk)

    if chunk_count >= 20:
        break

    if torch.cuda.is_available() and chunk_count % 5 == 0:
        torch.cuda.empty_cache()

if inputevents_list:
    inputevents = pd.concat(inputevents_list, ignore_index=True)
    print(f'Combined input events: {len(inputevents):,} rows')
else:
    inputevents = pd.DataFrame()

del inputevents_list
gc.collect()

print(f'Data loading completed in {time.time() - start_time:.2f} seconds')


def extract_patient_features(patients_df, admissions_df, diagnoses_df, icu_df):
    """Extract patient features from demographics, admissions, and diagnoses"""
    print('Extracting patient features from demographics, admissions, and diagnoses...')

    icu_patients = icu_df[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates('subject_id')

    patient_data = icu_patients.merge(
        patients_df[['subject_id', 'gender', 'anchor_age']],
        on='subject_id', how='left'
    )

    patient_data = patient_data.merge(
        admissions_df[['subject_id', 'hadm_id', 'admission_type', 'admission_location',
                       'insurance', 'race', 'marital_status', 'hospital_expire_flag']],
        on=['subject_id', 'hadm_id'], how='left'
    )

    diagnosis_complexity = diagnoses_df.groupby('subject_id').agg({
        'icd_code': 'count',
        'seq_num': 'max'
    }).rename(columns={'icd_code': 'diagnosis_count', 'seq_num': 'max_seq_num'})

    diagnosis_complexity = diagnosis_complexity.fillna(0)

    patient_data = patient_data.merge(diagnosis_complexity, on='subject_id', how='left')

    subject_ids = icu_df['subject_id'].unique()
    subject_id_map = {sid: i for i, sid in enumerate(subject_ids)}

    features = np.zeros((len(subject_ids), 16))

    for idx, subject_id in enumerate(subject_ids):
        patient_row = patient_data[patient_data['subject_id'] == subject_id]

        if len(patient_row) > 0:
            row = patient_row.iloc[0]

            age = row.get('anchor_age', 65)
            features[idx, 0] = safe_normalize(age, 100.0, 0.65)
            features[idx, 1] = 1.0 if str(row.get('gender', 'M')).upper() == 'M' else 0.0

            admission_type = str(row.get('admission_type', '')).upper()
            features[idx, 2] = 1.0 if 'EMERGENCY' in admission_type else 0.0
            features[idx, 3] = 1.0 if 'ELECTIVE' in admission_type else 0.0
            features[idx, 4] = 1.0 if 'URGENT' in admission_type else 0.0

            insurance = str(row.get('insurance', '')).upper()
            features[idx, 5] = 1.0 if 'MEDICARE' in insurance else 0.0
            features[idx, 6] = 1.0 if 'MEDICAID' in insurance else 0.0
            features[idx, 7] = 1.0 if 'OTHER' in insurance else 0.0

            race = str(row.get('race', '')).upper()
            features[idx, 8] = 1.0 if 'WHITE' in race else 0.0
            features[idx, 9] = 1.0 if 'BLACK' in race else 0.0
            features[idx, 10] = 1.0 if 'HISPANIC' in race else 0.0
            features[idx, 11] = 1.0 if 'ASIAN' in race else 0.0

            marital = str(row.get('marital_status', '')).upper()
            features[idx, 12] = 1.0 if 'MARRIED' in marital else 0.0

            diag_count = row.get('diagnosis_count', 0)
            max_seq = row.get('max_seq_num', 0)
            expire_flag = row.get('hospital_expire_flag', 0)

            features[idx, 13] = safe_normalize(diag_count, 20.0, 0.0)
            features[idx, 14] = safe_normalize(max_seq, 10.0, 0.0)
            features[idx, 15] = 1.0 if expire_flag == 1 else 0.0
        else:
            features[idx, 0] = 0.65
            features[idx, 1] = 0.5

    features = clean_features(features)

    print(f'Patient features extracted: {features.shape}, NaN count: {np.isnan(features).sum()}')
    return torch.tensor(features, dtype=torch.float32), subject_id_map


def extract_stay_features(icu_df, chartevents_df, inputevents_df):
    """Extract stay features from ICU data and aggregated events"""
    print('Extracting stay features from ICU data and aggregated events...')

    features = np.zeros((len(icu_df), 32))

    le_first = LabelEncoder()
    le_last = LabelEncoder()

    icu_df['first_careunit'] = icu_df['first_careunit'].fillna('UNKNOWN')
    icu_df['last_careunit'] = icu_df['last_careunit'].fillna('UNKNOWN')

    icu_df['first_careunit_encoded'] = le_first.fit_transform(icu_df['first_careunit'].astype(str))
    icu_df['last_careunit_encoded'] = le_last.fit_transform(icu_df['last_careunit'].astype(str))

    if len(chartevents_df) > 0:
        chartevents_clean = chartevents_df.copy()
        chartevents_clean['valuenum'] = pd.to_numeric(chartevents_clean['valuenum'], errors='coerce')
        chartevents_clean = chartevents_clean.dropna(subset=['valuenum'])

        if len(chartevents_clean) > 0:
            chart_stats = chartevents_clean.groupby('stay_id')['valuenum'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).fillna(0)
        else:
            chart_stats = pd.DataFrame()
    else:
        chart_stats = pd.DataFrame()

    if len(inputevents_df) > 0:
        inputevents_clean = inputevents_df.copy()
        for col in ['amount', 'rate', 'patientweight']:
            inputevents_clean[col] = pd.to_numeric(inputevents_clean[col], errors='coerce')

        input_stats = inputevents_clean.groupby('stay_id').agg({
            'amount': ['count', 'sum', 'mean'],
            'rate': ['mean', 'max'],
            'patientweight': 'first'
        }).fillna(0)
        input_stats.columns = ['_'.join(col) for col in input_stats.columns]
    else:
        input_stats = pd.DataFrame()

    for idx, (_, stay) in enumerate(icu_df.iterrows()):
        stay_id = stay['stay_id']

        los = stay.get('los', 1.0)
        features[idx, 0] = safe_normalize(los, 30.0, 0.033)
        features[idx, 1] = safe_normalize(stay['first_careunit_encoded'], 20.0, 0.0)
        features[idx, 2] = safe_normalize(stay['last_careunit_encoded'], 20.0, 0.0)
        features[idx, 3] = 1.0 if stay['first_careunit'] != stay['last_careunit'] else 0.0

        if stay_id in chart_stats.index and len(chart_stats) > 0:
            chart_row = chart_stats.loc[stay_id]
            features[idx, 4] = safe_normalize(chart_row['count'], 1000.0, 0.0)
            features[idx, 5] = safe_normalize(chart_row['mean'], 100.0, 0.0)
            features[idx, 6] = safe_normalize(chart_row['std'], 50.0, 0.0)
            features[idx, 7] = safe_normalize(chart_row['min'], 100.0, 0.0)
            features[idx, 8] = safe_normalize(chart_row['max'], 200.0, 0.0)

        if stay_id in input_stats.index and len(input_stats) > 0:
            input_row = input_stats.loc[stay_id]
            features[idx, 9] = safe_normalize(input_row.get('amount_count', 0), 100.0, 0.0)
            features[idx, 10] = safe_normalize(input_row.get('amount_sum', 0), 10000.0, 0.0)
            features[idx, 11] = safe_normalize(input_row.get('amount_mean', 0), 1000.0, 0.0)
            features[idx, 12] = safe_normalize(input_row.get('rate_mean', 0), 100.0, 0.0)
            features[idx, 13] = safe_normalize(input_row.get('rate_max', 0), 500.0, 0.0)
            features[idx, 14] = safe_normalize(input_row.get('patientweight_first', 80), 200.0, 0.4)

        try:
            intime = pd.to_datetime(stay['intime'])
            outtime = pd.to_datetime(stay['outtime'])

            if pd.notna(intime) and pd.notna(outtime):
                features[idx, 15] = safe_normalize(intime.hour, 24.0, 0.5)
                features[idx, 16] = safe_normalize(intime.weekday(), 7.0, 0.5)

                actual_los_hours = (outtime - intime).total_seconds() / 3600.0
                features[idx, 17] = safe_normalize(actual_los_hours, 24.0 * 30.0, features[idx, 0])
            else:
                features[idx, 15] = 0.5
                features[idx, 16] = 0.5
                features[idx, 17] = features[idx, 0]
        except:
            features[idx, 15] = 0.5
            features[idx, 16] = 0.5
            features[idx, 17] = features[idx, 0]

        features[idx, 18] = safe_normalize(stay_id % 12, 12.0, 0.5)
        features[idx, 19] = safe_normalize(stay_id % 7, 7.0, 0.5)

        total_events = features[idx, 4] + features[idx, 9]
        features[idx, 20] = min(total_events, 1.0)

        features[idx, 21] = 1.0 if features[idx, 0] > 0.5 else 0.0
        features[idx, 22] = 1.0 if features[idx, 3] > 0 else 0.0

        for i in range(23, 32):
            base_idx = i - 23
            if base_idx < 23:
                features[idx, i] = np.tanh(features[idx, base_idx])
            else:
                features[idx, i] = 0.0

    features = clean_features(features)

    print(f'Stay features extracted: {features.shape}, NaN count: {np.isnan(features).sum()}')
    return torch.tensor(features, dtype=torch.float32)


def extract_chartevent_features(chartevents_df, d_items_df):
    """Extract chart event features with medical context"""
    print('Extracting chart event features with medical context...')

    if len(chartevents_df) == 0:
        return torch.zeros(0, 8, dtype=torch.float32)

    chart_with_items = chartevents_df.merge(
        d_items_df[['itemid', 'label', 'category', 'lownormalvalue', 'highnormalvalue']],
        on='itemid', how='left'
    )

    features = np.zeros((len(chartevents_df), 8))

    for idx, (_, event) in enumerate(chart_with_items.iterrows()):
        itemid = event.get('itemid', 0)
        features[idx, 0] = safe_normalize(itemid, 230000.0, 0.0)

        value = pd.to_numeric(event.get('valuenum', 0), errors='coerce')
        if pd.isna(value):
            value = 0
        features[idx, 1] = np.tanh(safe_divide(value, 100.0, 0.0))

        low_normal = pd.to_numeric(event.get('lownormalvalue'), errors='coerce')
        high_normal = pd.to_numeric(event.get('highnormalvalue'), errors='coerce')

        if pd.notna(low_normal) and pd.notna(high_normal) and pd.notna(value) and value != 0:
            if value < low_normal:
                features[idx, 2] = -1.0
            elif value > high_normal:
                features[idx, 2] = 1.0
            else:
                features[idx, 2] = 0.0

            if value < low_normal and low_normal > 0:
                features[idx, 3] = min(safe_divide(low_normal - value, low_normal, 0.0), 1.0)
            elif value > high_normal and high_normal > 0:
                features[idx, 3] = min(safe_divide(value - high_normal, high_normal, 0.0), 1.0)
            else:
                features[idx, 3] = 0.0

        category = str(event.get('category', '')).upper()
        features[idx, 4] = 1.0 if 'VITAL' in category else 0.0
        features[idx, 5] = 1.0 if 'LAB' in category else 0.0

        features[idx, 6] = safe_normalize(itemid % 24, 24.0, 0.5)
        features[idx, 7] = safe_normalize(abs(value), 1000.0, 0.0) if pd.notna(value) else 0.0

    features = clean_features(features)

    print(f'Chart event features extracted: {features.shape}, NaN count: {np.isnan(features).sum()}')
    return torch.tensor(features, dtype=torch.float32)


def extract_inputevent_features(inputevents_df):
    """Extract input event features with medication context"""
    print('Extracting input event features with medication context...')

    if len(inputevents_df) == 0:
        return torch.zeros(0, 8, dtype=torch.float32)

    features = np.zeros((len(inputevents_df), 8))

    for idx, (_, event) in enumerate(inputevents_df.iterrows()):
        itemid = event.get('itemid', 0)
        features[idx, 0] = safe_normalize(itemid, 230000.0, 0.0)

        amount = pd.to_numeric(event.get('amount', 0), errors='coerce')
        if pd.isna(amount):
            amount = 0
        features[idx, 1] = np.tanh(safe_divide(amount, 1000.0, 0.0))

        rate = pd.to_numeric(event.get('rate', 0), errors='coerce')
        if pd.isna(rate):
            rate = 0
        features[idx, 2] = np.tanh(safe_divide(rate, 100.0, 0.0))

        weight = pd.to_numeric(event.get('patientweight', 80), errors='coerce')
        if pd.isna(weight) or weight <= 0:
            weight = 80
        features[idx, 3] = safe_normalize(weight, 200.0, 0.4)

        if weight > 0:
            features[idx, 4] = safe_normalize(safe_divide(amount, weight, 0.0), 10.0, 0.0)

        order_cat = str(event.get('ordercategoryname', '')).upper()
        features[idx, 5] = 1.0 if any(x in order_cat for x in ['DRUG', 'MED']) else 0.0
        features[idx, 6] = 1.0 if 'FLUID' in order_cat else 0.0

        total_amount = pd.to_numeric(event.get('totalamount', amount), errors='coerce')
        if pd.isna(total_amount):
            total_amount = amount
        features[idx, 7] = safe_normalize(total_amount, 10000.0, 0.0)

    features = clean_features(features)

    print(f'Input event features extracted: {features.shape}, NaN count: {np.isnan(features).sum()}')
    return torch.tensor(features, dtype=torch.float32)


print('Encoding care units safely...')
le_first = LabelEncoder()
le_last = LabelEncoder()

icu['first_careunit'] = icu['first_careunit'].fillna('UNKNOWN')
icu['last_careunit'] = icu['last_careunit'].fillna('UNKNOWN')

icu['first_careunit'] = le_first.fit_transform(icu['first_careunit'].astype(str))
icu['last_careunit'] = le_last.fit_transform(icu['last_careunit'].astype(str))

print('Building heterogeneous graph with medical features...')
data = HeteroData()

patient_features, subject_id_map = extract_patient_features(patients, admissions, diagnoses, icu)
stay_features = extract_stay_features(icu, chartevents, inputevents)
chartevent_features = extract_chartevent_features(chartevents, d_items)
inputevent_features = extract_inputevent_features(inputevents)

print('Final NaN verification:')
print(f'Patient features NaN count: {torch.isnan(patient_features).sum()}')
print(f'Stay features NaN count: {torch.isnan(stay_features).sum()}')
print(f'Chart event features NaN count: {torch.isnan(chartevent_features).sum()}')
print(f'Input event features NaN count: {torch.isnan(inputevent_features).sum()}')

data['patient'].x = patient_features.to(device)
print(f'Patient nodes: {data["patient"].x.shape[0]:,} with {data["patient"].x.shape[1]} features')

data['stay'].x = stay_features.to(device)
print(f'Stay nodes: {data["stay"].x.shape[0]:,} with {data["stay"].x.shape[1]} features')

careunits = pd.concat([icu['first_careunit'], icu['last_careunit']]).unique()
careunit_map = {cu: i for i, cu in enumerate(careunits)}

careunit_features = torch.zeros(len(careunit_map), 8)
for i, cu in enumerate(careunits):
    careunit_features[i, 0] = safe_normalize(cu, len(careunits), 0.0)

    first_count = (icu['first_careunit'] == cu).sum()
    last_count = (icu['last_careunit'] == cu).sum()

    careunit_features[i, 1] = safe_normalize(first_count, len(icu), 0.0)
    careunit_features[i, 2] = safe_normalize(last_count, len(icu), 0.0)

    unit_stays = icu[icu['first_careunit'] == cu]
    if len(unit_stays) > 0:
        mean_los = unit_stays['los'].mean()
        if pd.notna(mean_los):
            careunit_features[i, 3] = safe_normalize(mean_los, 30.0, 0.033)
        else:
            careunit_features[i, 3] = 0.033
    else:
        careunit_features[i, 3] = 0.033

    for j in range(4, 8):
        careunit_features[i, j] = careunit_features[i, j - 4] * 0.5

careunit_features = torch.nan_to_num(careunit_features, nan=0.0, posinf=1.0, neginf=0.0)

data['careunit'].x = careunit_features.to(device)
print(f'Careunit nodes: {len(careunit_map):,} with 8 features')

if len(chartevents) > 0:
    data['chartevent'].x = chartevent_features.to(device)
    print(f'Chart event nodes: {len(chartevents):,} with 8 medical features')

if len(inputevents) > 0:
    data['inputevent'].x = inputevent_features.to(device)
    print(f'Input event nodes: {len(inputevents):,} with 8 medical features')

icu_los_clean = icu['los'].fillna(1.0)
icu_los_clean = np.where(icu_los_clean <= 0, 0.1, icu_los_clean)

data['stay'].y_regression = torch.tensor(icu_los_clean, dtype=torch.float32).to(device)


def create_robust_clinical_los_bins(los_series):
    """Create clinically meaningful LOS bins with NaN handling"""
    print('Creating clinical LOS classification bins...')

    los_clean = los_series.copy()

    los_clean = los_clean.fillna(1.0)
    los_clean = np.where(los_clean <= 0, 0.1, los_clean)
    los_clean = np.where(np.isinf(los_clean), 1.0, los_clean)

    bins = [0, 2, 7, float('inf')]
    labels = [0, 1, 2]

    try:
        los_bins = pd.cut(los_clean, bins=bins, labels=labels, include_lowest=True)
        los_bins = los_bins.astype('Int64').fillna(0).astype(int)
    except Exception as e:
        print(f'Binning error: {e}, using fallback...')
        los_bins = np.zeros(len(los_clean), dtype=int)
        los_bins[los_clean <= 2] = 0
        los_bins[(los_clean > 2) & (los_clean <= 7)] = 1
        los_bins[los_clean > 7] = 2

    los_bins = np.where(pd.isna(los_bins), 0, los_bins)
    los_bins = los_bins.astype(int)

    unique, counts = np.unique(los_bins, return_counts=True)
    total = len(los_bins)
    print('Clinical LOS Classification Distribution:')
    class_names = ['Short (≤2 days)', 'Medium (3-7 days)', 'Long (≥8 days)']
    for val, count in zip(unique, counts):
        if val < len(class_names):
            percentage = (count / total) * 100
            print(f'  {class_names[val]}: {count:,} stays ({percentage:.1f}%)')

    return los_bins


los_bins = create_robust_clinical_los_bins(icu['los'])
data['stay'].y_classification = torch.tensor(los_bins, dtype=torch.long).to(device)

print('Creating edges...')
stay_to_idx = {sid: i for i, sid in enumerate(icu['stay_id'])}

patient_subject_ids = icu['subject_id'].fillna(-1)
patient_idx = patient_subject_ids.map(subject_id_map).fillna(-1).astype(int)

valid_patient_mask = patient_idx >= 0
patient_idx = patient_idx[valid_patient_mask]
stay_idx = icu.index.to_numpy()[valid_patient_mask]

data['patient', 'has_stay', 'stay'].edge_index = torch.tensor(
    [patient_idx, stay_idx], dtype=torch.long, device=device
)

print(f'Patient-Stay edges: {len(patient_idx):,}')

if len(chartevents) > 0:
    chartevents_clean_stay = chartevents['stay_id'].fillna(-1)
    stay_idx_ce = chartevents_clean_stay.map(stay_to_idx).dropna()

    valid_ce_mask = stay_idx_ce.index
    stay_idx_ce = stay_idx_ce.astype(int)
    chartevent_idx = chartevents.loc[valid_ce_mask].index.to_numpy()

    data['stay', 'has_chartevent', 'chartevent'].edge_index = torch.tensor([
        stay_idx_ce.values, chartevent_idx
    ], dtype=torch.long, device=device)

    print(f'Stay-ChartEvent edges: {len(stay_idx_ce):,}')

if len(inputevents) > 0:
    inputevents_clean_stay = inputevents['stay_id'].fillna(-1)
    stay_idx_ie = inputevents_clean_stay.map(stay_to_idx).dropna()

    valid_ie_mask = stay_idx_ie.index
    stay_idx_ie = stay_idx_ie.astype(int)
    inputevent_idx = inputevents.loc[valid_ie_mask].index.to_numpy()

    data['stay', 'has_inputevent', 'inputevent'].edge_index = torch.tensor([
        stay_idx_ie.values, inputevent_idx
    ], dtype=torch.long, device=device)

    print(f'Stay-InputEvent edges: {len(stay_idx_ie):,}')

stay_to_cu_first = list(zip(icu.index, icu['first_careunit']))
stay_to_cu_last = list(zip(icu.index, icu['last_careunit']))
stay_to_cu = list(set(stay_to_cu_first + stay_to_cu_last))

valid_stay_cu = [(s, c) for s, c in stay_to_cu if c in careunit_map and not pd.isna(c)]
stay_idx_cu, cu_idx = zip(*[(s, careunit_map[c]) for s, c in valid_stay_cu])

data['stay', 'in_careunit', 'careunit'].edge_index = torch.tensor(
    [stay_idx_cu, cu_idx], dtype=torch.long, device=device
)

print(f'Stay-Careunit edges: {len(stay_idx_cu):,}')

print('Performing final NaN verification on complete graph...')
nan_found = False
for node_type in data.node_types:
    if hasattr(data[node_type], 'x'):
        nan_count = torch.isnan(data[node_type].x).sum()
        if nan_count > 0:
            print(f'WARNING: {nan_count} NaN values found in {node_type} features!')
            nan_found = True
        else:
            print(f'{node_type} features: NaN-free')

if hasattr(data['stay'], 'y_regression'):
    reg_nan = torch.isnan(data['stay'].y_regression).sum()
    if reg_nan > 0:
        print(f'WARNING: {reg_nan} NaN values in regression targets!')
        nan_found = True
    else:
        print(f'Regression targets: NaN-free')

if hasattr(data['stay'], 'y_classification'):
    cls_invalid = (data['stay'].y_classification < 0).sum() + (data['stay'].y_classification > 2).sum()
    if cls_invalid > 0:
        print(f'WARNING: {cls_invalid} invalid values in classification targets!')
        nan_found = True
    else:
        print(f'Classification targets: Valid range [0-2]')

if not nan_found:
    print('GRAPH CONSTRUCTION SUCCESSFUL: NO NaN VALUES DETECTED!')
else:
    print('NaN values detected - applying emergency cleaning...')
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            data[node_type].x = torch.nan_to_num(data[node_type].x, nan=0.0, posinf=1.0, neginf=0.0)

    if hasattr(data['stay'], 'y_regression'):
        data['stay'].y_regression = torch.nan_to_num(data['stay'].y_regression, nan=1.0, posinf=30.0, neginf=0.1)

    print('Emergency cleaning completed!')

print('Saving datasets...')
data_cpu = data.cpu()

data_reg = data_cpu.clone()
data_reg['stay'].y = data_reg['stay'].y_regression
torch.save(data_reg, 'regression_los_clinical_real_features_nan_safe.pt')
print('Saved: regression_los_clinical_real_features_nan_safe.pt')

data_cls = data_cpu.clone()
data_cls['stay'].y = data_cls['stay'].y_classification
torch.save(data_cls, 'classification_los_clinical_real_features_nan_safe.pt')
print('Saved: classification_los_clinical_real_features_nan_safe.pt')

print('Final verification of saved data:')
saved_reg = torch.load('regression_los_clinical_real_features_nan_safe.pt')
saved_cls = torch.load('classification_los_clinical_real_features_nan_safe.pt')

print('Regression dataset:')
for node_type in saved_reg.node_types:
    if hasattr(saved_reg[node_type], 'x'):
        nan_count = torch.isnan(saved_reg[node_type].x).sum()
        print(f'  {node_type}: {nan_count} NaN values')

reg_target_nan = torch.isnan(saved_reg['stay'].y).sum()
print(f'  Regression targets: {reg_target_nan} NaN values')

print('Classification dataset:')
for node_type in saved_cls.node_types:
    if hasattr(saved_cls[node_type], 'x'):
        nan_count = torch.isnan(saved_cls[node_type].x).sum()
        print(f'  {node_type}: {nan_count} NaN values')

cls_target_invalid = (saved_cls['stay'].y < 0).sum() + (saved_cls['stay'].y > 2).sum()
print(f'  Classification targets: {cls_target_invalid} invalid values')

if torch.cuda.is_available():
    print(f'GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB')
    print(f'GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB')

total_time = time.time() - start_time
print(f'COMPLETED: Graph created in {total_time:.2f} seconds!')

print('Final graph statistics:')
print('Node types and feature dimensions:')
for node_type in data.node_types:
    if hasattr(data[node_type], 'x'):
        print(f'  {node_type}: {data[node_type].x.shape[0]:,} nodes, {data[node_type].x.shape[1]} features')

print('Edge types and counts:')
for edge_type in data.edge_types:
    print(f'  {edge_type}: {data[edge_type].edge_index.shape[1]:,} edges')

print('Output files:')
print('  regression_los_clinical_real_features_nan_safe.pt - Medical features for LOS prediction')
print('  classification_los_clinical_real_features_nan_safe.pt - Medical features for LOS categories')

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print('Process finished successfully')