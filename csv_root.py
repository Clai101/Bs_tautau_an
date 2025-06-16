import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
dataset = ds.dataset("sig_data/merged.parquet", format="parquet")
scanner = dataset.scanner(batch_size=100_000)

writer = None
previous_df = None

# Инициализация для первой группы кандидатов
best_lik = None
best_idx = None

def process_group(df, start_idx, end_idx, best_idx):
    df.loc[start_idx:end_idx, 'is_best_Bs_lik'] = 0
    if best_idx is not None:
        df.at[best_idx, 'is_best_Bs_lik'] = 1

for batch in scanner.to_batches():
    table = pa.Table.from_batches([batch])
    current_df = table.to_pandas()

    if previous_df is not None:
        df = pd.concat([previous_df, current_df], ignore_index=True)
    else:
        df = current_df

    df['is_best_Bs_lik'] = 0

    # Индексы для прохода по df
    best_lik = None
    best_idx = None
    group_start_idx = 0

    for idx, row in df.iterrows():
        candidate = row['__candidate__']
        bs_lik = row['Bs_lik']

        if candidate == 0 and idx != group_start_idx:
            # Новое событие началось, обработаем предыдущее
            process_group(df, group_start_idx, idx - 1, best_idx)

            # Начинаем новую группу
            group_start_idx = idx
            best_lik = bs_lik
            best_idx = idx
        else:
            # Внутри события
            if (best_lik is None) or (bs_lik > best_lik):
                best_lik = bs_lik
                best_idx = idx

    # Сохраним хвост для следующего батча (последнюю группу кандидатов)
    last_group_df = df.iloc[group_start_idx:].copy()
    previous_df = last_group_df

    # Запишем обработанную часть (кроме хвоста)
    to_write = df.iloc[:group_start_idx]
    if not to_write.empty:
        out_table = pa.Table.from_pandas(to_write)
        if writer is None:
            writer = pq.ParquetWriter('sig_data/processed_two_pass_correct.parquet', out_table.schema)
        writer.write_table(out_table)

# После всех батчей обработаем остаток
if previous_df is not None:
    df = previous_df
    df['is_best_Bs_lik'] = 0
    best_lik = None
    best_idx = None
    group_start_idx = 0

    for idx, row in df.iterrows():
        candidate = row['__candidate__']
        bs_lik = row['Bs_lik']

        if candidate == 0 and idx != group_start_idx:
            process_group(df, group_start_idx, idx - 1, best_idx)
            group_start_idx = idx
            best_lik = bs_lik
            best_idx = idx
        else:
            if (best_lik is None) or (bs_lik > best_lik):
                best_lik = bs_lik
                best_idx = idx

    # Обработать последнюю группу
    process_group(df, group_start_idx, len(df) - 1, best_idx)

    out_table = pa.Table.from_pandas(df)
    if writer is None:
        writer = pq.ParquetWriter('sig_data/processed_two_pass_correct.parquet', out_table.schema)
    writer.write_table(out_table)

if writer:
    writer.close()
