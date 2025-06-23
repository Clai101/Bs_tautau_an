import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pyarrow import Table

# Получаем число строк
parquet_file = pq.ParquetFile("merged.parquet")
row_count = parquet_file.metadata.num_rows

n_parts = 20
rows_per_part = row_count // n_parts

# Создаём dataset
dataset = ds.dataset("merged.parquet", format="parquet")
scanner = dataset.scanner()

batches = scanner.to_batches()

current_rows = 0
current_batches = []
part_num = 1

for batch in batches:
    current_batches.append(batch)
    current_rows += batch.num_rows

    if current_rows >= rows_per_part or part_num == n_parts:
        table = Table.from_batches(current_batches)
        pq.write_table(table, f"sig_{part_num}.parquet")
        part_num += 1
        current_rows = 0
        current_batches = []

# Если что-то осталось
if current_batches:
    table = Table.from_batches(current_batches)
    pq.write_table(table, f"sig_{part_num}.parquet")
