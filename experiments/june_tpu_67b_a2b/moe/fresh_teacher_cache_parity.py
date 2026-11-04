"""Cache parity: every row of the Levanter cache equals its parquet row (input_ids == ids, assistant_masks == loss), in order."""
import sys, glob, json, numpy as np, pyarrow.parquet as pq
from levanter.store.jagged_array import JaggedArrayStore
cache, parquet = sys.argv[1], sys.argv[2]
shards = sorted(glob.glob(cache + "/train/part-*"))
assert len(shards) == 1, shards
ids = JaggedArrayStore.open(shards[0] + "/input_ids", mode="r", dtype=np.int32)
msk = JaggedArrayStore.open(shards[0] + "/assistant_masks", mode="r", dtype=np.int32)
n = ids.num_rows; t = pq.ParquetFile(parquet); rows = t.metadata.num_rows
assert n == rows == msk.num_rows, (n, rows, msk.num_rows)
bad = 0; tok = 0; trained = 0; i = 0
for b in t.iter_batches(batch_size=256, columns=["ids", "loss"]):
    for r in b.to_pylist():
        a = np.asarray(ids[i]); m = np.asarray(msk[i])
        if not (np.array_equal(a, np.asarray(r["ids"], dtype=a.dtype)) and np.array_equal(m.astype(np.int8), np.asarray(r["loss"], dtype=np.int8))):
            bad += 1
            if bad <= 3: print("MISMATCH row", i, len(a), len(r["ids"]), flush=True)
        tok += len(a); trained += int(m.sum()); i += 1
    print("checked", i, flush=True)
print(json.dumps(dict(rows=n, mismatched=bad, tokens=tok, trained_tokens=trained)))
print("CACHE_PARITY_OK" if bad == 0 else "CACHE_PARITY_FAILED")
