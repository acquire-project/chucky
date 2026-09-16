# Auk Blosc measurements, September 16, 2026

200 configurations, one warmup and five measured repetitions each. The archive retains all 1,200 attempts byte for byte, including one failed warmup and 1,199 successful executions.

The XOR, 1 MiB chunk, 4 KiB block, Blosc-LZ4 byte-shuffle warmup failed with CUDA out of memory. Its five measured repetitions passed. The configuration remains in the table and downloads but is excluded from frontier membership; this failure alone does not establish a hard memory-fit limit.

Auk used the RTX 5070 Laptop GPU (8 GiB), Ryzen AI 9 365, eight allowed logical CPU threads and three compression workers. Desktop GPU usage varied during collection. This is a new experiment; the September 5 summary is retained separately. Source and binary identities, commands, allocation budget, build settings and collection counts are in the linked original metadata.

`python-outcomes-v1` validates the full attempt matrix, successful results, failed execution evidence and summary counts. Metrics summarize successful measured repetitions only. No failed attempt is replaced. Source files and measurements are unchanged; checksums below cover the retained archive.
