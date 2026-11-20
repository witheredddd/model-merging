# Excluded binaries

Six `ENDPOINT.pt` payloads, 2.1-4.8 GB each: the final 86,192,640-coordinate
parameter vector, the optimizer state and the learned modes. Their hashes are in
`EXCLUDED_BINARIES.sha256` (paths relative to the experiment output directory).

The V2 and V2.1 payloads for the same arm hash differently because the file also
stores the source lock; the parameter vectors themselves are bit-identical
(verified with torch.equal, max abs difference 0.0).

| bytes | file |
|---:|---|
| 4826796094 | v2_stageA_online/ENDPOINT.pt |
| 3447713316 | v2_stageA_frozen/ENDPOINT.pt |
| 2068630506 | v2_stageA_reference/ENDPOINT.pt |
| 4826796094 | v21_online/ENDPOINT.pt |
| 3447713316 | v21_frozen/ENDPOINT.pt |
| 2068630506 | v21_reference/ENDPOINT.pt |
