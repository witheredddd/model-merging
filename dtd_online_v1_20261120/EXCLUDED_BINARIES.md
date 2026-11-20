# Excluded binaries

Each `ENDPOINT.pt` holds the final parameter vector (86,192,640 float64)
plus the learned modes, the AdamW moments and the controller state.
They are 2.1-4.8 GB each and are not in git; their sha256 is in
`EXCLUDED_BINARIES.sha256`.

| run | bytes |
|---|---:|
| online_dtd_step0_r2 | 2068630506 |
| online_dtd_step0_r3 | 3447713316 |
| online_dtd_two_online | 4826796094 |
| online_dtd_two_frozen | 3447713316 |
| online_dtd_two_reference | 2068630506 |
