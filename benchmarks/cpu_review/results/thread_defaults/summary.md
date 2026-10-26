# Thread defaults: AMD Ryzen 9 9950X 16-Core Processor

Affinity exposes 32 logical CPUs and 16 physical cores.

| Profile | Torch threads | Native threads | b1_s32 ms | b8_s32 ms |
|---|---:|---:|---:|---:|
| hf_logical | 32 | — | 30.37 | 96.99 |
| hf_physical | 16 | — | 23.57 | 71.77 |
| engine_default | 32 | 16 | 17.21 | 80.50 |
| engine_logical | 32 | 32 | 54.13 | 375.22 |
| engine_physical | 16 | 16 | 13.30 | 59.00 |
| engine_default_passive | 32 | 16 | 13.65 | 59.69 |
