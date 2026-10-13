# Summarisation evaluation

Model `sshleifer/distilbart-cnn-12-6`, chunk 700 tokens, overlap 60, batch 4.

| doc_id | mode | rouge_l | compression | reading_ease | latency_s | peak_mb | forward_passes |
|---|---|---|---|---|---|---|---|
| mobile-money | abstractive | 0.2391 | 0.2359 | 54.9 | 8.567 | 165.25 | 1 |
| mobile-money | map-reduce | 0.2391 | 0.2359 | 54.9 | 1.243 | 0.11 | 1 |
| model-rollback | abstractive | 0.1728 | 0.2 | 66.9 | 1.154 | 0.09 | 1 |
| model-rollback | map-reduce | 0.1728 | 0.2 | 66.9 | 1.209 | 0.09 | 1 |
| retrieval-latency | abstractive | 0.1159 | 0.1627 | 23.4 | 0.878 | 0.07 | 1 |
| retrieval-latency | map-reduce | 0.1159 | 0.1627 | 23.4 | 0.877 | 0.07 | 1 |
