Fraud loss per session on 3,000 test sessions with a guaranteed takeover (learned fraudster: mean ± std over 2 attacker seeds, 150,000 training steps each).
Exploitability = learned − scripted fraud loss. Learning gain = learned − random.

| Defender | Scripted Rs | Random Rs | Learned Rs | Exploitability Rs | Learning gain Rs | Fraudster receives Rs (learned) | Net Rs/session at p=0.15: scripted → learned |
|---|---|---|---|---|---|---|---|
| Rule-KB | 3,363 | 4,318 | 12,265 | 8,902 | +7,947 | 11,557 | -103 → -1,282 |
| Amount-Threshold | 3,113 | 4,958 | 18,050 | 14,937 | +13,092 | 15,670 | -124 → -2,116 |
| DQN | 1,523 | 4,011 | 18,328 | 16,805 | +14,317 | 10,359 | 337 → -2,261 |
| PPO | 2,182 | 4,203 | 18,702 | 16,520 | +14,499 | 10,571 | 279 → -2,322 |

Learned attack style (mean over attacker seeds):

| Defender | Small % | Medium % | Large % | Stealth % | Fraud txns per attack | Attacks ended by cash-out % |
|---|---|---|---|---|---|---|
| Rule-KB | 0 | 0 | 100 | 0 | 4.6 | 4 |
| Amount-Threshold | 0 | 100 | 0 | 0 | 12.0 | 0 |
| DQN | 0 | 0 | 100 | 100 | 5.7 | 0 |
| PPO | 0 | 0 | 100 | 100 | 5.8 | 0 |
