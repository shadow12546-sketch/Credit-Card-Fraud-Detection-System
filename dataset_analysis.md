# Dataset analysis

## 1. Dataset overview

- Rows: **1,296,675**
- Columns: **28** (including the target)
- Target column: **`is_fraud`**
- Fraud: **7,506** (0.5789%) | Non-fraud: **1,289,169** (99.4211%)
- Imbalance ratio: **1 : 171.8**
- Time span: 2012-01-01 to 2013-06-21
- Purpose / business problem: *(written assessment - see note at the end)*

## 2. Feature description

| feature | dtype | unique | missing_% | example | min | median | max | likely_meaning | single_feature_AUC |
|---|---|---|---|---|---|---|---|---|---|
| cc_num | int64 | 983 | 0 | 2703186189652095 | 60,416,207,185 | 3.521e+15 | 4,992,346,398,065,154,184 | Card identifier | 0.5016 |
| merchant | str | 693 | 0 | fraud_Rippin, Kub and Mann |  |  |  | Merchant name |  |
| category | str | 14 | 0 | misc_net |  |  |  | Merchant / purchase category |  |
| amt | float64 | 52,928 | 0 | 4.97 | 1 | 47.52 | 2.895e+04 | Transaction amount | 0.8346 |
| gender | str | 2 | 0 | F |  |  |  | Cardholder gender |  |
| city | str | 894 | 0 | Moravian Falls |  |  |  | Cardholder city |  |
| state | str | 51 | 0 | NC |  |  |  | Cardholder state |  |
| zip | int64 | 970 | 0 | 28654 | 1,257 | 4.817e+04 | 99,783 | Cardholder ZIP code | 0.4911 |
| lat | float64 | 968 | 0 | 36.0788 | 20.03 | 39.35 | 66.69 | Cardholder latitude | 0.5046 |
| long | float64 | 969 | 0 | -81.1781 | -165.7 | -87.48 | -67.95 | Cardholder longitude | 0.5122 |
| city_pop | int64 | 879 | 0 | 3495 | 23 | 2,456 | 2,906,700 | Population of cardholder's city | 0.505 |
| job | str | 494 | 0 | Psychologist, counselling |  |  |  | Cardholder occupation |  |
| unix_time | int64 | 1,274,823 | 0 | 1325376018 | 1,325,376,018 | 1.349e+09 | 1,371,816,817 | Transaction unix time | 0.4819 |
| merch_lat | float64 | 1,247,805 | 0 | 36.011293 | 19.03 | 39.37 | 67.51 | Merchant latitude | 0.504 |
| merch_long | float64 | 1,275,745 | 0 | -82.048315 | -166.7 | -87.44 | -66.95 | Merchant longitude | 0.5122 |
| merch_zipcode | float64 | 28,337 | 0 | 28705.0 | -1 | 3.849e+04 | 9.94e+04 | (describe manually) | 0.4909 |
| age | int64 | 83 | 0 | 31 | 14 | 44 | 96 | (describe manually) | 0.5434 |
| hour | int64 | 24 | 0 | 0 | 0 | 14 | 23 | (describe manually) | 0.5851 |
| day | int64 | 31 | 0 | 1 | 1 | 15 | 31 | (describe manually) | 0.5147 |
| month | int64 | 12 | 0 | 1 | 1 | 6 | 12 | (describe manually) | 0.4493 |
| weekday | int64 | 7 | 0 | 1 | 0 | 3 | 6 | (describe manually) | 0.5049 |
| tx_count_card | int64 | 399 | 0 | 2028 | 7 | 2,000 | 3,123 | (describe manually) | 0.3206 |
| distance_km | float64 | 1,296,675 | 0 | 78.59756848823062 | 0.02225 | 78.23 | 152.1 | (describe manually) | 0.5008 |
| night_tx | int64 | 2 | 0 | 1 | 0 | 0 | 1 | (describe manually) | 0.808 |
| weekend | int64 | 2 | 0 | 0 | 0 | 0 | 1 | (describe manually) | 0.4886 |
| avg_amt_card | float64 | 983 | 0 | 87.39321499013806 | 42.95 | 65.09 | 948.8 | (describe manually) | 0.5976 |
| amt_ratio | float64 | 1,166,530 | 0 | 0.0568694034263511 | 0.008693 | 0.6694 | 386.5 | (describe manually) | 0.8036 |

## 3. Feature engineering analysis

Columns already engineered in the file: ['age', 'hour', 'tx_count_card', 'distance_km', 'night_tx', 'weekend', 'avg_amt_card', 'amt_ratio']

Features derived here from raw columns, with measured signal (AUC 0.5 = no signal; card-history features use only *earlier* rows per card):

| feature | AUC | signal_strength | mean_if_fraud | mean_if_legit |
|---|---|---|---|---|
| hour | 0.5851 | 0.1702 | 14.04 | 12.8 |
| weekday | 0.4642 | 0.07169 | 3.317 | 3.555 |
| month | 0.4493 | 0.1015 | 5.586 | 6.145 |
| is_weekend | 0.4681 | 0.06384 | 0.3339 | 0.3977 |
| night_tx | 0.7825 | 0.565 | 0.8617 | 0.2967 |
| distance_km | 0.5008 | 0.001511 | 76.27 | 76.11 |
| tx_count_card | 0.3538 | 0.2924 | 610.1 | 910.6 |
| avg_amt_card | 0.6058 | 0.2117 | 117.5 | 70.12 |
| amt_ratio | 0.8073 | 0.6147 | 6.983 | 0.9867 |

## 4. Data quality report

**Missing values:** none

**Duplicate rows (all columns):** 0

**Outliers (IQR rule, numeric columns):**

| feature | outliers | outlier_% | skew |
|---|---|---|---|
| amt | 67,290 | 5.189 | 42.28 |
| zip | 0 | 0 | 0.07968 |
| lat | 4,679 | 0.3608 | -0.186 |
| long | 49,922 | 3.85 | -1.15 |
| city_pop | 242,674 | 18.72 | 5.594 |
| unix_time | 0 | 0 | 0.003378 |
| merch_lat | 4,967 | 0.3831 | -0.1819 |
| merch_long | 41,994 | 3.239 | -1.147 |
| merch_zipcode | 0 | 0 | 0.2013 |
| age | 2,258 | 0.1741 | 0.6123 |
| hour | 0 | 0 | -0.2828 |
| day | 0 | 0 | 0.03085 |
| month | 0 | 0 | 0.2985 |
| tx_count_card | 0 | 0 | 0.02848 |
| distance_km | 0 | 0 | -0.2362 |
| avg_amt_card | 741 | 0.05715 | 14.4 |
| amt_ratio | 59,138 | 4.561 | 34.81 |

**Invalid-value checks:**

| check | violations |
|---|---|
| amount <= 0 | 0 |
| amount not numeric | 0 |
| lat outside [-90,90] | 0 |
| long outside [-180,180] | 0 |
| merch_lat outside [-90,90] | 0 |
| merch_long outside [-180,180] | 0 |
| unparseable timestamps | 0 |

**Class imbalance:** fraud rate 0.5789%

## 5. Exploratory data analysis

**`amt` by class:**

| is_fraud | count | mean | std | min | 25% | 50% | 75% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|---|
| legit | 1.289e+06 | 67.67 | 154 | 1 | 9.61 | 47.28 | 82.54 | 189.9 | 486.3 | 2.895e+04 |
| fraud | 7,506 | 531.3 | 390.6 | 1.06 | 245.7 | 396.5 | 900.9 | 1,084 | 1,180 | 1,376 |

**Fraud by gender:**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| M | 586,812 | 3,771 | 0.006426 | 1.11 |
| F | 709,863 | 3,735 | 0.005262 | 0.9089 |

**Fraud by state (top 10 / bottom 5, min 30 txns):**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| RI | 550 | 15 | 0.02727 | 4.711 |
| AK | 2,120 | 36 | 0.01698 | 2.934 |
| NV | 5,607 | 47 | 0.008382 | 1.448 |
| CO | 13,880 | 113 | 0.008141 | 1.406 |
| OR | 18,597 | 149 | 0.008012 | 1.384 |
| TN | 17,554 | 140 | 0.007975 | 1.378 |
| NE | 24,168 | 180 | 0.007448 | 1.287 |
| ME | 16,505 | 119 | 0.00721 | 1.246 |
| NH | 8,278 | 59 | 0.007127 | 1.231 |
| OH | 46,480 | 321 | 0.006906 | 1.193 |
| AZ | 10,770 | 37 | 0.003435 | 0.5935 |
| HI | 2,559 | 7 | 0.002735 | 0.4726 |
| MT | 11,754 | 32 | 0.002722 | 0.4703 |
| CT | 7,702 | 16 | 0.002077 | 0.3589 |
| ID | 5,545 | 11 | 0.001984 | 0.3427 |

**Fraud by category:**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| shopping_net | 97,543 | 1,713 | 0.01756 | 3.034 |
| misc_net | 63,287 | 915 | 0.01446 | 2.498 |
| grocery_pos | 123,638 | 1,743 | 0.0141 | 2.435 |
| shopping_pos | 116,672 | 843 | 0.007225 | 1.248 |
| gas_transport | 131,659 | 618 | 0.004694 | 0.8109 |
| misc_pos | 79,655 | 250 | 0.003139 | 0.5422 |
| grocery_net | 45,452 | 134 | 0.002948 | 0.5093 |
| travel | 40,507 | 116 | 0.002864 | 0.4947 |
| entertainment | 94,014 | 233 | 0.002478 | 0.4281 |
| personal_care | 90,758 | 220 | 0.002424 | 0.4188 |
| kids_pets | 113,035 | 239 | 0.002114 | 0.3653 |
| food_dining | 91,461 | 151 | 0.001651 | 0.2852 |
| home | 123,115 | 198 | 0.001608 | 0.2778 |
| health_fitness | 85,879 | 133 | 0.001549 | 0.2675 |

**Fraud by hour:**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| 0 | 42,502 | 635 | 0.01494 | 2.581 |
| 1 | 42,869 | 658 | 0.01535 | 2.652 |
| 2 | 42,656 | 625 | 0.01465 | 2.531 |
| 3 | 42,769 | 609 | 0.01424 | 2.46 |
| 4 | 41,863 | 46 | 0.001099 | 0.1898 |
| 5 | 42,171 | 60 | 0.001423 | 0.2458 |
| 6 | 42,300 | 40 | 0.0009456 | 0.1634 |
| 7 | 42,203 | 56 | 0.001327 | 0.2292 |
| 8 | 42,505 | 49 | 0.001153 | 0.1991 |
| 9 | 42,185 | 47 | 0.001114 | 0.1925 |
| 10 | 42,271 | 40 | 0.0009463 | 0.1635 |
| 11 | 42,082 | 42 | 0.0009981 | 0.1724 |
| 12 | 65,257 | 67 | 0.001027 | 0.1774 |
| 13 | 65,314 | 80 | 0.001225 | 0.2116 |
| 14 | 64,885 | 86 | 0.001325 | 0.229 |
| 15 | 65,391 | 79 | 0.001208 | 0.2087 |
| 16 | 65,726 | 76 | 0.001156 | 0.1998 |
| 17 | 65,450 | 78 | 0.001192 | 0.2059 |
| 18 | 66,051 | 81 | 0.001226 | 0.2118 |
| 19 | 65,508 | 81 | 0.001236 | 0.2136 |
| 20 | 65,098 | 62 | 0.0009524 | 0.1645 |
| 21 | 65,533 | 74 | 0.001129 | 0.1951 |
| 22 | 66,982 | 1,931 | 0.02883 | 4.98 |
| 23 | 67,104 | 1,904 | 0.02837 | 4.902 |

**Fraud by day of week (0=Mon):**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| 0 | 122,237 | 789 | 0.006455 | 1.115 |
| 1 | 135,206 | 934 | 0.006908 | 1.193 |
| 2 | 150,510 | 956 | 0.006352 | 1.097 |
| 3 | 152,319 | 999 | 0.006559 | 1.133 |
| 4 | 221,191 | 1,322 | 0.005977 | 1.032 |
| 5 | 261,509 | 1,266 | 0.004841 | 0.8363 |
| 6 | 253,703 | 1,240 | 0.004888 | 0.8443 |

**Fraud by month:**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| 1 | 104,727 | 849 | 0.008107 | 1.4 |
| 2 | 97,657 | 853 | 0.008735 | 1.509 |
| 3 | 143,789 | 938 | 0.006523 | 1.127 |
| 4 | 134,970 | 678 | 0.005023 | 0.8678 |
| 5 | 146,875 | 935 | 0.006366 | 1.1 |
| 6 | 143,811 | 688 | 0.004784 | 0.8265 |
| 7 | 86,596 | 331 | 0.003822 | 0.6603 |
| 8 | 87,359 | 382 | 0.004373 | 0.7554 |
| 9 | 70,652 | 418 | 0.005916 | 1.022 |
| 10 | 68,758 | 454 | 0.006603 | 1.141 |
| 11 | 70,421 | 388 | 0.00551 | 0.9518 |
| 12 | 141,060 | 592 | 0.004197 | 0.725 |

## 6. Fraud pattern discovery

Buckets with at least 30 transactions, ranked by fraud rate (lift = rate / overall rate):

**Highest-risk categories:**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| shopping_net | 97,543 | 1,713 | 0.01756 | 3.034 |
| misc_net | 63,287 | 915 | 0.01446 | 2.498 |
| grocery_pos | 123,638 | 1,743 | 0.0141 | 2.435 |
| shopping_pos | 116,672 | 843 | 0.007225 | 1.248 |
| gas_transport | 131,659 | 618 | 0.004694 | 0.8109 |

**Highest-risk hours:**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| 22 | 66,982 | 1,931 | 0.02883 | 4.98 |
| 23 | 67,104 | 1,904 | 0.02837 | 4.902 |
| 1 | 42,869 | 658 | 0.01535 | 2.652 |
| 0 | 42,502 | 635 | 0.01494 | 2.581 |
| 2 | 42,656 | 625 | 0.01465 | 2.531 |

**Highest-risk states:**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| RI | 550 | 15 | 0.02727 | 4.711 |
| AK | 2,120 | 36 | 0.01698 | 2.934 |
| NV | 5,607 | 47 | 0.008382 | 1.448 |
| CO | 13,880 | 113 | 0.008141 | 1.406 |
| OR | 18,597 | 149 | 0.008012 | 1.384 |

**Amount deciles:**

| k | transactions | frauds | fraud_rate | lift_vs_avg |
|---|---|---|---|---|
| (136.67, 28948.9] | 129,658 | 5,724 | 0.04415 | 7.626 |
| (7.75, 15.74] | 129,397 | 714 | 0.005518 | 0.9532 |
| (15.74, 32.13] | 129,630 | 672 | 0.005184 | 0.8955 |
| (4.11, 7.75] | 129,884 | 157 | 0.001209 | 0.2088 |
| (94.68, 136.67] | 129,666 | 135 | 0.001041 | 0.1799 |
| (47.52, 60.94] | 129,605 | 81 | 0.000625 | 0.108 |
| (0.999, 4.11] | 129,795 | 18 | 0.0001387 | 0.02396 |
| (32.13, 47.52] | 129,706 | 4 | 3.084e-05 | 0.005327 |
| (75.03, 94.68] | 129,665 | 1 | 7.712e-06 | 0.001332 |
| (60.94, 75.03] | 129,669 | 0 | 0 | 0 |

## 7. Feature importance estimate (no model trained)

Mutual information with the label on a sample, plus single-feature AUC. Higher = more predictive signal on its own; interactions are not captured.

| index | mutual_info | tier | single_AUC |
|---|---|---|---|
| is_weekend | 0.0209 | HIGH | 0.4681 |
| amt | 0.0177 | HIGH | 0.8346 |
| weekend | 0.01528 | HIGH | 0.4886 |
| amt_ratio | 0.01286 | HIGH | 0.8036 |
| night_tx | 0.01177 | HIGH | 0.808 |
| hour | 0.00738 | HIGH | 0.5851 |
| weekday | 0.006821 | HIGH | 0.5049 |
| city | 0.006104 | HIGH |  |
| tx_count_card | 0.00442 | HIGH | 0.3206 |
| avg_amt_card | 0.00421 | MEDIUM | 0.5976 |
| merchant | 0.003882 | MEDIUM |  |
| month | 0.003653 | MEDIUM | 0.4493 |
| job | 0.003024 | MEDIUM |  |
| zip | 0.002694 | MEDIUM | 0.4911 |
| lat | 0.002532 | MEDIUM | 0.5046 |
| long | 0.002383 | MEDIUM | 0.5122 |
| city_pop | 0.002369 | MEDIUM | 0.505 |
| category | 0.002228 | LOW |  |
| day | 0.001406 | LOW | 0.5147 |
| age | 0.00127 | LOW | 0.5434 |
| state | 0.0003199 | LOW |  |
| merch_long | 0.0001925 | LOW | 0.5122 |
| merch_lat | 7.625e-05 | LOW | 0.504 |
| merch_zipcode | 7.59e-05 | LOW | 0.4909 |
| gender | 2.098e-05 | LOW |  |
| distance_km | 0 | LOW | 0.5008 |

## 8. Machine-learning readiness (measured facts)

- Positive samples: 7,506 (rule of thumb: >1,000 positives is comfortable for gradient boosting)
- Categorical cardinalities: {'merchant': 693, 'category': 14, 'gender': 2, 'city': 894, 'state': 51, 'job': 494}
- Highly skewed numeric columns (|skew|>2): ['amt', 'city_pop', 'avg_amt_card', 'amt_ratio']
- Tree models need no scaling; linear/NN baselines do. SMOTE is usually unnecessary when positives are in the thousands - try `scale_pos_weight` and threshold tuning first.

## 9-11. Deployment readiness, improvements, final assessment

These need interpretation of the tables above. Paste this report (or upload it) and I will write them with your real numbers.
