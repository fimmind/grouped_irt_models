# Duolingo grouped IRT balanced-accuracy optimization

## Setup

- Dataset: `data/raw/duolingo_hlr/learning_traces.csv.gz`
- Language pair: `en->es`
- Dense matrix size: `500 x 1500`
- Known threshold: `0.8`
- Imputation strategy: `lexeme_majority`
- Difficulty source: `recall_mean`
- q values: `50, 100, 1000`
- Coarse repeats: `1`, fine repeats: `2`
- Candidate models: `57`

## Coarse stage ranking (mean BA over q=50,100)

| model | mean_ba |
| --- | --- |
| response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9076 |
| residual_sign12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9073 |
| response12_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9070 |
| residual_sign12_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9068 |
| response16_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9067 |
| residual12_s701_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9063 |
| response16_g16_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9063 |
| residual12_s1301_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9059 |
| residual16_s701_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9056 |
| residual12_s701_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9053 |
| residual12_s1301_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9050 |
| residual16_s701_g16_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9048 |

## Best model per q (fine stage)

| q | model | balanced_accuracy | pr_auc | auc | accuracy | log_loss | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9059 | 0.9674 | 0.9482 | 0.9079 | 0.3242 | 0.0897 |
| 100 | response16_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9092 | 0.9692 | 0.9524 | 0.9112 | 0.2877 | 0.0782 |
| 1000 | response16_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9164 | 0.9736 | 0.9598 | 0.9192 | 0.2157 | 0.0621 |

## Fine stage full results

| q | model | balanced_accuracy | pr_auc | auc | accuracy | log_loss | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9059 | 0.9674 | 0.9482 | 0.9079 | 0.3242 | 0.0897 |
| 50 | residual_sign12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9052 | 0.9680 | 0.9487 | 0.9076 | 0.3259 | 0.0902 |
| 50 | response12_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9051 | 0.9672 | 0.9481 | 0.9076 | 0.3053 | 0.0840 |
| 50 | residual_sign12_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9045 | 0.9678 | 0.9485 | 0.9077 | 0.3067 | 0.0844 |
| 50 | residual12_s701_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9045 | 0.9687 | 0.9494 | 0.9085 | 0.3154 | 0.0867 |
| 50 | response16_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9043 | 0.9668 | 0.9479 | 0.9071 | 0.3413 | 0.0955 |
| 50 | residual12_s1301_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9042 | 0.9685 | 0.9496 | 0.9088 | 0.3149 | 0.0865 |
| 50 | response16_g16_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9042 | 0.9667 | 0.9479 | 0.9072 | 0.3206 | 0.0887 |
| 50 | residual12_s701_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9036 | 0.9686 | 0.9492 | 0.9079 | 0.2972 | 0.0814 |
| 50 | residual16_s701_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9036 | 0.9675 | 0.9479 | 0.9081 | 0.3319 | 0.0921 |
| 50 | residual12_s1301_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9031 | 0.9685 | 0.9495 | 0.9085 | 0.2968 | 0.0813 |
| 50 | residual16_s701_g16_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9026 | 0.9675 | 0.9478 | 0.9078 | 0.3123 | 0.0860 |
| 100 | response16_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9092 | 0.9692 | 0.9524 | 0.9112 | 0.2877 | 0.0782 |
| 100 | response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9090 | 0.9693 | 0.9515 | 0.9107 | 0.2770 | 0.0757 |
| 100 | residual_sign12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9088 | 0.9700 | 0.9522 | 0.9112 | 0.2777 | 0.0758 |
| 100 | response12_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9087 | 0.9690 | 0.9513 | 0.9103 | 0.2665 | 0.0734 |
| 100 | response16_g16_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9087 | 0.9690 | 0.9524 | 0.9106 | 0.2748 | 0.0750 |
| 100 | residual_sign12_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9081 | 0.9697 | 0.9520 | 0.9108 | 0.2668 | 0.0734 |
| 100 | residual12_s701_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9079 | 0.9712 | 0.9532 | 0.9105 | 0.2693 | 0.0734 |
| 100 | residual16_s701_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9079 | 0.9706 | 0.9530 | 0.9115 | 0.2807 | 0.0762 |
| 100 | residual12_s1301_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9074 | 0.9712 | 0.9540 | 0.9112 | 0.2688 | 0.0732 |
| 100 | residual16_s701_g16_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9074 | 0.9706 | 0.9530 | 0.9111 | 0.2688 | 0.0734 |
| 100 | residual12_s701_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9074 | 0.9711 | 0.9530 | 0.9102 | 0.2595 | 0.0714 |
| 100 | residual12_s1301_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9071 | 0.9712 | 0.9539 | 0.9110 | 0.2589 | 0.0711 |
| 1000 | response16_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9164 | 0.9736 | 0.9598 | 0.9192 | 0.2157 | 0.0621 |
| 1000 | response16_g16_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9163 | 0.9736 | 0.9598 | 0.9191 | 0.2140 | 0.0620 |
| 1000 | response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9159 | 0.9720 | 0.9562 | 0.9182 | 0.2212 | 0.0642 |
| 1000 | residual_sign12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9159 | 0.9723 | 0.9568 | 0.9184 | 0.2204 | 0.0639 |
| 1000 | response12_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9158 | 0.9719 | 0.9562 | 0.9182 | 0.2202 | 0.0641 |
| 1000 | residual_sign12_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9157 | 0.9722 | 0.9567 | 0.9182 | 0.2193 | 0.0639 |
| 1000 | residual16_s701_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9146 | 0.9757 | 0.9609 | 0.9183 | 0.2142 | 0.0620 |
| 1000 | residual16_s701_g16_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9145 | 0.9757 | 0.9609 | 0.9183 | 0.2127 | 0.0619 |
| 1000 | residual12_s701_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9138 | 0.9749 | 0.9584 | 0.9164 | 0.2164 | 0.0632 |
| 1000 | residual12_s1301_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.9138 | 0.9755 | 0.9602 | 0.9174 | 0.2143 | 0.0624 |
| 1000 | residual12_s1301_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9138 | 0.9755 | 0.9602 | 0.9174 | 0.2132 | 0.0624 |
| 1000 | residual12_s701_g12_tau1p8_c8p0_observed_ba_opt_shrunk | 0.9138 | 0.9749 | 0.9584 | 0.9162 | 0.2154 | 0.0632 |
