# FIRM-model
Buy-Hold-Sell predictor

todo
1. remove Severe Multicollinearity features
    -Stock_vs_XLP vs. PX_LAST (0.78)
    -Stock_vs_XLP vs. PBJ_Price (0.97)
    -PX_LAST_MA7 vs. PX_LAST_lag1 (0.80)
These pairs are nearly identical – including both confuses the model.

2. remove near 0 features
    -Revenue_ROC
    -HP_ROC

3. Relabel strategy #############################TRY THIS FIRST#########################
    -The Label (Buy/Hold/Sell) is based on a 10-day forward return threshold, which may not align well with market behavior.

4. Engineer more features
    -Bollinger bands?