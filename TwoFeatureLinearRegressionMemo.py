MEMO = """
==================== WRITTEN MEMO ====================
 WRITTEN MEMO: Two-feature linear regression (bmi + age -> expenses)
 ---------------------------------------------------------------------------
 Q1. Did adding `age` improve R^2 over the `bmi`-only baseline? By how much?
 ---------------------------------------------------------------------------
 Yes, adding age improved the model: R^2 rose from roughly 3.9% for the bmi-only
 baseline to 11.73% for the two-feature model, a gain of about 7.8%.
 Error also fell, with RMSE dropping from about 11,864 to 11,374 and MAE from
 about 9,172 to 9,032, so the improvement is real but modest.

 ---------------------------------------------------------------------------
 Q2. Do the Normal Equation and Gradient Descent weights agree with each
     other? Why should they, in theory?
 ---------------------------------------------------------------------------
 Yes, the two methods agree: both give an intercept of -6437.35, a bmi weight
 of 333.39 and an age weight of 241.90, with a difference of 0.000000 on every
 coefficient. In theory they should agree because the MSE of a
 linear model is a convex, bowl-shaped function with a single minimum. The
 Normal Equation solves directly for that minimum, while Gradient Descent
 walks downhill and reaches the same point as Normal Equation when the learning rate and the
 number of epochs are adequate.

 ---------------------------------------------------------------------------
 Q3. In plain language, what does the sign and size of w2 (age) tell a
     non-technical stakeholder?
 ---------------------------------------------------------------------------
 The age weight w2 = 241.90 is positive, so older of customers' age could tend to have
 higher expenses: each extra year of age goes with about $242 more in expenses
 when bmi stays the same. In other words, a 10-year age gap is associated with
 roughly $2,400 higher expenses. This describes a pattern in this dataset and
 does not prove that aging alone causes the higher cost.

 ---------------------------------------------------------------------------
 Q4. Is this two-feature model good enough to deploy? Why or why not?
 ---------------------------------------------------------------------------
 No, I would not deploy this model as it stands. An R^2 of 0.117 (this is not a good number) means about
 88% of the variation in expenses is still unexplained even though the model 
 includes two features already found the minimum intercept and slope, and typical errors of
 roughly $9,000 to $11,400 are too large for pricing or budgeting decisions.
 The metrics were also computed on the same data used for training, with no
 held-out test set, so real-world performance would likely be no better. Next
 steps are to consider and add other likely drivers of cost (such as smoking status, if
 available in the data) and to validate on unseen data before reconsidering
 deployment."""
print(MEMO)