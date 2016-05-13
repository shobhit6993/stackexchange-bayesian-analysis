data
{
  int<lower=0> n_train;   // number of observations in training
  int<lower=0> n_test;   // number of observations in testing
  int<lower=0> n_feat;       // number of columns in X
  
  real y_train[n_train];       // response variable for training
  matrix[n_train, n_feat] X_train;  // feature matrix for training
  matrix[n_test, n_feat] X_test;  // feature matrix for testing
  
  
  vector[n_feat] mu_b;       // mean for priors of coeffs
  vector<lower=0.001>[n_feat] sigma_b;  // sigma for priors of coeffs
}

parameters
{
  vector[n_feat] b;
  real<lower=0.001> eps;
}

transformed parameters
{

}

model
{
  vector[n_train] linpred;
  
  // prior for coeff
  for(i in 1:n_feat) {
    b[i] ~ normal(mu_b[i], sigma_b[i]);  
  }
  
  // prior for sd of obs
  eps ~ uniform(0,5);

  linpred <- X_train*b;
  
  for(i in 1:n_train) {
    y_train[i] ~ normal(linpred[i], eps);
  }
}

generated quantities
{
  vector[n_test] y_test;    // response variable for testing
  vector[n_test] linpred;
  
  linpred <- X_test*b;
  for(i in 1:n_test) {
    y_test[i] <- normal_rng(linpred[i], eps);
  }
}