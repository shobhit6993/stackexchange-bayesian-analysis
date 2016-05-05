data
{
  int<lower=0> n_sites;  // number  of  groups
  int<lower=0>n_train;   // number of observations in training (sum over all groups)
  int<lower=0>n_test;   // number of observations in testing per site (sum over all groups)
  int<lower=0> n_feat;  // number of columns in X (equal for all websites)
  
  int site_id_train[n_train];   // maps each training instance to its group
  int site_id_test[n_test];     // maps each test instance to its group

  real y_train[n_train];       // response variable for training (union over all groups)
  matrix[n_train, n_feat] X_train;  // feature matrix for training (union over all groups)
  matrix[n_test, n_feat] X_test;  // feature matrix for testing (union over all groups)
  
  real mu_mu_b;                    // mean for hyperprior for mu of coeffs
  real<lower=0> sigma_mu_b;  // sigma for hyperprior for mu of coeff
  real<lower=0> l_sigma_b;     // lower limit for uniform hyperprior for sigma of coeff
  real<lower=0> u_sigma_b;     // upper limit for uniform hyperprior for sigma of coeff
}

parameters
{
  matrix[n_sites, n_feat] b;
  vector<lower=0>[n_sites] eps;
  real mu_b;
  real<lower=0> sigma_b;
}

transformed parameters
{

}

model
{
  vector[n_train] linpred;
  int s;
  
  // hyperprior for mu and sigma of coeff
  mu_b ~ normal(mu_mu_b, sigma_mu_b);
  sigma_b ~ uniform(l_sigma_b, u_sigma_b);
  
  // prior for coeff
  for(i in 1:n_feat) {
    for(k in 1:n_sites) {
      b[k, i] ~ normal(mu_b, sigma_b);  
    }
  }
  
  // prior for sd of obs
  for(k in 1:n_sites) {
    eps[k] ~ uniform(0,5);
  }

  for(i in 1:n_train) {
    s <- site_id_train[i];
    linpred[i] <- X_train[i, ]*b[s, ]';
  }
  
  for(i in 1:n_train) {
    s <- site_id_train[i];
    y_train[i] ~ normal(linpred[i], eps[s]);
  }
}

generated quantities
{
  real y_test[n_test];    // response variable for testing per site
  vector[n_test] linpred;
  int s;
  for(i in 1:n_test) {
    s <- site_id_test[i];
    linpred[i] <- X_test[i, ]*b[s, ]';
  }
  
  for(i in 1:n_test) {
    s <- site_id_test[i];
    y_test[i] <- normal_rng(linpred[i], eps[s]);
  }
}