library (arm)
library (gtools)
library(rstan)
library(ggmcmc)
library(cvTools)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

model_pooled <- stan_model(file = "~/code/stackexchange-bayesian-analysis/bayesian-regression-pooled.stan")
model_hierarchical <- stan_model(file = "~/code/stackexchange-bayesian-analysis/bayesian-regression-hierarchical.stan")

# reads csv files and returns the extracted dataframe
read_data <- function(folder_path) {
  files = list.files(path=folder_path, pattern="*.csv", full.names=T, recursive=T)
  files = sort(files)
  
  user = list()
  post = list()
  
  post_dont_scale = c("ques_id", "ans_id", "user_id")
  user_dont_scale = c("user_id")
  
  mu_for_scale = 0  # mean of time_to_ans colm
  sd_for_scale = 0  # sd of time_to_ans colm
  for (f in files) {
    if (grepl("posts", f)) {
      df = read.csv(file=f, header=T, sep=",")
      # scale all colms except those in post_dont_scale
      col_list = match(post_dont_scale, colnames(df))
      
      stemp = scale(df[, -col_list])
      mu_for_scale = attr(stemp, 'scaled:center')['time_to_ans']
      sd_for_scale = attr(stemp, 'scaled:scale')['time_to_ans']
      df[, -col_list] = stemp
      
      i = length(post)
      post[[i+1]] = df
    }
    if (grepl("users", f)) {
      df = read.csv(file=f, header=T, sep=",")
      # scale all colms except those in post_dont_scale
      col_list = match(user_dont_scale, colnames(df))
      df[, -col_list] = scale(df[, -col_list])
      i = length(user)
      user[[i+1]] = df
    }
  }
  
  # join (left outer join) user df and post df by the user_id column
  n_sites = length(post)
  data = list()
  for (i in 1:n_sites) {
    data[[i]] = merge(x=post[[i]], y=user[[i]], by="user_id", all.x=T)
  }
  ret = list("data"=data, "mu_for_scale"=mu_for_scale, 
             "sd_for_scale"=sd_for_scale)
  return(ret)
}

# returns the mean squared error, given predicitons vector and target vector
mean_sq_error <- function(pred, target) {
  return(sqrt(mean((pred-target)^2)))
}

# fits a seperate regression model to each website, creating a hierarchy
no_pooling <- function(data) {
  pooled_data = data[[1]]
  n_sites = length(data)
  for(i in 2:n_sites) {
    pooled_data = rbind(pooled_data, data[[i]])
  }
  n = dim(pooled_data)[1]
  
  # create site_id mapping
  site_id = vector(length = n)
  l = 1
  for(i in 1:n_sites) {
    n_this_site = dim(data[[i]])[1]
    site_id[l:(l+n_this_site-1)] = i
    l = l + n_this_site
  }
  
  col_list = c("view_count", "score", "comment_count", "favorite_count", "views", "downvotes", "reputation", "upvotes")
  feat_list = match(col_list, colnames(data[[1]]))
  
  # get cv folds
  set.seed(6)
  K=5
  mse = 0
  cv = cvFolds(n=n, K=K)
  for(k in 1:K) {
    # k=1
    n_feat = length(feat_list)
    a = table(cv$which)
    n_test = a[names(a)==k][[1]]
    n_train = n - n_test
    X_train = matrix(nrow=n_train, ncol=n_feat)
    X_test = matrix(nrow=n_test, ncol=n_feat)
    y_train = vector(length = n_train)
    y_test = vector(length = n_test)
    site_id_train = vector(length = n_train)
    site_id_test = vector(length = n_test)
    te = 1
    tr = 1
    for(i in 1:n) {
      r = cv$subsets[i]
      if(cv$which[i] == k) {
        X_test[te,] = as.numeric(pooled_data[r,col_list])
        y_test[te] = pooled_data[r, "time_to_ans"]
        site_id_test[te] = site_id[r]
        te = te + 1
      }
      if(cv$which[i] != k) {
        X_train[tr,] = as.numeric(pooled_data[r,col_list])
        y_train[tr] = pooled_data[r, "time_to_ans"]
        site_id_train[tr] = site_id[r]
        tr = tr + 1
      }
    }
    
    # add dummy 1's colm to X_train, X_test
    ones <- matrix(1, nrow=n_train)
    X_train <- cbind(ones, X_train)
    ones <- matrix(1, nrow=n_test)
    X_test <- cbind(ones, X_test)
    n_feat = n_feat+1
    
    obs <- list(n_sites=n_sites, n_train=n_train, n_test=n_test, n_feat=n_feat,
                site_id_train=site_id_train, site_id_test=site_id_test,
                y_train=y_train, X_train=X_train, X_test=X_test, 
                mu_mu_b=0, sigma_mu_b=10, l_sigma_b=0.001, u_sigma_b=10)
    
    # initialization
    map_est = optimizing(model_hierarchical, data=obs, seed=6)
    b_init = matrix(nrow=n_sites, ncol=n_feat)
    eps_init = vector()
    for(i in 1:n_sites) {
      # i=1
      e = paste("eps[", i, "]", sep="")
      eps_init[[i]] = map_est$par[e]
      for(j in 1:n_feat) {
        # j=1
        b = paste("b[", paste(i, j, sep=","), "]", sep="")
        b_init[i, j] = map_est$par[b]
      }
    }
    init <- list(list(b=b_init, eps=eps_init, 
                      mu_b=map_est$par["mu_b"], sigma_b=map_est$par["sigma_b"]))
    
    num_iter = 5000
    warmup = 3000
    
    #NUTS
    nuts <- sampling(model_hierarchical, data=obs, iter=num_iter, warmup=warmup, thin=2, chains=1, seed=6)
    samples = rstan::extract(nuts, c("linpred"))
    y_pred = vector(length = n_test)
    for(i in 1:n_test) {
      y_pred[i] = mean(samples$linpred[,i])
    }
    print(mean_sq_error(y_pred, y_test))
    mse = mse + mean_sq_error(y_pred, y_test)
  }
  print("Average mse")
  print(mse/K)
}

# fits a single Bayesian linear regression to the entire data
pooling <- function(data) {
  pooled_data = data[[1]]
  n_sites = length(data)
  for(i in 2:n_sites) {
    pooled_data = rbind(pooled_data, data[[i]])
  }
  n = dim(pooled_data)[1]
  col_list = c("view_count", "score", "comment_count", "favorite_count", "views", "downvotes", "reputation", "upvotes")
  feat_list = match(col_list, colnames(data[[1]]))
  
  # get cv folds
  set.seed(6)
  K=5
  mse = 0
  cv = cvFolds(n=n, K=K)
  for(k in 1:K) {
    # k=1
    n_feat = length(feat_list)
    a = table(cv$which)
    n_test = a[names(a)==k][[1]]
    n_train = n - n_test
    X_train = matrix(nrow=n_train, ncol=n_feat)
    X_test = matrix(nrow=n_test, ncol=n_feat)
    y_train = vector(length = n_train)
    y_test = vector(length = n_test)
    te = 1
    tr = 1
    for(i in 1:n) {
      r = cv$subsets[i]
      if(cv$which[i] == k) {
        X_test[te,] = as.numeric(pooled_data[r,col_list])
        y_test[te] = pooled_data[r, "time_to_ans"]
        te = te + 1
      }
      if(cv$which[i] != k) {
        X_train[tr,] = as.numeric(pooled_data[r,col_list])
        y_train[tr] = pooled_data[r, "time_to_ans"]
        tr = tr + 1
      }
    }
    
    # add dummy 1's colm to X_train, X_test
    ones <- matrix(1, nrow=n_train)
    X_train <- cbind(ones, X_train)
    ones <- matrix(1, nrow=n_test)
    X_test <- cbind(ones, X_test)
    n_feat = n_feat+1
    
    obs <- list(n_train=n_train, n_test=n_test, n_feat=n_feat, 
                 y_train=y_train, X_train=X_train, X_test=X_test, 
                 mu_b=rep(0, n_feat), sigma_b=rep(200, n_feat))
    
    # initialization
    map_est = optimizing(model_pooled, data=obs, seed=6)
    b_init = vector()
    for(i in 1:n_feat) {
      b = paste("b[", i, "]", sep="")
      b_init[[i]] = map_est$par[b]
    }
    init <- list(list(b=b_init, eps=map_est$par["eps"]))
    
    num_iter = 5000
    warmup = 3000
  
    #NUTS
    nuts <- sampling(model_pooled, data=obs, init=init, iter=num_iter, warmup=warmup, thin=2, chains=1, seed=6)
    samples = rstan::extract(nuts, c("linpred"))
    y_pred = vector(length = n_test)
    for(i in 1:n_test) {
      y_pred[i] = mean(samples$linpred[,i])
    }
    print(mean_sq_error(y_pred, y_test))
    mse = mse + mean_sq_error(y_pred, y_test)
  }
  print("Average mse")
  print(mse/K)
}

folder_path = "/home/shobhit/code/stackexchange-bayesian-analysis/data"
ret = read_data(folder_path)
data = ret$data
mu_for_scale = ret$mu_for_scale
sd_for_scale = ret$sd_for_scale
n_sites = length(data)

# LM without pooling
no_pooling(data)

# LM pooling data from all the websites
pooling(data)
