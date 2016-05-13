library(cvTools)

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




linear_regression <- function(formula, data, n_sites) {
  if (n_sites == 1) {
    print(cvFit(object=lm, formula=formula, data = data, seed=6))
    formula = time_to_ans ~ view_count + score + comment_count + favorite_count + views + downvotes + reputation + upvotes
    data = pooled_data
    fit <- lm(formula=formula, data=data)
  }
  if (n_sites > 1) {
    for (i in 1:n_sites) {
      print(cvFit(object=lm, formula=formula, data = data[[i]], seed=6))
      formula = time_to_ans ~ view_count + score + comment_count + favorite_count + views + downvotes + reputation + upvotes
      data = data[[1]]
      fit <- lm(formula=formula, data=data)
    }
  }
}

# fits a seperate linear model to each website
no_pooling <- function(data) {
  linear_regression(time_to_ans ~ view_count + score + comment_count + favorite_count + views + downvotes + reputation + upvotes, data, length(data))
  # view_count and comment_count features gave best result with Linear Regression using merged
  # actually just view_count
}

pooling <- function(data) {
  pooled_data = data[[1]]
  n_sites = length(data)
  for(i in 2:n_sites) {
    pooled_data = rbind(pooled_data, data[[i]])
  }
    
  formula = (time_to_ans ~ view_count + score + comment_count + favorite_count + views + downvotes + reputation + upvotes)
  linear_regression(formula, data, 1)
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
