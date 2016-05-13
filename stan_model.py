# coding=utf-8
import pystan
import numpy as np
import os
import pandas as pd
import pdb
import pylab as py
import seaborn as sns
import matplotlib


def load_data(filepath, train_test_split = 0.9, logit=False, more_drop=False):
    '''
    For a given file, loads in data and returns a dictionary with data values
    filepath should contain posts.csv and users.csv
    Returns:
        tuple of (dict of stan data, beta headers)
    '''

    # reads files into one data frame
    if type(filepath) == list:
        data = []
        for dir in filepath:
            for file in os.listdir(dir):
                if 'posts' in file:
                    post_data = pd.read_csv(os.path.join(dir, file))
                elif 'users' in file:
                    user_data = pd.read_csv(os.path.join(dir, file))
                else:
                    continue

            data.append(pd.merge(post_data, user_data, how='inner', on='user_id', left_index=True))
        all_data = pd.concat(data)

    # only reading one stack exchange & therefore one file
    else:
        for file in os.listdir(filepath):
            if 'posts' in file:
                post_data = pd.read_csv(os.path.join(filepath, file))
            elif 'users' in file:
                user_data = pd.read_csv(os.path.join(filepath, file))

        # join files on "user_id" - note - way more users than there are posts. This should get rid of duplicate users
        all_data = pd.merge(post_data, user_data, how='inner', on='user_id', left_index=True)

    # data cleaning and prep
    all_data = all_data[all_data['time_to_ans'] > 0]

    # split into train/test sets
    train_test_ind = np.arange(all_data.shape[0])
    np.random.shuffle(train_test_ind)
    break_ind = int(np.floor(all_data.shape[0] * train_test_split))
    train_inds = train_test_ind[0:break_ind]
    test_inds = train_test_ind[break_ind:]

    train_data = all_data.iloc[train_inds]
    test_data = all_data.iloc[test_inds]

    # separate x and y
    y_train = train_data['time_to_ans']
    y_test = test_data['time_to_ans']

    # handle data prep if logit
    if logit:
        y_train = 1 * (y_train < 86400)
        y_test = 1 * (y_test < 86400)

    # drop other columns if needed

    train_data = train_data.drop(['time_to_ans', 'user_id', 'ans_id', 'ques_id'], axis=1)
    test_data = test_data.drop(['time_to_ans', 'user_id', 'ans_id', 'ques_id'], axis=1)

    # drop additional data as a test
    if more_drop:
        train_data = train_data.drop(['views','downvotes','age','reputation','upvotes'], axis=1)
        test_data = test_data.drop(['views','downvotes','age','reputation','upvotes'], axis=1)

    # good old feature normalization - zero mean and unit variance
    means = train_data.mean()
    std = train_data.std()
    train_data = (train_data - means) / std
    test_data = (test_data - means) / std

    # min-max scaling for actual values to keep gamma format
    # max_val = max(y_train)
    # min_val = min(y_train)
    # y_train = y_train / max_val
    # y_test = y_test / max_val


    # return a dictionary with N, N_test, K, x, y, x_test, y_test, columns as numpy arrays (pd.values)
    N = train_data.shape[0]
    N_test = test_data.shape[0]
    K = train_data.shape[1]
    x = train_data.values
    y = y_train.values
    x_test = test_data.values
    y_test = y_test.values
    beta_labels = train_data.columns

    to_return = ({'N': N, 'N_test': N_test, 'K': K, 'x': x, 'y': y, 'x_test': x_test, 'y_test': y_test}, beta_labels)

    return to_return


def basic_linear(data):
    '''
    Use as baseline for stan implementation
    :return:
    '''

    # Model for matrix linear regression with D = {X, Y}, X is an n*k matrix, y is n*1 values
    # explicit priors are specified in the model section, otherwise a uniform prior is assumed
    # use generated quantities to predict N new values:
    stan_data = data[0]

    stan_model = """
    data {
        int<lower=0> N;
        int<lower=0> N_test;
        int<lower=0> K;
        matrix[N,K] x;
        vector[N] y;

        matrix[N_test,K] x_test;
    }

    parameters {
        real alpha;
        vector[K] beta;
        real <lower=0> sigma;
    }

    model {
        y ~ normal(alpha + x * beta, sigma);
    }
    generated quantities {
        vector[N_test] y_test;
        for (n in 1:N_test)
            y_test[n] <- normal_rng(x_test[n] * beta + alpha, sigma);   // NOTE: generated quantities do NOT vectorize
    }

    """

    fit = pystan.stan(model_code=stan_model, data=stan_data, iter=1000, chains=4, thin=1)

    print 'Model:'
    print fit

    # fit.traceplot()
    # py.show()
    eval_acc(fit, data)

    # figure out what else we can actually show here

def gamma_glm(data):
    '''
    Gamma GLM for use in bayesian class project
    input: data - dictionary of data for stan model
    :return:
    '''

    # if you want to make this better/run more expts, dynamically generate some of this model.
    # presently, assume flat priors for all parameters
    stan_data = data[0]

    stan_model = """
    data {
        int<lower=0> N;              // number of samples
        int<lower=0> N_test;
        int<lower=0> K;              // length of each X vector
        matrix[N,K] x;
        vector[N] y;

        matrix[N_test,K] x_test;    // samples for testing
    }

    parameters {
        real<lower = 0> alpha;
        real beta_0;
        vector[K] beta;
        real<lower=.001> phi;
    }

    model {
        alpha ~ normal(1,0.25);     // space similar to exponential distribution
        for (k in 1:K)
            beta[k] ~ normal(0,1);
        beta_0 ~ normal(0,1);
        phi ~ normal(0,1);
        y ~ gamma(alpha, (exp(x * beta + beta_0)) + phi);       // log-link
    }

    generated quantities {                              // predictions!
        vector[N_test] y_test;
        for (n in 1:N_test)
            y_test[n] <- gamma_rng(alpha, (exp(x_test[n] * beta + beta_0)) + phi);
    }
    """

    # fit model
    fit = pystan.stan(model_code=stan_model, data=stan_data, iter=2000, chains=4, thin=1)

    print "Model:"

    print fit

    eval_acc(fit, data)
    # fit.traceplot()
    # py.show()

def bayes_logit(data):
    stan_data = data[0]

    stan_model = """
    data {
        int<lower=0> N;              // number of samples
        int<lower=0> N_test;
        int<lower=0> K;              // length of each X vector
        matrix[N,K] x;
        int<lower=0, upper=1> y[N];

        matrix[N_test,K] x_test;    // samples for testing
    }
    parameters {
        real alpha;
        vector[K] beta;
    }
    model {
        alpha ~ normal(0,1);
        for (k in 1:K)
            beta[k] ~ normal(1,1);
        for (n in 1:N)
            y[n] ~ bernoulli(Phi(alpha + x[n] * beta));
    }
    generated quantities{
        int<lower=0, upper=1> y_test[N_test];
        for (n in 1:N_test)
            y_test[n] <- bernoulli_rng(Phi(alpha + x_test[n] * beta));
    }

    """

    fit = pystan.stan(model_code=stan_model, data=stan_data, iter=2000, chains=4, thin=1)

    print fit

    pdb.set_trace()

    logit_acc(fit, data)

#does accuracy for probit regression
def logit_acc(fit, data):
    # Tally right/wrong
    names = fit.flatnames
    fit_means = fit.get_posterior_mean()

    # programmatically determine when y_test values start appearing in
    start = 0
    for val in names:
        if 'test' in val:
            break
        else:
            start += 1

    # get a list of expected values for all y
    y_means = np.mean(fit_means[start:-1], axis=1)

    # length check
    if len(y_means) != len(data[0]['y_test']):
        print "lengths are off, you moron"
        pdb.set_trace()
    else:
        pairs = zip(1*y_means > 0.5, data[0]['y_test'])
        errors = [(val[0] == val[1])*1 for val in pairs]
        acc = sum(errors)/float(len(errors))
        print 'accuracy: ', acc

    # print out parameters and values
    num_params = len(data[1])

    start = 0
    for val in names:
        if 'beta' in val:
            break
        else:
            start += 1

    for val in zip(data[1], np.mean(fit_means[start:start + num_params], axis=1)):
        print val

    #make plots of values
    betas = fit.extract('beta')['beta']
    ax = sns.violinplot(data=betas[1000:, :])
    ax.set_xticklabels(data[1],rotation='vertical')
    ax.set_title('Beta values for Academia', fontsize=24)
    py.savefig('academia_vplot.jpg')
    py.show()


    pdb.set_trace()


def eval_acc(fit, data):
    # MSE calculation
    names = fit.flatnames
    fit_means = fit.get_posterior_mean()

    # programmatically determine when y_test values start appearing in
    start = 0
    for val in names:
        if 'test' in val:
            break
        else:
            start += 1

    # get a list of expected values for all y
    y_means = np.mean(fit_means[start:-1], axis=1)

    # length check
    if len(y_means) != len(data[0]['y_test']):
        print "lengths are off, you moron"
        pdb.set_trace()
    else:

        # put errors into a list so that you can do more interesting things with them (visualize, etc)
        errors = [(val[0] - val[1]) for val in zip(y_means, data[0]['y_test'])]
        rmse = np.sqrt(np.sum(np.power(errors, 2)) / len(y_means))

        print "Avg answer time in days: "
        print np.mean(data[0]['y_test'])/(3600*24)

        print "RMSE in days: "
        print rmse / (3600*24)

    # print out parameters and values
    print 'parameters: '
    num_params = len(data[1])

    start = 0
    for val in names:
        if 'beta' in val:
            break
        else:
            start += 1

    for val in zip(data[1], np.mean(fit_means[start:start + num_params], axis=1)):
        print val

    #make plots of values
    betas = fit.extract('beta')['beta']
    ax = sns.violinplot(data=betas[1000:, :])
    ax.set_xticklabels(data[1],rotation='vertical')
    ax.set_title('Beta values for Academia, gamma GLM', fontsize=24)
    py.savefig('academia_glm_vplot.jpg')
    py.show()

    pdb.set_trace()

    #TODO: visualization


if __name__ == '__main__':
    #params = load_data(['./data/datascience.stackexchange.com', './data/academia.stackexchange.com', './data/3dprinting.stackexchange.com'], logit=True)
    params = load_data('./data/datascience.stackexchange.com', logit=False, more_drop=False)
    # do a basic linear regression
    #basic_linear(params)

    # pdb.set_trace()

    # do a less basic gamma generalized linear modelc
    gamma_glm(params)
    #bayes_logit(params)


# noinspection PyByteLiteral
def gamma_tutorial():
    '''
    Basic stan gamma GLM with log link function
    Taken from http://seananderson.ca/2014/04/08/gamma-glms.html
    :return:
    '''
    # define stan data
    N = 100
    x = np.random.uniform(-1, 1, N)
    a = 0.5
    b = 1.2
    y_true = np.exp(a + b * x)
    shape = 10.0

    # Passing a vector for scale generates one sample per value in vector; |y_true| = 100 => |y| = 100
    # note that scale is generated in odd fashion (w/ shape parameter), hence odd shape parameter in stan model
    y = np.random.gamma(shape, scale=y_true/shape)

    # Now we put data into a dictionary so we can give to stan
    this_data = {
        'N': N,
        'x': x,
        'y': y
    }

    # define stan model
    stan_code = """
    data {
      int<lower=0> N;
      vector[N] x;
      vector[N] y;
    }
    parameters {
      real a;
      real b;
      real<lower=0.001,upper=100> shape;
    }
    model {
      a ~ normal(0,100);
      b ~ normal(0,100);
      for(i in 1:N)
        y[i] ~ gamma(shape, (shape / exp(a + b * x[i])));
    }
    """

    # fit model
    fit = pystan.stan(model_code = stan_code, data=this_data, iter=1000, chains=4, thin=3)

    '''
    Note that the following two statements are equivalent:
    y ~ normal(alpha + beta * x, sigma)
    for (n in 1:N):
        y[n] ~ normal(alpha + beta * x[n], sigma)
    ...meaning that stan automatically vectorizes
    '''

    """
     summary(fit)

    Call:
    lm(formula = formula, data = data)

    Residuals:
       Min     1Q Median     3Q    Max
    -0.812 -0.139 -0.102 -0.080 32.970

    Coefficients:
                     Estimate Std. Error t value Pr(>|t|)
    (Intercept)    -0.0098974  0.0130572  -0.758 0.448471
    view_count     -0.0151797  0.0136391  -1.113 0.265766
    score          -0.0223985  0.0171350  -1.307 0.191194
    comment_count   0.0444109  0.0133026   3.339 0.000847 ***
    favorite_count -0.0054002  0.0146576  -0.368 0.712566
    views           0.0018720  0.0044461   0.421 0.673741
    downvotes       0.0047046  0.0035467   1.326 0.184729
    reputation     -0.0008014  0.0031688  -0.253 0.800361
    upvotes         0.0018862  0.0032116   0.587 0.557022
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

    Residual standard error: 0.9991 on 6926 degrees of freedom
    Multiple R-squared:  0.002732,  Adjusted R-squared:  0.00158
    F-statistic: 2.372 on 8 and 6926 DF,  p-value: 0.0151

    """