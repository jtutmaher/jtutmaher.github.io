---

layout: post

title: Residual Analysis for Regression Models

category: Machine Learning

excerpt_separator:  <!--more-->

comments: true

---

It's amazing how powerful basic statistics can be in the context of machine learning. The best data scientists leverage basic statistical methods to implement clever feature engineering and feature selection. Despite the insistence of AI evangelists, nothing can replace careful data processing and handling. 

<!--more-->

I think that advances in deep learning have given the industry the *perception* that feature engineering is obsolete, but this is far from the truth. For example, you can't implement good data augmentation for a CNN without understanding useful representations of an image dataset. Does a 180 degree rotation make the network more prone to overfitting, or is it a valid data augmentation technique? Well, this rotation might work for images of cats, but could fail for images of numbers. **[1]**

This article isn't about data augmentation. Rather, it's a review some common statistical methods in machine learning . The sections below summarize these ideas, such as skew and kurtosis, in the context of residual analysis and regression models. It also leverages a few great libraries in R, which are always worth reviewing.

## Normal Distributions

Most people are familiar with normal distributions. They underlie many physical systems, such as human height, weight, and even body temperature. This is most impressively formalized by the Central Limit Theorem, which states that certain systems, with an arbitrary number of independent random variables, approach a normal distribution under addition in the limit of large N - *even if the underlying variables aren't normally distributed themselves*. **[2]** This might seem like an edge case, but it's important to remember that A LOT of useful properties are additive - such as the mean of a set of random variables... 

```R
library(ggplot2)
require(gridExtra)

# CONSTRUCT DF
data <- data.frame(xs = seq(-5, 5, 0.1), ys = dnorm(seq(-5, 5, 0.1), mean = 0, sd = 1, log = FALSE))

# COMPUTE THE MOMENTS
data$first.moment = data$xs*data$ys
data$second.moment = data$xs**2*data$ys
data$third.moment = data$xs**3*data$ys
data$fourth.moment = data$xs**4*data$ys


# GENERATE PLOTS
# normal distribution
p1 <- ggplot() + ggtitle('Normal PDF') +
    geom_line(data = data, aes(x = xs, y = ys), alpha=0.5, lwd=1.0) +
    xlab('Index Variable') +
    ylab('PDF') + 
    theme_classic()

# normal moment plots
p2 <- ggplot() + ggtitle('Normal Distribution - Moments') +
  geom_line(data = data, aes(x = xs, y = first.moment, color = "first.moment"), alpha=0.4) +
  geom_line(data = data, aes(x = xs, y = second.moment, color = "second.moment"), alpha=0.4) +
  geom_line(data = data, aes(x = xs, y = third.moment, color = "third.moment"), alpha=0.4) +
  geom_line(data = data, aes(x = xs, y = fourth.moment, color = "fourth.moment"), alpha=0.4) +
  geom_line(data = data, aes(x = mu.xs, y = mu.ys, color = "mean - 0"), lty='dashed') +
  xlab('Index Variable') +
  ylab('Moment Functions') +
  labs(colour='legend') +
  theme_classic() +
  theme(legend.position='left')
  
grid.arrange(p1, p2, ncol=1)
```



![Normal](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/normal_moments.png?raw=true)

It is also important to understand that normal distributions are parameterized by several factors, including the mean $\mu$ and the standard deviation $\sigma$. A more general extension of these parameters are *moments*, which are defined as:

$$ \mu_n = \int_{-\infty}^{\infty} (x-c)^n f(x) dx,$$

where the moment is being computed relative to a reference point, c. When the reference point is the mean, the moment is referred to as a *central moment*. **[3]** This definition is generally correct for continuous PDFs, although there are exceptions. To define these in terms of the normal distribution plotted above (with $\mu=0$): **[4]**

$$f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{\frac{x^2}{2\sigma^2}},$$

- $$ \mu_0 = \int_{-\infty}^{\infty} f(x) dx,$$ (zeroth-moment, equals 1)
- $$ \mu_1 = \int_{-\infty}^{\infty} x f(x) dx,$$ (first moment, mean)
- $$ \mu_2 = \int_{-\infty}^{\infty} x^2f(x) dx,$$ (second moment, variance)
- $$ \mu_3 = \int_{-\infty}^{\infty} x^3 f(x) dx,$$ (third moment, skew)
- $$ \mu_4 = \int_{-\infty}^{\infty} x^4 f(x) dx,$$ (fourth moment, kurtosis)

It is clear from the plots above, for zero-centered normal distributions, that the first and third moments (mean, skew) will be zero for normal distributions, since they are odd functions integrated against an even (normal) function. If the distributions deviate from normality, these moments will cease to be zero, which will result in a general asymmetry - or skew - about the first moment. Skew values can be positive or negative depending on the distribution. 

Kurtosis, the fourth moment, is slightly more subtle. In a sense, it quantifies the ratio of the distributions's width to height. Normal distributions have a kurtosis of 3, where distributions that may be Gaussian in nature are not considered normal with kurtosis values that deviate from 3. Kurtosis can span a range from 0 to infinity, in theory, although in practice these extreme values are unlikely and should be heavily scrutinized. 

Inspecting the moment plot above provides additional context for the kurtosis value. Ultimately, the functional form, $x^4$, weights the tails of the distribution, with wider distributions having higher kurtosis values than thinner distributions. This is precisely the same effect that the variance, $x^2$, quantifies, which is a more well-known quantification of the width distribution, with the primary difference being that between $(-1, 1)$, $x^2$ tends to be more sensative to probabilities and outside of $(-1,1)$, $x^4$ tends to be more sensative. 

The equations are not standarized. Often, the moments are standardized by dividing each moment by factors of the varince to make them dimensionless.  Moreover, samples of the population - which do not represent the total population - have corrections applied to reflect the uncertainty due to low sample count N. This is discussed more completely in the sections below. **[5]**

## Skew and Kurtosis

Skewness quantifies the asymmetry of a distribution about the mean. A right skew distribution has a longer (positive) tail , and a left-skew distribution has a longer (negative) tail. The figures below illustrate this behavior for a skew-normal distribution. In the standardized case, the skewness $\mu_3$ is scaled by the variance, $\mu_2^{3/2}$, generating a defined skewness: **[6]**

$$ \gamma_1 = \frac{\mu_3}{\mu_2^{3/2}}.$$

![SkewNormal](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/skew_normal.png?raw=true)

![SkewNormalMoments](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/skew_normal_third_moment.png?raw=true)

It isn't surprising that a metric evaluating asymmetry depends on an asymmetric function ($x^3$). In this example, I've used the *dsn* function from R's *sn* pacakge, leveraging a skew-normal distribution. **[7]** A skew-normal distribution is not a normal distribution, but rather a generalization of a normal distribution. It is predominantly parameterized by a skew parameter, alpha, and can be written as the combination of a normal PDF, $\phi(x)$, and a normal CDF, $\Phi(x)$:

$$ f(x) = 2 \phi(x)\Phi(\alpha x).$$

The standardized skewness of the figure above is 1.59, indicating a highly right skewed distribution. The third moment plot is particularly informative, as the area is entirely on the postitive end of the x-axis. In the case of a left skew normal distribution, the skewness would be strongly negative, with the function resembling a mirror image of the third moment plot shown above. 

The kurtosis is illustrated in the plot below, with several kurtosis parameters being depicted. In the standardized case, the kurtosis $\mu_4$ is scaled by the variance, $\mu_2$, generating a defined *excess* kurtosis of:

$$ \gamma_2 = \frac{\mu_4} {\mu_2^2} - 3,$$ 

where subtracting 3 accounts for the fact that a normal distribution has a kurtosis of 3. In the event excess kurtosis is 0, the distribution is said to be *mesokurtic*. Thin-tailed distributions with negative excess kurtosis are said to be *platykurtic*, and thick-tailed distributions with positive excess kurtosis are said to be *leptokurtic*. Each kurtosis behavior is illustrated below. **[8]**  

![Kurtosis](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/kurtosis.png?raw=true)

1. Red curve (leptokurtic): excess kurtosis 0.24
2. Green curve (mesokurtic): excess kurtosis 0.00
3. Blue curve (platykurtic): excess kurtosis -0.41

As mentioned before, the excess kurtosis is additional kurtosis above 3. In terms of the kurtosis functions illustrated above, it means that the area under the red curve is greater than 3, the area under the green curve is (approximately) equal to 3, and the area under the blue curve is less than 3. Mathematically, the moments were computed by normalizing the moment definitions above:

$$ \mu_4 = \int x^4 f(x) dx / \int f(x)dx,$$

$$\mu_2^2 = (\int x^2 f(x) dx / \int f(x) dx)^2,$$ 

which is represented in code as:

```R
# FUNCTION ADDING KURTOSIS TO DNORM
fs = function(x,epsilon,delta) dnorm(sinh(delta*asinh(x)-epsilon))*delta*cosh(delta*asinh(x)-epsilon)/sqrt(1+x^2)

# CONSTRUCT TEST PDFS WITH KURTOSIS
data3 <- data.frame(xs = seq(-5, 5, 0.01), ys.1 = dnorm(seq(-5, 5, 0.01), 0, 1))
data3$ys.1 = data3$ys.1
data3$ys.2 = fs(data3$xs, 0, 0.9)
data3$ys.3 = fs(data3$xs, 0, 1.3)

# DEFINE SECOND MOMENT (VARIANCE)
data3$second.moment.1 <- data3$xs**2*data3$ys.1
data3$second.moment.2 <- data3$xs**2*data3$ys.2
data3$second.moment.3 <- data3$xs**2*data3$ys.3

# DEFINE FOURTH MOMENT (KURTOSIS)
data3$fourth.moment.1 <- data3$xs**4*data3$ys.1
data3$fourth.moment.2 <- data3$xs**4*data3$ys.2
data3$fourth.moment.3 <- data3$xs**4*data3$ys.3

# FIRST STANDARDIZED KURTOSIS
std1_norm <- sum(data3$second.moment.1) / sum(data3$ys.1)
kurt1_norm <- sum(data3$fourth.moment.1) / sum(data3$ys.1)
mom1 <-  kurt1_norm / std1_norm**2

# SECOND STANDARDIZED KURTOSIS
std2_norm <- sum(data3$second.moment.2) / sum(data3$ys.2)
kurt2_norm <- sum(data3$fourth.moment.2) / sum(data3$ys.2)
mom2 <-  kurt2_norm / std2_norm**2

# THIRD STANDARDIZED KURTOSIS
std3_norm <- sum(data3$second.moment.3) / sum(data3$ys.3)
kurt3_norm <- sum(data3$fourth.moment.3) / sum(data3$ys.3)
mom3 <-  kurt3_norm / std3_norm**2

# EXCESS KURTOSIS
kurt1 <- mom1 - 3
kurt2 <- mom2 - 3
kurt3 <- mom3 - 3
```

Note, kurtosis was added by augmenting the normal distribution with sinh and cosh functions to increase or decrease the distribution's height or width, as needed. The details of this approach can be obtained here. **[9]**

## The Linear Model

### Introduction

The linear model can be mathematically represented as:

$$ y = \beta_0 + \beta X + \epsilon,$$

where $\beta$ is a vector of coefficients and X is a matrix of independent variables (nsamples x nfeatures). The $\epsilon$ term represents the random, or irreducible, error in the system. Irreducible error is error that, practically speaking, cannot be eliminated - typically due to physical limitations of the data collection system or random fluctuations generated by ancillary, unquantifiable processes.[^bignote] 

[^bignote]: For example, in physics, it is known that force is proportional to mass via the object's acceleration ($F=ma$). However, if one measures an object's force at various accelerations, the resulting output will *not* be perfectly linearly (although it should be very close). This is due to random error, such as fluctuations in air pressure or friction, which are ultimately difficult to control and quantify.

The model's residuals - or the difference between predictions and actuals within the *training set* - should have the following properties, as stated by Robert Hyndman in Chapter 3 of Introduction to Forecasting **[12]**: 

> 1. The residuals are uncorrelated. If there are correlations between the residuals, then there is information left in the residuals which should be used in computing forecasts.
> 2. The residuals have zero mean. If the residuals have a mean other than zero, then the forecasts are biased.

In addition, it is useful (but not nessary) to have the following properties:

> 1. The residuals have constant variance.
> 2. The residuals are normally distributed.

If the residuals are not normally distributed, then standard parametric techniques should not be used to calculate prediction intervals, and nonparametric techniques, such as bootstrapping, should be used. While non-normally distributed residuals are (often) an indicator of poor modeling, there are valid reasons why - in practice - the residuals might be non-normal. For example, random error obtained from pulsed laser data can follow a Poisson distribution since the underlying system is memoryless. **[CITE]**. Moreover, some random errors in partical physic follow a Cauchy distribution. **[CITE]** Therefore, non-normal residual distributions should be investigated, but not necessarily deemed "incorrect". *It is also important to note that none of the linear model assumptions above guarantee normally distribution residuals.*

### An Example

The statistical techniques described above are most explained via an example. Let's say a prediction about the spread of an infectious disease is being made. In this hypothetical scenario, the prediction will be used to plan the amount of medical supplies and resources needed to combat the disease. The true underlying model is exponential in time

$$ N_{theoretical} = e^{\alpha t},$$

where $\alpha$ parameterizes the spread of the infection - although the model is unknown in advance. As such, several models are fit to the training data, which has some nosie thought to be an artifact of the collection process:

$$ N_{actual} = e^{\alpha t} + \epsilon.$$

The noise, $\epsilon$, is normally distributed with zero mean and constant variance. The data, and model fits, are illustrated in the plot below:

![ModelFits](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/model_fits.png?raw=true)

1. **Fit 1**: A linear fit on the entire data set.
2. **Fit 2**: A linear fit on the entire data set with an exponetially transformed input variable.

A quick check of the training RMSE values indicates that fit 2 performs fit 1 with an RMSE of 20.5 infectious cases. However, there are two important reasons why additional analysis should be done:

1. While the RMSE for fit 2 outperforms fit 1, is it actually optimal given the model's assumptions stated above, or can additional information be parsed out and possibly better transformations be applied to the model?
2. What is the correct prediction interval? Is it valid to compute the prediction interval using a standard t-distribution parameterized by the residuals, or should a more sophisticated bootstraping technique be pursued?

With respect to the second point, often a single (point) forecast is insufficient, whereas generating a prediction interval is more desirable.[^bignote2]  However, most prediction intervals leverage parametric techiques, which assume normality. If parametric techniques (rooted in normal distributions) are used, then normality of residuals should be tested. **[CITE]** 

[^bignote2]: For example, the statement: "within 95% confidence the infectious disease will spread to between 450 and 550 individuals at timestep 6.1," is preferable over "the infections disease will spread to 500 individuals at timestep 6.1." The former statement communicates a level of confidence and significance, whereas the latter statement is vague - and ultimately incorrect given the known residual (and subsequently forecast) error.

### Residual Distributions and the Omnibus K2 Test

![ModelFits](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/residual_distributions.png?raw=true)

The residual profiles for fit 1 and fit 2 are illustrated above. Fit 1 has a long left tail, indicating a negative skew, which can be calculated as -0.9. It also has a higher ( *leptokurtic*) kurtosis value of 3.8, indicating a larger width relative to a normal distribution making the fit 2 residuals. Fit 2, on the other hand, has kurtosis and skew values similar to a normal distribution (0 skew and 3 kurtosis). 

Rather than compute both skew and kurtosis for the residual distribution manually, and then compare them manually, a single number can be used to quantify these values and assess normality. This number is a composite of the D'Agostino skewness transformation and the Anscombe & Glynn kurtosis transformation, known as an omnibus statistic **[10]**:

$$ Z_{k2} = Z_1^2(\gamma_1) + Z_2^2(\gamma_2).$$

Here, additional transformations are applied to both kurtosis and skew to speed up convergence to a normal distribution with respect to N in order to aid the test. On their own, $Z_1$ and $Z_2$ roughly approximate normal distrbutions, and when combined in a manner reflecting a Euclidean norm roughly approximate a chi-squared distribution, with two degrees of freedom. **[11]**

The D'Agostino and Anscombe transformations are somewhat complicated, but can be defined as:

$$Z_1(\gamma_1) = \delta \cdot asinh(\frac{\gamma_1}{\alpha \sqrt{\mu_2}}),$$

$$Z_2(\gamma_2) = \sqrt{\frac{9A}{2}} [1 - \frac{2}{9A} - (\frac{1 - 2/A}{1 + \frac{\gamma_2 - \mu_1}{\sqrt{\mu_2}} \sqrt{2/(A-4)}})^{1/3}],$$

where $\mu_1$ and $\mu_2$ in these equations do *not* stand for the moments of the sample distributions being analyzed, but rather the moments of the transformed distributions for the skew and kurtosis, respectively. D'Agostino notes that the omnibus test above is preferred over the chi-squared or Kolmogorov tests due to their low power properties. **[11]** 

As standard in hypothesis testing, the test statistics can be mapped to a p-value, which is then used to assess normality with respect to a confidence interval. If a 95 percent confidence interval is assumed, then fit 1 rejects normality with Anscombe-Glynn and D'Agostino p-values of $4e^{-6}$ and $2e^{-16}$, respectively. Fit 2 accepts the null hypothesis with p-values of 0.86 and 0.49, and therefore has evidence of normality. 

Since fit 2 has evidence of normality, there is legitimacy in applying a prediction interval based on a parametric t-distribution. In this case, the variance of the residuals is 20.5 cases, and the size of the training set is over 1200 time steps, resulting in a 95% prediction interval of: 

$$ f_{density} = f_{mean} \pm 1.96\sigma_r \sqrt{1 + \frac{1}{N}},$$

$$f_{density} =  f_{mean} \pm 1.96\sigma_r,$$

where the bottom equation approximates the prediction interval in the limit of large N. The 1.96 factor represents the number of standard deviations away from the mean the 95% confidence interval lies. In this case, for timestep 6.1, the final forecast prediction would be:

$$f(6.1) = 445 \pm 40$$ cases,

within a 95% confidence interval.

## Additional Considerations

The example discussed above is actually a time-varying process, and there are plenty of caveauts to applying a prediction inteval parametrized by a normal distribution - as testing normality of the residuals is not a proxy check for autocorrelation or heteroscadiscity. Fit 2 was ultimately successful because the data's trend was adjusted, and thereby stationary, which effectively eliminated trend effects in the residuals. Seasonality and trend adjustments are often required for linear time-series forecasting. If the features are not removed from the data, then the fit will be suboptimal, and the remaining information is typically evident in the residuals themselves. To illustrate this point, the fit 1 is shown below. 

![ModelFits](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/residual_plots_fit1.png?)

The plots were generated using the checkresidual() function in the R forecast package. The three plots above include a residual plot, a residual distribution plot, and an autocorrelation function (ACF) plot. The first two plots are fairly self-explanatory, with the first plot illustrating the residual values at various timesteps in the training data, and the second plot illustrating a generic histogram of the residuals. 

In timeseries forecast, ACF can be defined as:

$$ r_k = \frac{\sum_{t=k+1}^T (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^T (y_t - \bar{y})^2}.$$ **[12]**



This can be expressed in matrix notation by thinking of each term in the time varying signal as an element in a vector. 

$$ \vec{y} = (y_1, y_2, …, y_T), $$

$$\vec{y}' = (\vec{y} - \mu) / \sigma.$$

Where the second expression standardizes the signal by its mean and standard deviation. A matrix (similar to a Pearson correlation matrix) can be constructed by taking the outer product of the standardized signal vector with itself:

$$ \mathbf{R} = \vec{y}'^T \vec{y}'.$$



$$\mathbf{R} =  \begin{pmatrix} y_1'y_1' & y_1'y_2' & … & y_1'y_T' \\ y_2'y_1' & y_2'y_2' & …. \\ y_3'y_1' & y_3'y_2' & y_3'y_3' \\ . \\ y_T'y_1' & y_T'y_2' & ... & y_T'y_T' \end{pmatrix}$$



The $r_k$ coefficient above can be computed be summing the elements *along diagonal cuts* of the matrix, where the lag, k, represents how far off the diagonal to shift each cut.  Similar to Pearson correlation, the normalized ACF values fall between -1 (perfect anti-correlation) and 1 (perfect correlation). In the case of white noise, the off-diagonal elements are zero, and diagonal elements are 1.

It should be noted that the Pearson correlation matrix is typically more complicated than the standardized autocovariance matrix defined above. Specifically, the Pearson correlation matrix in machine learning is often computed by taking the product of a feature vector $n_{features} \times n_{samples}$ with its transpose - making the starting point a matrix - not a vector - although the ending point for both is ultimately a matrix. As such, each element in a Pearson correlation matrix typically represents the inner product of two random variables with N samples, and the Pearson correlation coefficient is just an element in the matrix. Looking at the autocovariance matrix defined above, each element is the standardized autocovariance of a time-varying signal *at a specific point in time*, multiplied by iself at various points in time - so basically two scalars, not two vectors. This makes each element in the standardized autocovariance matrix somewhat simpler, and explains why a diagonal sum is required in the first place to compute the ACF.  

As illustrated in the ACF plot, the coefficients decay with increasing lag. This indicates a trend in the signal should be removed before applying a linear fit - possibly via a polynomial or exponential transform. 

## Conclusion

The residuals in fit1 failed our normality tests. Had a normal prediction interval been applied, the future values would have exceeded the P90 values within a few timesteps. This would have put any activities based on this forecast at risk, such as order or transportation of medical supplies. This is why its important to evaluate and understand models in several different ways. 

**[1]** [https://snow.dog/blog/data-augmentation-for-small-datasets](https://snow.dog/blog/data-augmentation-for-small-datasets)

**[2]** [http://mathworld.wolfram.com/CentralLimitTheorem.html](http://mathworld.wolfram.com/CentralLimitTheorem.html)

**[3]** [http://www.r-tutor.com/elementary-statistics/numerical-measures/moment](http://www.r-tutor.com/elementary-statistics/numerical-measures/moment)

**[4]** [https://en.wikipedia.org/wiki/Moment_(mathematics)#Central_moments_in_metric_spaces](https://en.wikipedia.org/wiki/Moment_(mathematics)#Central_moments_in_metric_spaces)

**[5]** [http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/SkewStatSignif.pdf](http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/SkewStatSignif.pdf)

**[6]** [http://www.r-tutor.com/elementary-statistics/numerical-measures/skewness](http://www.r-tutor.com/elementary-statistics/numerical-measures/skewness)

**[7]** [https://cran.r-project.org/web/packages/sn/sn.pdf](https://cran.r-project.org/web/packages/sn/sn.pdf)

**[8]** [http://www.r-tutor.com/elementary-statistics/numerical-measures/kurtosis](http://www.r-tutor.com/elementary-statistics/numerical-measures/kurtosis)

**[9]** [https://academic.oup.com/biomet/article-abstract/96/4/761/220523](https://academic.oup.com/biomet/article-abstract/96/4/761/220523)

**[10]** [https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test](https://en.wikipedia.org/wiki/D'Agostino's_K-squared_test)

**[11]** [https://web.archive.org/web/20120325140006/http://www.cee.mtu.edu/~vgriffis/CE%205620%20materials/CE5620%20Reading/DAgostino%20et%20al%20-%20normaility%20tests.pdf](https://web.archive.org/web/20120325140006/http://www.cee.mtu.edu/~vgriffis/CE 5620 materials/CE5620 Reading/DAgostino et al - normaility tests.pdf)

**[12]** [https://otexts.com/fpp2/autocorrelation.html](https://otexts.com/fpp2/autocorrelation.html)

