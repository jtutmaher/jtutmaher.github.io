---

layout: post

title: An Introduction to Statistics for Machine Learning

excerpt_separator:  <!--more-->

comments: true

---

It's amazing how powerful basic statistics can be in the context of machine learning. The best data scientists leverage basic statistical methods to implement clever feature engineering and feature selection. Despite the insistence of AI evangelists, nothing can replace careful data processing and handling. 

<!--more-->

I think that advances in deep learning have given the industry the *perception* that feature engineering is obsolete, but this is far from the truth. For example, you can't implement good data augmentation for a CNN without understanding useful representations of an image dataset. Does a 180 degree rotation make the network more prone to overfitting, or is it a valid data augmentation technique? Well, this rotation might work for images of cats, but could fail for images of numbers. **[1]**

This blog isn't about data augmentation. Rather, it's a review some common statistical methods in machine learning . The sections below summarize these ideas, such as skew and kurtosis, in the context of model evaluation. It also leverages a few great libraries in R, which are always worth reviewing.

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

$$f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{\frac{(x-\mu)^2}{2\sigma^2}},$$

- $$ \mu_0 = \int_{-\infty}^{\infty} f(x) dx,$$ (zeroth-moment, equals 1)
- $$ \mu_1 = \int_{-\infty}^{\infty} x f(x) dx,$$ (first moment, mean)
- $$ \mu_2 = \int_{-\infty}^{\infty} x^2f(x) dx,$$ (second moment, variance)
- $$ \mu_3 = \int_{-\infty}^{\infty} x^3 f(x) dx,$$ (third moment, skew)
- $$ \mu_4 = \int_{-\infty}^{\infty} x^4 f(x) dx,$$ (fourth moment, kurtosis)

It is clear from the plots above, for zero-centered normal distributions, that the first and third moments (mean, skew) will be zero for normal distributions, since they are odd functions integrated against an even (normal) function. If the distributions deviate from normality, these moments will cease to be zero, which will result in a general asymmetry - or skew - about the first moment. Skew values can be positive or negative depending on the distribution. 

Kurtosis, the fourth moment, is slightly more subtle. In a sense, it quantifies the ratio of the distributions's width to height. Normal distributions have a kurtosis of 3, where distributions that may be Gaussian in nature are not considered normal with kurtosis values that deviate from 3. Kurtosis can span a range from 0 to infinity, in theory, although in practice these extreme values are unlikely and should be heavily scrutinized. 

Inspecting the moment plot above provides additional context for the kurtosis value. Ultimately, the functional form, $x^4$, weights the tails of the distribution, with wider distributions having higher kurtosis values than thinner distributions. This is precisely the same effect that the variance, $x^2$, quantifies, which is a more well-known quantification of the width distribution, with the primary difference being that between $(-1, 1)$, $x^2$ tends to be more sensative to probabilities and outside of $(-1,1)$, $x^4$ tends to be more sensative. 

The equations are not standarized. Often, the moments are standardized by dividing each moment by factors of the varince to make them dimensionless.  Moreover, samples of the population - which do not represent the total population - have corrections applied to reflect the uncertainty due to low sample count N. This is discussed more completely in the sections below. **[5]**

#Skew and Kurtosis

Skewness quantifies the asymmetry of a distribution about the mean. A right skew distribution has a longer (positive) tail , and a left-skew distribution has a longer (negative) tail. The figures below illustrate this behavior for a skew-normal distribution. In the standardized case, the skewness $\mu_3$ is scaled by the variance, $\mu_2^{3/2}$, generating a defined skewness: **[6]**

$$ \gamma_1 = \frac{\mu_3}{\mu_2^{3/2}}.$$

![SkewNormal](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/skew_normal.png?raw=true)

![SkewNormalMoments](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/skew_normal_third_moment.png?raw=true)

It isn't surprising that a metric evaluating asymmetry depends on an asymmetric function ($x^3$). In this example, I've used the *dsn* function from R's *sn* pacakge, leveraging a skew-normal distribution. **[7]** A skew-normal distribution is not a normal distribution, but rather a generalization of a normal distribution. It is predominantly parameterized by a skew parameter, alpha, and can be written as the combination of a normal PDF, $\phi(x)$, and a normal CDF, $\Phi(x)$:

$$ f(x) = 2 \phi(x)\Phi(\alpha x).$$

The standardized skewness of the figure above is 1.59, indicating a highly right skewed distribution. The third moment plot is particularly informative, as the area is entirely on the postitive end of the x-axis. In the case of a left skew normal distribution, the skewness would be strongly negative, with the function resembling a mirror image of the third moment plot shown above. 

The kurtosis is illustrated in the plot below, with several kurtosis parameters being depicted. In the standardized case, the kurtosis $\mu_4$ is scaled by the variance, $\mu_2$, generating a defined *excess* kurtosis of:

$$ \gamma_4 = \frac{\mu_4} {\mu_2} - 3,$$ 

where subtracting 3 accounts for the fact that a normal distribution has a kurtosis of 3. In the event kurtosis is 0, the distribution is said to be *mesokurtic*. Thin-tailed distributions with negative excess kurtosis are said to be *platykurtic*, and thick-tailed distributions with positive excess kurtosis are said to be *leptokurtic*. **[8]**  

![Kurtosis](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/kurtosis.png?raw=true)

#The Linear Model






**[1]** [https://snow.dog/blog/data-augmentation-for-small-datasets](https://snow.dog/blog/data-augmentation-for-small-datasets)

**[2]** [http://mathworld.wolfram.com/CentralLimitTheorem.html](http://mathworld.wolfram.com/CentralLimitTheorem.html)

**[3]** [http://www.r-tutor.com/elementary-statistics/numerical-measures/moment](http://www.r-tutor.com/elementary-statistics/numerical-measures/moment)

**[4]** [https://en.wikipedia.org/wiki/Moment_(mathematics)#Central_moments_in_metric_spaces](https://en.wikipedia.org/wiki/Moment_(mathematics)#Central_moments_in_metric_spaces)

**[5]** [http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/SkewStatSignif.pdf](http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/SkewStatSignif.pdf)

**[6]** [http://www.r-tutor.com/elementary-statistics/numerical-measures/skewness](http://www.r-tutor.com/elementary-statistics/numerical-measures/skewness)

**[7]** [https://cran.r-project.org/web/packages/sn/sn.pdf](https://cran.r-project.org/web/packages/sn/sn.pdf)

**[8]** [http://www.r-tutor.com/elementary-statistics/numerical-measures/kurtosis](http://www.r-tutor.com/elementary-statistics/numerical-measures/kurtosis)

