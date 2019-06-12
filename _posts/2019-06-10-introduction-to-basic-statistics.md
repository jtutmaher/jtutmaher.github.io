---

layout: post

title: An Introduction to Statistics for Machine Learning

excerpt_separator:  <!--more-->

comments: true

---

It's amazing how powerful basic statistics can be in the context of Machine Learning. The best data scientists leverage basic statistical methods to implement clever feature engineering and feature selection. Despite the insistence by AI evangelists, nothing can replace careful data processing and handling. 

<!--more-->

I think that advances in deep learning have given the industry the *perception* that data understanding is obsolete, but this is far from the truth. For example, you can't implement good data augmentation for a CNN without understanding useful representations of an image dataset. Does a 180 degree rotation introduce error to the network, or is it a valid data augmentation technique? Well, this rotation might work for images of cats, but could fail for images of numbers. **[1]**

This blog isn't about data augmentation. Rather, it's a review some common statistical methods in Machine Learning that I've found useful. The sections below summarize these ideas, such as skew, kurtosis, power, and cluster sampling. It also leverages a few great libraries in R, which are always worth reviewing.

## Normal Distributions

Most people are familiar with normal distributions. They underline many physical systems, such as human height, weight, and even body temperature. This is most impressively formalized in the Central Limit Theorem, which states that some systems, with an arbitrary number of independent random variables, approach a normal distribution under addition - *even if the underlying variables aren't normally distributed themselves*. **[2]** This might seem like I'm harping on an edge case, but it's important to remember that A LOT of useful properties are additive - like the mean of a series... 

```R
library(ggplot2)
library(data.table)

# CONSTRUCT DF
data <- data.frame(xs = seq(-5, 5, 0.1), ys = dnorm(seq(-5, 5, 0.1), mean = 0, sd = 1, log = FALSE))

# PLOT IT
p <- qplot(x=xs, y=ys, data=data, geom='line',alpha=I(.5), 
   main="Sample Normal Distribution", xlab="Index Variable", 
   ylab="PDF")
p + theme_classic()
```



![Normal](https://raw.githubusercontent.com/jtutmaher/jtutmaher.github.io/master/_screenshots/normal.png?raw=true)

$$f_{pdf} = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{\frac{(x-\mu)^2}{2\sigma^2}}$$

It's important to remember that normal distributions are parameterized by two factors, the mean $\mu$ and the standard deviation $\sigma$. Typically normal distributions have zero skewness and kurtosis; however, certain extensions of normal distributions, such as skew normal distributions, violate this behavior. These distributions are discussed more completely below. 

## Skew and Kurtosis

Skewness quantifies the asymmetry of a distribution. It is often, but not always, the case that a right-skew distribution has a mean greater than the median, and a left-skew distribution has a mean less than the median. This statement can break down for multimodal or discrete distributions. Regardless, it is always the case that a unimodal distribution with zero skew has mean=median=mode. 

An important type of skew distribution is the skew normal distribution. 


**[1]** [https://snow.dog/blog/data-augmentation-for-small-datasets](https://snow.dog/blog/data-augmentation-for-small-datasets)

**[2]** [http://mathworld.wolfram.com/CentralLimitTheorem.html](http://mathworld.wolfram.com/CentralLimitTheorem.html)
