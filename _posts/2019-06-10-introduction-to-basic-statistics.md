---

layout: post

title: An Introduction to Basic Statistics

excerpt_separator:  <!--more-->

---

It's amazing how powerful basic statistics can be in the context of Machine Learning. The best data scientists leverage basic statistical methods to implement clever feature engineering and feature selection. Despite the AI evangelists insistence, nothing can replace careful data processing and handling. 

<!--more-->

I think that advances in deep learning have given the industry the *perception* that data understanding is obsolete, but this is far from the truth. For example, you can't implement good data augmentation for a CNN without understanding useful image representations of your dataset. Does a 180 degree rotation introduce error to the network, or is it a valid data augmentation technique? Well, this rotation might work well for images of cats, but could fail miserably for images of numbers. **[1]**

This blog isn't about data augmentation. Rather, it's an effort to return to basics and review some of the most common statistical methods. Below are summaries and examples of some of the most basic statistical definitions, such as skew, kurtosis, power, and sampling techniques. It also leverages some excellent libraries in R, which are always worth reviewing.

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

$$ f_{pdf} = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{\frac{(x-\mu)^2}{2\sigma^2}}.$$



**[1]** [https://snow.dog/blog/data-augmentation-for-small-datasets](https://snow.dog/blog/data-augmentation-for-small-datasets)

**[2]** [http://mathworld.wolfram.com/CentralLimitTheorem.html](http://mathworld.wolfram.com/CentralLimitTheorem.html)
