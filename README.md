# Attenuation-Corrected Correlations

A python package to compute the attenuation-corrected correlation coefficient.
This is useful to estimate the 'true' correlation between two classes of data by correcting for the reliability within and between classes - as if one had drawn an infinite amount of samples.

See the [`Introduction`](#introduction) section for detailed conceptual and math-based introduction,
[`Pseudocode`](#pseudocode) for a code-based intro.

## Usage

### Parameters and output

#### Mandatory inputs

To compute the r_ac, the following are mandatory:

| Parameter | Description                                             |
|-----------|---------------------------------------------------------|
| `arr_a`   | First data array of shape (`n_samples_a`, `n_features`) |
| `arr_b`   | First data array of shape (`n_samples_b`, `n_features`) |

Note: `n_features` needs to be the same across groups (repeated measures)

#### Optional inputs

These arguments are not required for the computation to succeed, but are required to fine-tune the computations:

| Parameter               | Description                                       | Default Value                                  |
|-------------------------|---------------------------------------------------|------------------------------------------------|
| `statistical_threshold` | Alpha level to ensure internal group reliability  | 0.05                                           |
| `feature_labels_a`      | Feature labels of arr_a of shape (`n_samples_a`,) | 0 -> `n_samples_a`                             |
| `feature_labels_b`      | Feature labels of arr_b of shape (`n_samples_b`,) | `n_samples_a` -> `n_samples_a` + `n_samples_b` |

Note:

- The statistical threshold is used to check if one can assume that there is some reliability within each dataset
- The feature labels are used to account for repeated measures - if some rows are measured more often than others.

#### Outputs

These outputs are computed:

| Parameter                   | Description                                          |
|-----------------------------|------------------------------------------------------|
| reliability_a               | Internal reliability of `arr_a`                      |
| reliability_b               | Internal reliability of `arr_b`                      |
| reliability_across          | Reliability across data arrays                       |
| corrected_correlation       | Attenuation corrected correlations                   |
| pearson                     | Pearson's correlation coefficient across data arrays |
| significance_level_a        | p-value of the reliability of `arr_a`                |
| significance_level_b        | p-value of the reliability of `arr_b`                |
| pairwise_correlation_a      | Pairwise internal reliabilities of `arr_a`           |
| pairwise_correlation_b      | Pairwise internal reliabilities of `arr_b`           |
| pairwise_correlation_across | Pairwise reliabilities across data arrays            |

### Basic Usage

```python
from accorr import accorr

ac = accorr()
result = ac.compute(data_a, data_b)

print(f'Attenuation-corrected correlation: {result.corrected_correlation}')
print(f'Reliability data a: {result.reliability_a}')
print(f'Reliability class b: {result.reliability_b}')
print(f'Reliability between classes: {result.reliability_across}')
print(f'Pearson`s correlation coefficient: {result.pearson}')
```

### Advanced Usage

With feature labels (e.g. repeated measures defined by `labels_a` and `labels_b`):

```python
from accorr import accorr

ac = accorr()
result = ac.compute(
    data_a,
    data_b,
    statistical_threshold=0.01,
    feature_labels_a=labels_a,
    feature_labels_b=labels_b
    )
```

## Installation

### Dependencies

- numpy
- scipy
- pydantic

### With pip from GitHub

Install the latest Commit on branch `main`

```bash
pip install git+https://github.com/jkschluesener/accorr.git@main
```

### Poetry

If you are using Poetry:

```bash
poetry add git+https://github.com/jkschluesener/accorr.git
```

## Introduction

### Terminology

The terminology used in this README and the code are as follows:

The correlation coefficient is calculated between 2 arrays of data.  
Each class is comprised of one or several `samples` (rows, `array.shape[0]`, e.g. repeated measurements of different subjects) and `features` (columns, `array.shape[1]`, e.g. samples of a confusion matrix)

### Conceptual and Mathematical Introduction

Pearson's correlation coefficient $\rho_{X, Y}$ is a measure of the linear relationship between two variables $X$ and $Y$.  
It is defined as:

$$
\rho_{X, Y} = \frac{ \text{cov}(X, Y) }{ \sigma X \sigma Y }
$$

where:

- $\sigma_X$, $\sigma_Y$ are the standard deviation of $X$ and $Y$,
    respectively.
- $cov$ is the covariance

#### Standard Deviation

The standard deviation $\sigma_X$ is the square root of the variance of $X$:

$$
\sigma_X = \sqrt{\text{var}(X)} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}{(X_i - \mu_x)^2}}
$$

#### Covariance

The covariance $cov$ is the joint variability of 2 sets of datapoints $X$, $Y$ and can be expressed as:

$$
\text{cov}(X, Y) = \frac{1}{n}\sum_{i=1}^{n}(X_i - \mu_x)(Y_i - \mu_x)
$$

where $\mu_x$ and $\mu_y$ are the mean of $X$ and $Y$.

Variance can be seen as a special case of covariance of a variable with itself:

$$
\text{var}(X) = \text{cov}(X, X)
$$

#### Back to the original equation

And thus our initial equation summarized in terms of covariance:

$$
\rho(X, Y) = \frac{ \text{cov}(X, Y) }{ \sqrt{ \text{var}(X)} * \sqrt{\text{var}(Y) } }= \frac{ \text{cov}(X, Y) }{ \sqrt{ \text{cov}(X, X)} * \sqrt{\text{cov}(Y, Y) } } 
$$

or in a human-readable format:

Pearson's correlation coefficient is a measure of linear correlation between two groups of data in the range $(-1, 1)$.  
It is the covariance across variables normalized by the square root of the product of covarainces within each class.

## Reliability in Attenuation Correction

In the context of attenuation correction, reliability is defined as the mean of all pairwise correlation coefficients within and across sets of data. Given two sets of variables $\text{A}$ and $\text{B}$.
$\overline{ \text{rel}(A, A) }$ is the mean reliability within $A$, $\overline{ \text{rel}(B, B) }$ is the mean reliability within $B$, and $\overline{ \text{rel}(A, B) }$ is the mean reliability between $A$ and $B$.

$$
A = { A_1, A_2 \dots }
$$

$$
B = { B_1, B_2 \dots }
$$

Each element $A_1$, $B_1$, $\dots$ is assumed to be a vector of features with identical length, the count of elements in $A$ and $B$ does not need to be identical.

### Reliability within A

$$
\overline{\text{rel}(A, A)} = \frac{2}{n_A(n_A - 1)} \sum_{i=1}^{n_A} \sum_{j\ne i}^{n_A} \rho(A_i, A_i)
$$

### Reliability within B

$$
\overline{\text{rel}(B, B)} = \frac{2}{n_B(n_B - 1)} \sum_{i=1}^{n_B} \sum_{j\ne i}^{n_B} \rho(B_i, B_i)
$$

### Reliability across A and B

$$
\overline{\text{rel}(A, B)} = \frac{1}{n_A n_B} \sum_{i=1}^{n_A} \sum_{j=1}^{n_B} \rho(A_i, B_i)
$$

## Attenuation-Corrected Correlation

Attenuation correction is a method that allows to estimate the relationship between two groups of data as if one had drawn an infinite amount of samples.
While Pearson's correlation coefficient is a useful measure, it can be influenced by measurement error or noise in the data.
This would mean that they are free of noise (as it averages out) and would provide a more accurate measure of the true correlation

In other words, as the number of samples approaches infinity, the Pearson correlation coefficient will approach the attenuation-corrected correlation coefficient:

$$
\lim_{( n_X, n_Y )  \to ( \infty, \infty )} \rho(X, Y) = \rho_{ac}(X,Y)
$$

where:

- $\rho_{ac}$ is the attuation-corrected correlation coefficient
- $n_X$ and $n_Y$ are the sample count of variables $X$ and $Y$, respectively.

The attenuation-corrected correlation can be calculated as:

$$
\rho_{ac}(A, B) = \frac{ \overline{ \text{rel}(A,B) }}{ \sqrt{ \overline{\text{rel}(A,A) } * \overline{ \text{rel}(B,B) } } }
$$

### Pearson's r - Sampling Distribution - Z-Score

If the absolute correlation within a population is low (\< e.g. 0.4),
then of Pearson's r is approximately normally distributed.
With higher values of correlation, the distribution will develop a negative skew.

Using the Fisher Z-Transformation (z-score), this skewed distribution can be transformed to a normal distribution.

### Computing the attenuation-corrected correlation coefficient

Putting the above together, a mathematical workflow to calculate $r_{ac}$:

$$
\text{rel}(A, B) = \text{pairwiseCorrelations}(A, B)
$$

$$
\text{rel}_Z(A, B) = \text{arctanh}(\text{rel}(A, B))
$$

$$
\overline{ \text{rel}_Z(A, B) } = \text{mean}(\text{rel}_Z(A, B))
$$

$$
\overline{ \text{rel}(A, B) } = \text{tanh}( \overline{ \text{rel}_Z(A, B) } )
$$

Or, in a human-readable format:

The attenuation corrected correlation between data $A$ and data $B$ is the reliability across classes normalized by the square root of the product of the reliabilities within classes.

### Pseudocode

#### Data definition

```raw
A <- array of floats of size (o, n)
B <- array of floats of size (p, n)
```

where $o$, $p$ are the numbers of features and $n$ the numbers of samples

#### Reliability between classes

```raw
C <- array of shape (o, p)

for  i, j in 0:len(o), 0:len(p) do
    c = corr(A[i, :], B[j, :])
    c = arctanh(c)
    C[i, j] = c

if one-sided ttest of C[C[not isnan(C)]] is significantly greater than 0, then

    if A, B contain repeated measures then
        rel_ab = repeated_measures_nanmean(C)
    else if all entries in o, p are non-repeated measures then
        rel_ab = nanmean(C)

    rel_ab = tanh(rel_a)

else

    rel_ab = NaN
```

#### Reliabilities within a class

Exemplary for class A

```raw
C <- array of shape (o, o)

for each entry i, j in 0:len(o), 0:len(o) do
    c = corr(A[i, :], A[j, :])
    c = arctanh(c)
    C[i, j] = c

C[diagonal_index(C)] = NaN

if ttest of C[C[not isnan(C)]] is significantly greater than 0 then

    if A, B contain repeated measures (e.g. from the same subject) then
        rel_a = repeated_measures_nanmean(C)
    else if all entries o, p are unique variables then
        rel_a = nanmean(C)

    rel_a = tanh(rel_a)

else

    rel_a = NaN
```

### Computing the attenuation corrected correlation coefficient

```raw
r_ac = arctanh(rel_ab / sqrt(rel_a * rel_b))
```

## Internal statistical testing of reliabilities

Reliabilities need to be significantly different from 0, so a simple ttest compared to 0 needs to be calculated for each reliability term.  
This is only meant as an indication and thus is not a very reliable method.  
Keep in mind that the ttest function assumes all independent measures - should some of your data be repeated measurements, e.g. of the same subject, then the result will be unreliable as the degrees of freedom don't match - you can try averaging them before.
If you set `labels_a` and `labels_b` correctly, averaging will be done automatically.  

## Reliability - Range of values

Reliabilities should not be in the range $(0, 1)$, but it is possible
that by chance values greater than 1 can occur by chance.
However their occurence should be random (not significant with a statistical test) and not systematically occuring.

## Statistical testing of attenuation corrected correlation

Should you need error bars around your attenuation corrected correlation values, e.g. for statistical testing, you could consider resampling your input data by e.g. bootstrapping or jacknifing.

<!-- ## Further Reading

C. Spearman, "'General Intelligence', Objectively Determined and Measured", The American Journal of Psychology, vol. 15, no. 2, p. 201, Apr. 1904, doi: 10.2307/1412107. Available: <https://www.jstor.org/stable/1412107>. -->

## License

This repo carries a MIT license.
