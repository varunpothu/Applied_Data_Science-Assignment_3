# -*- coding: utf-8 -*-
"""
Created on Thu May 10 03:02:48 2023

@author: varun
"""

import pandas as pds
import numpy as nps
import matplotlib.pyplot as plts
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import seaborn as sns
import cluster_tools as ct
import errors as err


def map_corr(df, size=10):
    
    
    """
    Plots a correlation matrix for a given DataFrame.
    
    Args:
    df: DataFrame for which the correlation matrix will be computed
    size: Size of the plot (default=10)
    """
    
    
    corr = df.corr()
    plts.figure(figsize=(size, size))
    plts.matshow(corr, cmap='coolwarm')
    plts.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plts.yticks(range(len(corr.columns)), corr.columns)
    plts.colorbar()


def plot_clusters(data, labels, centers, xlabel, ylabel, marker_size=10):
    
    
    """
    Plots clusters of points with their centers.

    Args:
    data: DataFrame containing the data points to be plotted
    labels: Cluster labels for each point
    centers: Cluster centers
    xlabel: Label for the x-axis
    ylabel: Label for the y-axis
    marker_size: Size of the marker for the points (default=10)
    """
    
    
    plts.figure(figsize=(6, 6))
    cm = plts.cm.get_cmap('tab10')
    
    # Create a scatter plot for each cluster
    for i in range(len(centers)):
        cluster_data = data[labels == i]
        plts.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], 
                     marker_size, label=f"Cluster {i+1}", cmap=cm)

    # Plot the cluster centers
    plts.scatter(centers[:, 0], centers[:, 1], 45, "k", marker="d", label="Centers")
    
    plts.xlabel(xlabel)
    plts.ylabel(ylabel)
    plts.legend()  # Add the legend
    plts.show()


def exponential(t, n0, g):
    
    
    """
    Calculates exponential function with scale factor n0 and growth rate g.

    Args:
    t: Independent variable (time)
    n0: Scale factor
    g: Growth rate

    Returns:
    f: Value of the exponential function at t
    """
    
    
    t = t - 1990
    f = n0 * nps.exp(g * t)
    return f


def logistic(t, n0, g, t0):
    
    
    """
    Calculates the logistic function with scale factor n0 and growth rate g.

    Args:
    t: Independent variable (time)
    n0: Scale factor
    g: Growth rate
    t0: Turning point

    Returns:
    f: Value of the logistic function at t
    """
    
    
    f = n0 / (1 + nps.exp(-g * (t - t0)))
    return f


def poly(x, a, b, c, d, e):
    
    
    """
    Calculates a polynomial of degree 4.

    Args:
    x: Independent variable
    a, b, c, d, e: Polynomial coefficients

    Returns:
    f: Value of the polynomial at x
    """
    
    
    x = x - 1990
    f = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4
    return f


# Read data
electric = pds.read_csv("Electric power consumption (kWh per capita).csv")
electric1 = electric[["1971", "1981", "1991", "2001", "2011", "2014"]]


# Compute and display the correlation matrix
map_corr(electric1)
plts.show()

# Clustering
df_ex = electric1[["1971", "2014"]]
df_ex = df_ex.dropna()
df_ex = df_ex.reset_index()
df_ex = df_ex.drop("index", axis=1)


# Normalize the data and store the minimum and maximum values
df_norm, df_min, df_max = ct.scaler(df_ex)


# Perform clustering with the optimal number of clusters (7 in this case)
n_clusters = 7
kmeans = cluster.KMeans(n_clusters=n_clusters)
kmeans.fit(df_norm)
labels = kmeans.labels_
centers = kmeans.cluster_centers_


# Plot the clusters
plot_clusters(df_norm, labels, centers, "Electricity (1971)", 
              "Electricity (2014)")

# Reverse scaling of the cluster centers
scaled_centers = ct.backscale(centers, df_min, df_max)

# Plot the clusters with scaled centers
plot_clusters(df_ex, labels, scaled_centers, "Electricity (1971)", 
              "Electricity (2014)")


# Read the GDP data and cleaning
df_gdp = pds.read_csv("GDP per unit of energy use.csv")
df_gdp = df_gdp.set_index('Country Name', drop=True)
df_gdp = df_gdp.loc[:, '1990':'2014']
df_gdp = df_gdp.transpose()
df_gdp = df_gdp.loc[:, 'India']

# Create a data frame with year and GDP columns
df_gdp_years = pds.DataFrame(df_gdp.index).astype(int)
df_gdp_values = pds.DataFrame(df_gdp.values)

# Convert the data to a 1D array
df_gdp_years = nps.ravel(df_gdp_years)
df_gdp_values = nps.ravel(df_gdp_values)

# Fit exponential, logistic, and polynomial models to the GDP data
param_exp, covar_exp = curve_fit(exponential, df_gdp_years, df_gdp_values, 
                                 p0=(1.2e12, 0.03))
param_log, covar_log = curve_fit(logistic, df_gdp_years, df_gdp_values, 
                                 p0=(1.2e12, 0.03, 1990))
param_poly, covar_poly = curve_fit(poly, df_gdp_years, df_gdp_values)


# Plot the actual GDP data and the fitted models
forecast_years = nps.arange(1990, 2030)

# Exponential model
plts.figure()
plts.plot(df_gdp_years, df_gdp_values, label="GDP")
plts.plot(forecast_years, exponential(forecast_years, *param_exp), 
          label="Exponential fit")

plts.xlabel("Year")
plts.ylabel("GDP")
plts.legend()
plts.show()

# Calculate error ranges for logistic
low_log, up_log = err.err_ranges(forecast_years, logistic, param_log, 
                              nps.sqrt(nps.diag(covar_log)))

# Logistic model
plts.figure()
plts.plot(df_gdp_years, df_gdp_values, label="GDP")
plts.plot(forecast_years, logistic(forecast_years, *param_log), 
          label="Logistic fit")

plts.fill_between(forecast_years, low_log, up_log, color="yellow", alpha=0.7)
plts.xlabel("Year")
plts.ylabel("GDP")
plts.legend()
plts.show()

# Polynomial model
plts.figure()
plts.plot(df_gdp_years, df_gdp_values, label="GDP")
plts.plot(forecast_years, poly(forecast_years, *param_poly), 
          label="Polynomial fit")

# Calculate error ranges for polynomial
low_poly, up_poly = err.err_ranges(forecast_years, poly, param_poly, 
                                   nps.diag(covar_poly))

# Add the error fill for the polynomial plot
plts.fill_between(forecast_years, low_poly, up_poly, color="yellow", alpha=0.9)

plts.xlabel("Year")
plts.ylabel("GDP")
plts.legend()
plts.show()
