import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 02-03 What is a company worth?
def draw_value_of_future_dollor(label, future_value, IR):
    time_years = np.linspace(0, 10, 21)
    present_value = future_value / (1 + IR) ** time_years
    plt.plot(time_years, present_value, linewidth=2.0, label=label)


# draw_value_of_future_dollor("US GOVT IR 1%", 1, 0.01)
# draw_value_of_future_dollor("Balch Bond IR 5%", 1, 0.05)
# plt.legend(loc='upper right')
# plt.show()

# IR : interest rate. how risky that company. IR = Discount Rate.
# sigma(i~inf) FutureValue / (n ** i) = Future Value / (n-1) = Future Value / Discount Rate
def calculate_intrinsic_value(dividends_per_year, discount_rate):
    return dividends_per_year / discount_rate


# dividends_per_year=2
# discount_rate=0.04
# print "${}".format(calculate_intrinsic_value(dividends_per_year, discount_rate))

# Book value : Total assets minus intangible assets and liabilities.
# Ex. If Factories : $40M, Patents : $15M, Liabilities : $10M. => Book value : 40M - 10M = 30M
def calculate_book_value(total_assets, intangible_assets, liabilities):
    return total_assets - intangible_assets - liabilities


# print calculate_book_value(40+15, 15, 10)

# Market Capitalization = Number of shares * price
def calculate_market_capitalization(number_of_shares, price_per_share):
    return number_of_shares * price_per_share


# print calculate_market_capitalization(1000000, 75)


# Example.
# ex_tangible_assets=100000000
# ex_intangible_assets=10000000
# ex_liabilities=20000000
# ex_dividends_per_year=1000000
# ex_discount_rate=0.05
# ex_number_of_shares=1000000
# ex_price_per_share=75
# ex_book_value=calculate_book_value(ex_tangible_assets+ex_intangible_assets,ex_intangible_assets,ex_liabilities)
# ex_intrinsic_value=calculate_intrinsic_value(ex_dividends_per_year, ex_discount_rate)
# ex_market_cap=calculate_market_capitalization(ex_number_of_shares, ex_price_per_share)
# print "book value : {}".format(ex_book_value)
# print "intrinsic value : {}".format(ex_intrinsic_value)
# print "market capitalization : {}".format(ex_market_cap)

# should be buy. becuase book value > market capitalization.

# 02-04 The Capital Assets Pricing Model(CAPM)
# stock_a_return_on_day_1=0.01
# stock_b_return_on_day_1=-0.02
# stock_a_weight=0.75
# stock_b_weight=-0.25
# portfolio_return_on_day_1=stock_a_return_on_day_1*stock_a_weight + stock_b_return_on_day_1*stock_b_weight
# print portfolio_return_on_day_1

# The market portfolio
# market weight for company a = market capitalization of that company / market capitalization of all company

# The CAPM equation
# return of company i(day t) = beta(company i) * return of market(day t) + alpha(company i, day t)
# CAPM says alpha is random & expectation of alpha(company i, day t) is 0.
# 1. draw scatter chart of daily return of compay i vs daily return of market.
# 2. fit a line
# 2-1. beta : slope of line
# 2-2. alpha : y intercept of line

# passive management : make portfolio following market weight. Following CAPM. alpha is random.
# active management : over weight, under weight. Think predict alpha.


# portfolio_return(day t) = sigma(company i) weight(company i) * (beta(company i) * market_return(day t) + alpha(company i, day t))
#                         = beta(portfolio p) * market_return(day t) + alpha(portfolio p, day t)
#                         = beta(portfolio p) * market_return(day t) + sigma(company i) weight(company i) * alpha(company i, day t) = Active

# Implications of CAPM
# r_p = b_p * r_m + a_p
# E of a_p is 0
# Choose high beta in up markets. Choose low beta in down markets.
# But, Efficient Markets Hypothesis(EMH) says you can't predict the market.
# CAPM says you can't beat market. ??

# Arbitrage Pricing Theory (APT)
# if b_i * r_m break down into b_i_finance * r_finance + b_i_tech * r_tech + ... , we can predict more accurately.
# not use in this class.


# 02-05 How hedge funds use the CAPM
def calculate_capm_stock_return(market_return, stock_alpha, stock_beta):
    if stock_alpha < 0:
        return -(stock_beta * market_return + stock_alpha)
    return stock_beta * market_return + stock_alpha


# stock_a_alpha=0.01
# stock_a_beta=1.0
# stock_b_alpha=-0.01
# stock_b_beta=2.0

# market_return=0
# stock_a_return=calculate_capm_stock_return(market_return, stock_a_alpha, stock_a_beta)
# stock_b_return=calculate_capm_stock_return(market_return, stock_b_alpha, stock_b_beta)
# print "market_return:{}, a:{}, b:{}".format(market_return, stock_a_return, stock_b_return)

# market_return=0.1
# stock_a_return=calculate_capm_stock_return(market_return, stock_a_alpha, stock_a_beta)
# stock_b_return=calculate_capm_stock_return(market_return, stock_b_alpha, stock_b_beta)
# print "market_return:{}, a:{}, b:{}".format(market_return, stock_a_return, stock_b_return)

# Two stock CAPM math
def calculate_capm_portfolio_return(market_return, allocs, alphas, betas):
    return (allocs * betas * market_return + allocs * alphas).sum()


def error(allocs, betas):
    for alloc in allocs:
        if alloc >= 1:
            return 1000000

    err = abs(allocs[0] * betas[0] + allocs[1] * betas[1]) + abs(sum(abs(allocs)) - 1)
    return err


def fit_line(betas, error_func):
    allocs = np.float32([.9, .1])
    result = spo.minimize(error_func, allocs, args=(betas,), method="SLSQP", options={'disp': True, 'maxiter': 1000})
    return result.x


def calculate_capm_weights_from_betas(betas):
    l_fit = fit_line(betas, error)
    print "Fitted line:C0={}, C1={}".format(l_fit[0], l_fit[1])


import scipy.optimize as spo

market_return = 0.9
betas = pd.Series([1.0, 2.0])
alphas = pd.Series([0.01, -0.01])
allocs = pd.Series([0.66, -0.33])  # make sigma weight * beta = 0 => eliminate market effect.
print calculate_capm_portfolio_return(market_return, allocs, alphas, betas)
print calculate_capm_weights_from_betas(betas)
