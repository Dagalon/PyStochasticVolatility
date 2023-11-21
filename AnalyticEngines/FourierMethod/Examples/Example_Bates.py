import numpy as np
from AnalyticEngines.FourierMethod.CharesticFunctions import JumpDiffusionCharesticFunction
from AnalyticEngines.FourierMethod.COSMethod import COSRepresentation
from functools import partial
from Tools import Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
import time

# European option price
k_s = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0, 170.0])
no_strikes = len(k_s)
f0 = 100.0
x0 = np.log(f0)
T = 2.0

# Bates para
r = 0.00
v0 = 0.25
k = 1.5
theta = 0.2
epsilon = 0.05
rho = -0.8
lambdaJ = 0.1
muJ = 0.1
sigmaJ = 0.2
b2 = k
u2 = -0.5

# Upper and lower bound for cos integral
a = -6.0
b = 6.0

cf_bates = partial(JumpDiffusionCharesticFunction.get_bates_cf, t=T, r_t=r, x=x0, v=v0, theta=theta, rho=rho, k=k,
                   epsilon=epsilon, jump_mean=muJ, jump_std=sigmaJ, jump_intensity=lambdaJ, b=b2, u=u2)
start_time = time.time()
cos_price = COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 256, k_s, cf_bates)
end_time = time.time()
diff_time = end_time - start_time
print("COS method time ---- " + str(diff_time))

start_time = time.time()
price_cf_integration = []
for i in range(0, no_strikes):
    european_option = EuropeanOption(k_s[i], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)
    price_cf_integration.append(european_option.get_analytic_value(r, theta, rho, k, epsilon, v0, muJ, sigmaJ, lambdaJ,
                                                                   model_type=Types.ANALYTIC_MODEL.BATES_MODEL_LEWIS,
                                                                   compute_greek=False))

end_time = time.time()
diff_time = end_time - start_time
print("Lewis's formula integration method time ---- " + str(diff_time))
