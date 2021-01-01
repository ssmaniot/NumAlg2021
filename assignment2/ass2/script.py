import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import interpolate

# PART a)

def generate_table(name, results):
	with open("{}_table.tex".format(name), "w") as f:
		f.write(r"\begin{table}" + '\n')
		f.write(r"\centering" + '\n')
		f.write(r"\begin{tabular}{||c|c||c|c||}" + '\n')
		f.write('\t' + r"\hline" + '\n')
		f.write('\t' + r"$e_n^{int}$ & $e_n^{ext}$ & $\min_i P_n(t_i)$ & $\max_i P_n(t_i)$ \\" + '\n')
		f.write('\t' + r"\hline\hline" + '\n')
		mins = np.argmin(results, axis = 0)
		maxs = np.argmax(results, axis = 0)
		for i in range(results.shape[0]):
			f.write('\t')
			for j in range(4):
				if i == mins[j]:
					f.write(r"\cellcolor{blue!25}")
				elif i == maxs[j]:
					f.write(r"\cellcolor{red!25}")
				f.write("{:.4e}".format(results[i][j]))
				if (j < 3):
					f.write(r" & ")
			f.write(r" \\" + '\n')
			f.write('\t' + r"\hline" + '\n')
		f.write(r"\end{tabular}" + '\n')
		f.write(r"\caption{}" + '\n')
		label = "tab:{}".format(name)
		f.write("\\label{{{}}}\n".format(label))
		f.write(r"\end{table}")

fname = 'covid19_hospitalized_italy.csv'
data = pd.read_csv(fname)

# This breaks down the np.polyfit algorithm:
# data['date'] = pd.to_datetime(data['date'], format = '%d-%b-%Y')
# t = data['date'].to_numpy().astype(np.float64)

y = data['EmiliaRomagna'].to_numpy().astype(np.float64)
t = np.arange(-y.shape[0] / 2, y.shape[0] / 2)

N = y.shape[0]
N_e = 6
N_int = N - N_e

P = []
results = np.empty((0, 4), dtype = np.float64)

# Polynomial Interpolation
for n in range(1, 21):
	try:
		int_nodes = np.linspace(0, N_int, n + 1, endpoint = False, dtype = int)
		P.append(np.polyfit(t[int_nodes], y[int_nodes], deg = n))
		est = np.polyval(P[-1], t)
		err_int = np.linalg.norm(est[:N_int] - y[:N_int]) / np.linalg.norm(y[:N_int])
		err_ext = np.linalg.norm(est[N_int:] - y[N_int:]) / np.linalg.norm(y[N_int:])
		results = np.vstack([results, np.array([err_int, err_ext, min(est), max(est)])])
	except Exception as e:
		print("[ERROR ERROR ERROR]")
		print(e)

generate_table("polyfit", results)

plt.scatter(t, y, c='C0')
p1 = plt.plot(t, np.polyval(P[np.argmin(results[:, 0])], t), c = 'red')
p2 = plt.plot(t, np.polyval(P[np.argmin(results[:, 1])], t), c = 'green')
plt.title("Polynomial interpolation")
plt.xlabel('t')
plt.ylabel('# of hospitalized people')
plt.legend((p1[0], p2[0]), (r"$P_{}(t)$".format(np.argmin(results[:, 0]) + 1), r"$P_{}(t)$".format(np.argmin(results[:, 1]) + 1)))
plt.savefig("polyfit.pdf")
plt.clf()

results = np.empty((0, 4), dtype = np.float64)
csi = []

# Cubic Spline Interpolation
for n in range(3, 21):
	int_nodes = np.linspace(0, N_int, n + 1, endpoint = False, dtype = int)
	csi.append(interpolate.splrep(t[int_nodes], y[int_nodes]))
	est = interpolate.splev(t, csi[-1])
	err_int = np.linalg.norm(est[:N_int] - y[:N_int]) / np.linalg.norm(y[:N_int])
	err_ext = np.linalg.norm(est[N_int:] - y[N_int:]) / np.linalg.norm(y[N_int:])
	results = np.vstack([results, np.array([err_int, err_ext, min(est), max(est)])])

generate_table("interp", results)

plt.scatter(t, y, c='C0')
i = np.argmin(results[:, 0]) 
p1 = plt.plot(t, interpolate.splev(t, csi[i]), c = 'red')
i = np.argmin(results[:, 1])
p2 = plt.plot(t, interpolate.splev(t, csi[i]), c = 'green')
plt.title("Cubic spline interpolation")
plt.xlabel('t')
plt.ylabel('# of hospitalized people')
plt.legend((p1[0], p2[0]), (r"$s_{" + "{}".format(np.argmin(results[:, 0]) + 3) + "}(t)$", 
	r"$s_{" + "{}".format(np.argmin(results[:, 1]) + 3) + r"}(t)$"))
plt.savefig("interp.pdf")
plt.clf()

# PART b)

# Least Squares Approximation

P = np.polyfit(t[:N_int], np.log(y)[:N_int], 1)
alpha = P[0]

est = np.log(y[0]) + alpha * (t - t[0])
err = np.sqrt((est - np.log(y)) ** 2) / np.sqrt(np.log(y) ** 2)
err_int = err[:N_int]
err_ext = err[N_int:]

p1 = plt.scatter(t[:N_int] - t[0], err_int, c='C0')
p2 = plt.scatter(t[N_int:] - t[0], err_ext, c='C1')
plt.title(r"Relative Error on H(t)$")
plt.xlabel('t')
plt.ylabel('Relative error')
plt.legend([r"$e_i^{int}$", r"$e_i^{ext}$"])
plt.savefig("lstsq_err.pdf")
plt.clf()

alpha = np.empty(N_int - 6, dtype = np.float64)
for i in range(N_int - 6):
	P = np.polyfit(t[i:i+6], np.log(y)[i:i+6], 1)
	alpha[i] = P[0]
p1 = plt.scatter(t[:N_int-6] - t[0], alpha)
plt.title(r"Value of $\alpha$ for 6-day windows")
plt.xlabel(r"$t$")
plt.ylabel(r"$\alpha$")
plt.savefig("lstsq_window_alpha.pdf")
plt.clf()

est = np.log(y[N_int-1]) + alpha[-1] * (t - t[N_int-1])
err = np.sqrt((est - np.log(y)) ** 2) / np.sqrt(np.log(y) ** 2)
err_int = err[:N_int]
err_ext = err[N_int:]

p1 = plt.scatter(t[:N_int] - t[0], err_int, c='C0')
p2 = plt.scatter(t[N_int:] - t[0], err_ext, c='C1')
plt.title(r"Relative Error on $H(t)$ with last window estimate")
plt.xlabel('t')
plt.ylabel('Relative error')
plt.legend([r"$e_i^{int}$", r"$e_i^{ext}$"])
plt.savefig("lstsq_window_err.pdf")
plt.clf()

H_no_lockdown = np.exp(np.log(y[0]) + alpha[0] * (t[-1] - t[0]))
print(t[0] - t[0], t[-1] - t[0], alpha[0])
print("The number of hospitalized individuals if alpha would have remained unchanged at the 15th of April would be: {}".format(H_no_lockdown))
print(y[0])
print(alpha[0]*(t[-1]-t[0]))
print(y[0] * np.exp(alpha[0]*(t[-1]-t[0])))