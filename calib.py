import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import time

from utils.ReadSC import ReadSC2
from utils.MatchIOFiles import matching_io

capi_path = "/home/jayantdubey/Desktop/CAPIdata"
sc_path = "/home/jayantdubey/Desktop/SCdata"
input_output_pairs = matching_io(sc_path,capi_path)  # List of matching input/output file paths

input = f"/home/jayantdubey/Desktop/SCdata/Calibration/P6_.sc"
SCreader = ReadSC2(input) 

'''t = (SCreader.tstamp2 - SCreader.tstamp2[0]) / 1e6  # Convert to seconds


rounded = [np.ceil(entry*100)/100 for entry in vi]
#list = [round(entry,2) for entry in vi if np.isclose(round(entry,2),entry,rtol=0.1,atol=0.1)]

rounded.sort()
unique_vals, counts = np.unique(rounded, return_counts=True)
unique_vals = unique_vals[counts > 15]
counts = counts[counts > 15]


plt.figure()

#plt.plot(np.linspace(0,len(list),len(list)),list)
plt.plot(t, rounded)

plt.figure()
plt.bar(unique_vals, counts, width=0.01)
plt.show()

target, input = input_output_pairs[2]
sc_file = input
'''
'''
'''
'''
for i in range(1,12):
    if i == 2:
        continue
    
    input = f"/home/jayantdubey/Desktop/SCdata/Calibration/P{i}.sc"
    

    if i == 6:
        input = f"/home/jayantdubey/Desktop/SCdata/Calibration/P{i}_2.sc"

    SCreader = ReadSC2(input) 
    vi = SCreader.Vi[:, i-1]
    if i == 1:
        vi = SCreader.Vi[:, i]
    t = t = (SCreader.tstamp2 - SCreader.tstamp2[0]) / 1e6  # Convert to seconds
    plt.figure()
    plt.plot(t, vi)  # raw voltage
    print(input)
    print(f"{vi.min()}, {vi.max()}")
    
    rounded = [np.ceil(entry*100)/100 for entry in vi]
    #list = [round(entry,2) for entry in vi if np.isclose(round(entry,2),entry,rtol=0.1,atol=0.1)]

    rounded.sort()
    unique_vals, counts = np.unique(rounded, return_counts=True)
    unique_vals = unique_vals[counts > 15]
    counts = counts[counts > 15]


    plt.figure()

    #plt.plot(np.linspace(0,len(list),len(list)),list)
    plt.plot(t, rounded)

    plt.figure()
    plt.bar(unique_vals, counts, width=0.01)
    plt.show()

    #SCreader.plotData()
'''
y = np.array([0.01, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 1])
p1 = np.array([1e-5, 0.16, 1.56, 1.78, 2.50, 3.06, 3.78, 4.1, 4.24, 4.51, 4.56, 4.68, 4.76, 4.82])
p3 =  np.array([1e-5, 0.99, 1.21, 1.77, 2.06, 3.22, 3.56, 3.92, 4.14, 4.35, 4.46, 4.56, 4.62, 4.87])
p4 =  np.array([1e-5, 0.47, 1.13, 1.455, 2.40, 3.24, 3.41, 3.79, 4.1, 4.21, 4.52, 4.64, 4.67, 4.83])
p5 =  np.array([1e-5, 1.47, 2.62, 3.41, 3.54, 3.99, 4.32, 4.47, 4.54, 4.67, 4.71, 4.76, 4.79, 4.85])
#o6 = -1.8
p6 = np.array([2.39,2.60,2.90,3.56,3.62, 3.81, 3.90, 4.36, 4.51, 4.58, 4.73, 4.79, 4.82, 4.90])
#o8 = -0.8
p7 = np.array([0.94, 1.71, 2.69, 3.06, 3.34, 3.65, 3.96, 4.10, 4.33, 4.55, 4.58, 4.64, 4.70, 4.79])
p8 = np.array([1.26, 1.65, 2.1, 2.24, 2.31, 2.78, 3.46, 3.66, 3.79, 4.22, 4.31, 4.38, 4.54, 4.71])
p9 =  np.array([1e-5, 1.85, 2.18, 2.53, 2.96, 3.57, 3.77, 3.92, 4.03, 4.30, 4.40, 4.52, 4.59, 4.84])
p10 =  np.array([1e-5, 1.21, 1.61, 1.75, 2.62, 2.91, 3.49, 3.64, 4.06, 4.39, 4.50, 4.64, 4.69, 4.77])
p11 = np.array([0.04, 1.5, 2.54, 2.68, 3.12, 3.52, 4.00, 4.12, 4.29, 4.46, 4.55, 4.62, 4.69, 4.79])

sensors = np.array([p1, p3, p4, p5, p6, p7, p8, p9, p10, p11])

N = 100
C_N = int(N * 1)
'''
# For model
A = np.linspace(799,802,num=N)
B = np.linspace(0.001,0.02,num=N)
C = np.linspace(0.2,0.5,num=N)'''

'''
# For model4
A = np.linspace(-20,100,num=N)
B = np.linspace(-20,100,num=N)
C = np.linspace(-5,5,num=C_N)
'''

# For model3

A = np.linspace(0,5,num=N)
B = np.linspace(1e-6,5,num=N)
C = np.linspace(0,10,num=N)


'''
# For model 5, y = y*1000 and using voltage instead of current
A = np.linspace(25,100,num=N)
B = np.linspace(-150,0,num=N)
C = np.linspace(0,50,num=C_N)
'''
def convert(p):
    if p.any() == 0:
        p = 1e-5
    Rs = 10000*((5.0-p)/p)
    Cs = (1/Rs)
    return Cs

def model(a, b, c, x):
    x = convert(x)
    y = ((a*x)-b)*c
    return y


def model2(a, b, c, x):
    x = convert(x)
    y = (a*np.exp(-b*x)) + c
    return y

def model3(a, b, c, x):
    x = convert(x)
    y = (a*np.log(b*x)) + c
    return y


def model4(a, b, c, x):
    x = convert(x)
    y = a*b*(x) + c
    return y

def model5(a, b, c, x):
    x = convert(x)
    y = (a*x**2) + b*x + c
    return y




def SSE(y, y_hat):
    r = np.sum((y - y_hat)**2)
    #r = r/len(y)
    return np.sqrt(r)

def parameter_search(sensors):

    # A is row B is column. Results store the SSE of the predicted value and 
    result = np.empty((len(A), len(B), len(C), len(sensors)))

    for idx, sensor in enumerate(sensors):
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                for k, c in enumerate(C):
                    
                    y_hat = np.array([model3(a, b, c, x) for x in sensor])   # builds vector of predictions
                
                    result[i,j,k, idx] = SSE(y, y_hat) # vector operation
    
    coeffs = np.zeros((4,result.shape[3]))  # a, b, c, error
    for i in range(result.shape[3]):
        minarg = np.argmin(result[:, :, :, i])
        mrow, mcol, mdepth = np.unravel_index(minarg, result[:, :, :, i].shape)
        coeffs[:, i] = [A[mrow], B[mcol], C[mdepth], np.min(result[:, :, :, i])]

        print(f" p{i+1} | a, b, c: {A[mrow]}, {B[mcol]}, {C[mdepth]}")
        print(f"min error: {np.min(result[:, :, :, i])}")
    
    return coeffs

def poly_search(sensors):
    
    result = np.empty((4,len(sensors)))

    for idx, sensor in enumerate(sensors):
        c = np.polyfit(convert(sensor), y, deg=3)
        result[:, idx] = c
    
    return result

def main():

    #result = parameter_search(sensors)
    result = poly_search(sensors)

    t = (SCreader.tstamp2 - SCreader.tstamp2[0]) / 1e6  # Convert to seconds

    idx = 5

    print(result[:,idx])
    poly = np.poly1d(result[:,idx])

    wght = SCreader.Wght[:, idx]
    p = convert(SCreader.Vi[:,idx])
    plt.figure()

    fig, ax=plt.subplots(figsize=(10,7))
    [ax.axhline(y=i, linestyle='--') for i in [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]]
    ax.plot(t, poly(p), label='Y_hat')
    ax.plot(t, wght, label='Wght')
    plt.legend()
    plt.show()
    


main()
exit(-1)

idx = 2

plt.figure()
a, b, c =  802.0, 0.001, 0.5
y_hat = model(a,b,c,SCreader.Vi[:,idx])
y = SCreader.Wght[:, idx]

y_both = (y + y_hat) / 2.0

t = (SCreader.tstamp2 - SCreader.tstamp2[0]) / 1e6  # Convert to seconds


plt.figure()

fig, ax=plt.subplots(figsize=(10,7))
[ax.axhline(y=i, linestyle='--') for i in [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]]
ax.plot(t, y_hat, label='Y_hat')
ax.plot(t, y, label='Wght')
ax.plot(t, y_both, label='both')


pconv = 220*3.64
offset = [0.002, 0, 0, 0, 0.012, 0, 0.002, 0, 0.007, 0.16]
Wrel = 355.0/np.array([640, 490, 400, 450, 600, 500, 500, 500, 540, 540, 550])

p = convert(SCreader.Vi[:, idx])
bruno_eq = pconv*p
bruno_eq = bruno_eq - offset[idx]
bruno_eq = 1.0*(bruno_eq*Wrel[idx])


ax.plot(t, bruno_eq, label='bruno')
plt.legend()
plt.show()

plt.figure()
plt.plot(t, p)
plt.show()


m = np.array([[6.633165829145728, 55.95477386934674, 0.0],
    [6.633165829145728, 55.95477386934674, 0.0],
    [14.698492462311556, 27.110552763819097, 0.0],
    [11.608040201005025, 25.0, 0.0],
    [68.04020100502512, 84.321608040201, -0.7894736842105265],
    [12.562814070351756, 48.743718592964825, -0.2631578947368425],
    [35.47738693467336, 68.04020100502512, -0.2631578947368425],
    [6.633165829145728, 60.0, 0.0],
    [14.547738693467336, 31.683417085427138, 0.0],
    [16.180904522613062, 39.09547738693467, -0.2631578947368425]])



'''
Model results:  0.1 avg error (100g)
Model 4

P1: min/max: 0.04266541888567182, 5.318383884495096
a, b, c: 6.633165829145728, 55.95477386934674, 0.0

P3: min/max: 0.14340904345172115, 5.318383884495096
a, b, c: 5.954773869346733, 55.60301507537689, 0.0

P4: min/max: 0.08135927246484194, 5.318383884495096
a, b, c: 14.698492462311556, 27.110552763819097, 0.0

P5: min/max: 0.019176464357316608, 5.318383884495096
a, b, c: 11.608040201005025, 25.0, 0.0

P6: min/max: 0.11847955293043287, 6.599236750423125
a, b, c: 68.04020100502512, 84.321608040201, -0.7894736842105265

P7: min/max: 0.15067009083057215, 13.973314259858125
a, b, c: 12.562814070351756, 48.743718592964825, -0.2631578947368425

P8: min/max: 0.0563070863062824, 7.143729443936921
a, b, c: 35.47738693467336, 68.04020100502512, -0.2631578947368425

P9: 0.11945560440466343, 5.318383884495096
a, b, c: 6.633165829145728, 60.0, 0.0

P10: min/max: 0.039768595510194266, 5.318383884495096
a, b, c: 14.547738693467336, 31.683417085427138, 0.0

P11: min/max: 0.1611426093788122, 13.57092229619579
a, b, c: 16.180904522613062, 39.09547738693467, -0.2631578947368425

'''

'''
Model results: 0.092 avg error (83g)
Model 3

P1: min/max: min/max: 0.09068349258627117, 107.9370796657871
a, b, c: 0.25, 1.28, 2.18

P3: min/max: 0.06522799644890145, 108.42392225456689
a, b, c: 0.2763819095477387, 4.422110668341708, 2.051282051282051

P4: min/max: 0.06910149458433638, 108.55372091581606
a, b, c: 0.2763819095477387, 1.1306540402010048, 2.4358974358974357

P5: min/max: 0.12331534139074457, 105.9288715934665
a, b, c: 0.2512562814070352, 3.969849452261306, 1.7948717948717947

P6: min/max: 0.1368674342384031, 112.41341066963133
a, b, c: 0.9045226130653267, 3.994975075376884, 6.842105263157895

P7: min/max: 0.0983595229769639, 107.11176337477745
a, b, c: 0.30150753768844224, 4.824120638190954, 2.1052631578947367

P8: min/max: 0.09603975342757509, 111.5909118467728
a, b, c: 0.577889447236181, 0.3517597236180904, 5.789473684210526

P9: 0.06994507076200879, 108.0813096639921
a, b, c: 0.30150753768844224, 1.2814077788944722, 2.564102564102564

P10: min/max: 0.07987981916091144, 108.47576141861572
a, b, c: 0.2763819095477387, 2.8140707889447234, 2.1794871794871793

P11: min/max: 0.09961582870554138, 107.49453208042078
a, b, c: 0.2763819095477387, 0.45226221608040196, 2.631578947368421
'''

'''
slide_win = 50
slide_stride = 50
total_time_len = sc_data.shape[0]


range_step = range(slide_win, total_time_len, slide_stride)
print(len(range_step))

coeffs = []
points = []


So far the code below takes the windowed time of sc_data (the voltage) of a single channel, computes the fourier transform of said window, 
and stores the coefficients as an element in a list. Then, I take a window like window 0, or window 2, and sort the coeffecients (I lose info of which coeff
corresponds to which sinusoid). Then take the magnitudes and use the top k values. 

What I want to do, is actually either find the variance of the signal by just using the variance equation of a 1D signal and compare that window by window,
or, 
'''

for i in range_step:
    window = sc_data[i-slide_win:i,0]   #  [Nd_input, window:i]
    coeffs.append(fft(window))  # Get fourier coefficients across the window 
    points.append([i-slide_win])

k = 10
coeff_win0 = np.sort(coeffs[0])
coeff_win2 = np.sort(coeffs[2])

if k < slide_win:
    coeff_mags0 = abs(coeff_win0[slide_win-k:])
    coeff_mags2 = abs(coeff_win2[slide_win-k:])

else:
    print("k is bigger than the slide window")

print(coeff_mags0)
print(coeff_mags2)



# Plot the pressure (voltage) data
plt.figure(1)
t = (SCreader.tstamp2 - SCreader.tstamp2[0]) / 1e6  # Convert to seconds
plt.plot(t, SCreader.Vi)  # raw voltage
plt.title(SCreader.SC_data)
plt.gcf().set_size_inches(12, 4)
plt.xlabel("Time (s)")
plt.ylabel("Voltage")
points = np.array(points)
points = points.flatten()

print(points.shape)

for i in range(points.shape[0]):
    plt.axvline(x=points[i], color='red', linestyle='--', linewidth=2)
plt.show()

# Get length of time by: num_of_points / (sample/second) = seconds
# This is your duration 
'''
ffts = []
for i in range(sc_data.shape[1]):
    signal = sc_data[:, i]
    sig_fft = fft(signal)
    #ffts.append(sig_fft)

    s = sig_fft
    N = len(s)
    xf = fftfreq(N, 1 / 860)

    plt.figure()
    plt.plot(xf, np.abs(s))
    plt.show()
'''