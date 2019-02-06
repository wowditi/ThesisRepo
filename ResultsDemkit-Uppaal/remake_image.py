import matplotlib.pyplot as plt
import pylab
import pickle
import os
import matplotlib.ticker as tkr

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def hourfmt(x, pos):
    s = f'{int(x/96)}'
    return s

def yaxfmt(y, pos):
    s = f'{int(y/3600)}'
    return s
xfmt = tkr.FuncFormatter(hourfmt) 
yfmt = tkr.FuncFormatter(yaxfmt) 

def plot(plot: str, yticks, tsCosts, evCosts,
         batteryCosts, staticCosts, summedTS, summedEV, summedBat, combined):
	plt.rcParams.update({'font.size': 22})
	plt.figure(figsize=(20, 5))
	plt.xlabel('time (days)')
	plt.ylabel(r'energy consumption ($Wh$)')	
	plt.bar(range(len(combined)), combined, color='limegreen', linewidth=8.0, label='Combined')
	plt.plot(range(len(summedTS)), summedTS, linewidth=3.0, label="Timeshiftable Costs", color='crimson')
	plt.plot(range(len(summedEV)), summedEV, linewidth=3.0, label="EV Costs", color='royalblue')
	plt.plot(range(len(summedBat)), summedBat, linewidth=3.0, label="Battery Costs", color='c')
	plt.plot(range(len(staticCosts)), staticCosts, linewidth=3.0, label="staticCosts", color='m')
	pylab.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
	       fancybox=True, shadow=True, ncol=5, prop={'size': 20})
	pylab.gca().xaxis.set_major_formatter(xfmt)
	pylab.gca().yaxis.set_major_formatter(yfmt)
	if yticks is None:
		minVal = int(min(combined+summedTS+summedBat+staticCosts+summedEV)/3600)
		maxVal = int(max(combined+summedTS+summedBat+staticCosts+summedEV)/3600)
		minY = (minVal+(10-minVal%10))*3600
		if minVal%10 < 5:
			minY-=36000
		maxY = (maxVal-maxVal%10)*3600+1
		if maxVal%10 > 5:
			maxY+=36000
		yticks = range(minY, maxY, int(round((maxVal-minVal)/25)*5)*3600)
	plt.yticks(yticks)
	plt.xticks(range(0, len(combined), 96))
	plt.savefig(plot, format='eps', dpi=5000, bbox_inches='tight', pad_inches=0.1)
	plt.close()
	return yticks


if __name__ == "__main__":
	files = os.listdir('.')
	for directory in [f for f in files if f[:6]=='Thesis']:
		print(directory)
		yticks = plot(f'Images/Stratego_{directory[6:]}.eps', None, **load_obj(f'{directory}/Stratego.pickle'))
		plot(f'Images/DEMKit_{directory[6:]}.eps', yticks, **load_obj(f'{directory}/DEMKit.pickle'))
	

