import importlib.util
import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import re
import subprocess
import time
from datetime import datetime, timedelta
from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet
from itertools import islice
from pylab import legend
from pytz import timezone
from shutil import copyfile
from typing import List, Dict, Union, Sequence

from StrategoDeclaration import createHouses, EV, Battery
from StrategoOnline import IntInterval, StrategoAlgorithm, Component, LearningParams, Model, SimPoint


@dataclass
class ALPGConfig:
	alpg: str
	config: str
	configName: str
	outputPath: str
	outputDir: str
	numDays: int
	startDay: int
	numHouses: int


@dataclass
class DEMKitConfig:
	demkit: str
	workspace: str
	folder: str
	plot: str
	plotValues: str


@dataclass
class StrategoConfig:
	verifyta: str
	workingDir: str
	baseModel: str
	workingModel: str
	query: str
	requiredTemplates: List[str]
	components: List[Component]
	result: str
	plot: str
	plotValues: str


class Comparator:
	def __init__(self, alpgConfig: ALPGConfig, demkitConfig: DEMKitConfig, strategoConfig: StrategoConfig,
	             devices: List[str], resultDir: str, force=False):
		self.alpgConf = alpgConfig
		self.demkitConf = demkitConfig
		self.strategoConf = strategoConfig
		self.devices = devices
		self.situation = None
		self.resultDir = resultDir
		self.force = force
		if os.path.exists(self.resultDir) and not self.force:
			raise Exception("Results Dictory already exists!")
		if not os.path.exists(self.resultDir):
			os.makedirs(self.resultDir)

	def createDeclaration(self):
		timeZone = timezone('Europe/Amsterdam')
		dst = bool(datetime.fromtimestamp(
			int(timeZone.localize(datetime(2018, 1, 1) + timedelta(days=self.alpgConf.startDay)).timestamp()),
			timeZone).dst())
		self.situation = createHouses(f'{self.alpgConf.outputPath}/', f'{self.alpgConf.outputPath}/UppaalDeclarion.txt',
		                              self.alpgConf.numDays, self.alpgConf.numDays - 2,
		                              self.alpgConf.startDay * 24 * 4 - int(dst) * 4, 15)

	def writeTimes(self, file: str, startTimes: Dict[str, List[int]], endTimes: Dict[str, List[int]]):
		with open('%s_Starttimes.txt' % file, 'w') as f:
			for house, values in startTimes.items():
				f.write("%s:%s\n" % (house, ",".join([str(v) for v in values])))
		with open('%s_Endtimes.txt' % file, 'w') as f:
			for house, values in endTimes.items():
				f.write("%s:%s\n" % (house, ",".join([str(v) for v in values])))

	def rewriteEVCharge(self, file: str, removed: Dict[str, List[int]]):
		copyfile('%s_RequiredCharge.txt' % file,
		         '/%s_RequiredCharge.bak' % file)
		charges: Dict[str, List[int]] = dict()
		with open('/%s_RequiredCharge.bak' % file) as f:
			for idx, line in enumerate(f.readlines()):
				charges[line.split(':', 1)[0]] = [int(x) for x in (line.split(':', 1)[1].rstrip().split(','))]
		for house, values in removed.items():
			for j in sorted(values, reverse=True):
				del charges[house][j]
		with open('%s_RequiredCharge.txt' % file, 'w') as f:
			for house, values in charges.items():
				f.write("%s:%s\n" % (house, ",".join([str(v) for v in values])))

	def getTimes(self, file: str):
		startTimes: Dict[str, List[int]] = dict()
		endTimes: Dict[str, List[int]] = dict()
		copyfile('%s_Starttimes.txt' % file,
		         '/%s_Starttimes.bak' % file)
		copyfile('/%s_Endtimes.txt' % file,
		         '/%s_Endtimes.bak' % file)
		with open('/%s_Starttimes.bak' % file) as f:
			for idx, line in enumerate(f.readlines()):
				startTimes[line.split(':', 1)[0]] = [int(x) for x in (line.split(':', 1)[1].rstrip().split(','))]
		with open('/%s_Endtimes.bak' % file) as f:
			for idx, line in enumerate(f.readlines()):
				endTimes[line.split(':', 1)[0]] = [int(x) for x in (line.split(':', 1)[1].rstrip().split(','))]
		return startTimes, endTimes

	def removeEdgeCases(self):
		maxTime: int = (self.alpgConf.startDay + self.alpgConf.numDays - 2) * 3600 * 24
		devices = ["WashingMachine", "Dishwasher", "ElectricVehicle"]
		for device in devices:
			startTimes, endTimes = self.getTimes('%s/%s' % (self.alpgConf.outputPath, device))
			removedIdx: Dict[str, List[int]] = dict()
			for house, startArray in startTimes.items():
				for j, start in enumerate(startArray):
					if start < maxTime < endTimes[house][j]:
						startTimes[house].remove(start)
						endTimes[house].remove(endTimes[house][j])
						if house in removedIdx:
							removedIdx[house].append(j)
						else:
							removedIdx[house] = [j]
			self.writeTimes('%s/%s' % (self.alpgConf.outputPath, device), startTimes, endTimes)
			if device == "ElectricVehicle":
				self.rewriteEVCharge('%s/%s' % (self.alpgConf.outputPath, device), removedIdx)

	def createSituation(self):
		if os.path.exists(self.alpgConf.outputPath) and not self.force:
			raise Exception("ALPG data already exists")
		p = subprocess.Popen(
			" ".join(["/home/wowditi/anaconda3/bin/python", self.alpgConf.alpg, '-c', self.alpgConf.configName, '-o',
			          self.alpgConf.outputDir]), shell=True, universal_newlines=True)
		p.wait()
		self.removeEdgeCases()

	def createDEMKitComposer(self):
		if os.path.exists('%s/%s.py' % (self.demkitConf.workspace, self.alpgConf.configName)) and not self.force:
			raise Exception("DEMKit composer file already exists!")
		with open('%s/%s.py' % (self.demkitConf.workspace, self.alpgConf.configName), 'w') as f:
			f.write("\n".join([
				"from util.modelComposer import ModelComposer",
				"composer = ModelComposer('%s')" % self.alpgConf.configName,
				"composer.add('misc/components.py')",
				"composer.add('settings/settings_%s.py')" % self.alpgConf.configName,
				"composer.add('misc/alpgdata.py')",
				"composer.add('environment/simulator.py')",
				"composer.add('environment/global.py')",
				"composer.add('house/connectedhouse.py')",
				"composer.add('house/baseload.py')",
				"composer.add('house/pv.py')",
				"composer.add('house/timeshifters.py')" if "TimeShiftable" in self.devices else "",
				"composer.add('house/electricvehicle.py')" if "EV" in self.devices else "",
				"composer.add('house/battery.py')" if "Battery" in self.devices else "",
				"composer.add('street/streetcontrol.py')",
				"composer.add('street/street.py')",
				"composer.compose()",
				"composer.load()"
			]))

	def createDEMKitConfig(self):
		if os.path.exists('%s/settings/settings_%s.py' % (
				self.demkitConf.workspace, self.alpgConf.configName)) and not self.force:
			raise Exception("DEMKit settings file already exists!")
		with open('%s/settings/settings_%s.py' % (self.demkitConf.workspace, self.alpgConf.configName), 'w') as f:
			f.write("\n".join([
				"from datetime import datetime, timedelta",
				"from pytz import timezone",
				"timeZone = timezone('Europe/Amsterdam')",
				"startTime = int(timeZone.localize(datetime(2018, 1, 1)+timedelta(days=%d)).timestamp())" % self.alpgConf.startDay,
				"timeOffset = -1 *int(timeZone.localize(datetime(2018, 1, 1)).timestamp())",
				"timeBase = 900",
				"intervals = %d*24*int(3600/timeBase)" % (self.alpgConf.numDays - 2),
				"database = 'dem'",
				"dataPrefix = '%s-'" % self.alpgConf.configName,
				"numOfHouses = %d" % self.alpgConf.numHouses,
				"alpgFolder = '%s/'" % self.alpgConf.outputPath,
				"logDevices = True",
				"enablePersistence = False",
				"useEliaPV = False",
				"useCtrl = True",
				"useAuction = False",
				"usePlAuc = False",
				"useCongestionPoints = False",
				"useMultipleCommits = False",
				"useChildPruning = False",
				"useIslanding = False",
				"ctrlTimeBase = 900",
				"useEC = False",
				"usePP = True",
				"useQ = False",
				"useMC = False",
				"clearDB = False",
				"random.seed(1337)"
			]))

	def getFromRS(self, data, name):
		return data[('%s-devices' % self.alpgConf.configName, {'name': name})]

	def retrieveDEMKitResults(self) -> (
			Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]):
		dbClient = InfluxDBClient('localhost', '8086', 'admin', 'admin', 'dem')
		series: ResultSet = dbClient.query(f'show series from "{self.alpgConf.configName}-devices"; ')
		seriesDicts: List[Dict[str, str]] = [
			{elem.split('=', 1)[0]: elem.split('=', 1)[1] for elem in entry['key'].split(',')[1:]} for entry in
			series.get_points()]
		evNames: List[str] = [elem['name'] for elem in
		                      filter(lambda x: x['devtype'] == 'BufferTimeshiftable', seriesDicts)]
		tsNames: List[str] = [elem['name'] for elem in filter(lambda x: x['devtype'] == 'Timeshiftable', seriesDicts)]
		batteryNames: List[str] = [elem['name'] for elem in filter(lambda x: x['devtype'] == 'Buffer', seriesDicts)]
		loadNames: List[str] = [elem['name'] for elem in filter(lambda x: x['devtype'] == 'Curtailable', seriesDicts)]
		timeZone = timezone('Europe/Amsterdam')
		start = int(timeZone.localize(datetime(2018, 1, 1) + timedelta(days=self.alpgConf.startDay)).timestamp())
		end = int(timeZone.localize(
			datetime(2018, 1, 1) + timedelta(days=self.alpgConf.startDay + self.alpgConf.numDays - 2)).timestamp())
		data: ResultSet = dbClient.query(
			f'SELECT sum("W-power.real.c.POWER")*15 FROM "{self.alpgConf.configName}-devices" WHERE time >= {start}s and time <= {end}s GROUP BY time(15m), "name" fill(previous)')
		evCosts: Dict[str, List[float]] = {ev: [round(point['sum'], 6) for point in self.getFromRS(data, ev)] for ev in
		                                   evNames}
		tsCosts: Dict[str, List[float]] = {
			ts if ts[0:2] == 'WM' else f'DW{ts.rsplit("-",1)[1]}': [round(point['sum'], 6) for point in
			                                                        self.getFromRS(data, ts)] for ts in tsNames}
		batteryCosts: Dict[str, List[float]] = {battery: [point['sum'] for point in self.getFromRS(data, battery)]
		                                        for battery in batteryNames}
		staticCosts: Dict[str, List[float]] = {load: [point['sum'] for point in self.getFromRS(data, load)] for load
		                                       in loadNames}
		staticCosts: List[float] = [sum([staticCosts[load][i] for load in loadNames]) for i in
		                            range(len(staticCosts[loadNames[0]]))]
		dbClient.close()
		return tsCosts, evCosts, batteryCosts, staticCosts

	def getIntervals(self, data: List[float], length: int) -> List[IntInterval]:
		intervals = list()
		it = iter(enumerate(data))
		for idx, point in it:
			if point != 0.0:
				intervals.append(IntInterval(idx, idx + length))
				next(islice(it, length, length), None)
		return intervals

	@staticmethod
	def plot(plot: str, values: str, tsCosts: Dict[str, List[float]], evCosts: Dict[str, List[float]],
	         batteryCosts: Dict[str, List[float]], staticCosts: List[float]):
		summedTS = [sum([vals[i] for vals in tsCosts.values()]) for i in range(len(staticCosts))]
		summedEV = [sum([vals[i] for vals in evCosts.values()]) for i in range(len(staticCosts))]
		summedBat = [sum([vals[i] for vals in batteryCosts.values()]) for i in range(len(staticCosts))]
		combined = [static + summedTS[idx] + summedEV[idx] + summedBat[idx] for idx, static in enumerate(staticCosts)]
		plt.rcParams.update({'font.size': 22})
		plt.figure(figsize=(20, 5))
		plt.bar(range(len(combined)), combined, color='limegreen', linewidth=8.0, label='Combined')
		import pylab
		plt.xticks()
		# plt.stackplot(range(len(combined)), combined,color='limegreen')
		# plt.plot([], [], label='Combined', color='limegreen', linewidth=8.0)
		plt.plot(range(len(summedTS)), summedTS, linewidth=4.0, label="Timeshiftable Costs", color='red')
		plt.plot(range(len(summedEV)), summedEV, linewidth=4.0, label="EV Costs", color='blue')
		plt.plot(range(len(summedBat)), summedBat, linewidth=4.0, label="Battery Costs", color='yellow')
		plt.plot(range(len(staticCosts)), staticCosts, linewidth=4.0, label="staticCosts", color='purple')
		legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
		       fancybox=True, shadow=True, ncol=5, prop={'size': 20})
		plt.savefig(plot, format='eps', dpi=5000, bbox_inches='tight', pad_inches=0)
		with open(values, 'wb') as file:
			pickle.dump(
				dict(summedTS=summedTS, summedEV=summedEV, summedBat=summedBat, combined=combined, tsCosts=tsCosts,
				     evCosts=evCosts, batteryCosts=batteryCosts, staticCosts=staticCosts), file,
				pickle.HIGHEST_PROTOCOL)

	def validate_results(self, tsCosts: Dict[str, List[float]], evCosts: Dict[str, List[float]],
	                     batteryCosts: Dict[str, List[float]], staticCosts: List[float]):
		# test whether ts and ev are only active within intervals
		if "TimeShiftable" in self.devices:
			for ts in tsCosts:
				tsId = int(ts[2:])
				if ts[0:2] == 'WM':
					if f'House{tsId}' not in self.situation.washingmachineTimes:
						continue
					actualIntervals = self.situation.washingmachineTimes[f'House{tsId}']
					consumption = sum(self.situation.washingmachineConsumption[f'House{tsId}'])
					cLength = len(self.situation.washingmachineConsumption[f'House{tsId}'])
				elif ts[0:2] == 'DW':
					if f'House{tsId}' not in self.situation.dishwasherTimes:
						continue
					actualIntervals = self.situation.dishwasherTimes[f'House{tsId}']
					consumption = sum(self.situation.dishwasherConsumption[f'House{tsId}'])
					cLength = len(self.situation.dishwasherConsumption[f'House{tsId}'])
				else:
					raise Exception(f"TS {ts} should not exist?!")
				activeIntervals = self.getIntervals(tsCosts[ts], cLength)
				for idx, activeInterval in enumerate(activeIntervals):
					interval = actualIntervals[idx]
					if not (interval.start <= activeInterval.start < interval.end and
					        interval.start < activeInterval.end <= interval.end):
						print(activeIntervals)
						print(tsCosts[ts])
						raise Exception(f"The interval of {ts} which is {activeInterval} cannot fit inside {interval}")
					if round(sum(tsCosts[ts][activeInterval.start:activeInterval.end]), 3) != round(consumption, 3):
						print(interval)
						print(tsCosts[ts][activeInterval.start:activeInterval.end])
						raise Exception(
							f"The consumption of {ts} is {sum(tsCosts[ts][activeInterval.start:activeInterval.end])} which is not {consumption}")
			print("All ts tested succesfully!")
		if "EV" in self.devices:
			for ev in evCosts:
				evId = int(ev[2:])
				evObj: EV = self.situation.EVList[evId]
				for idx, start in enumerate([start for start in evObj.startTimes if
				                             start * 15 * 60 < (self.alpgConf.numDays - 2) * 3600 * 24]):
					if round(sum(evCosts[ev][start:evObj.endTimes[idx]]), 3) != round(evObj.requiredCharge[idx] * 60,
					                                                                  3):
						raise Exception(
							f"The consumption of {ev} is {sum(evCosts[ev][start:evObj.endTimes[idx]])} which is not {evObj.requiredCharge[idx]*60}")
			print("All ev tested succesfully!")
		if "Battery" in self.devices:
			for battery in batteryCosts:
				batteryId = int(battery[3:])
				batteryObj: Battery = self.situation.BatteryList[batteryId]
				SoC = batteryObj.initialCharge
				capacity = batteryObj.capacity
				maxRate = batteryObj.maximumRate / 60
				for idx, c in enumerate(batteryCosts[battery][:(self.alpgConf.numDays-2)*96]):
					if round(abs(c / 60), 3) > maxRate:
						raise Exception(
							f"The battery{batteryId} is charging {c/60} while the maximumrate is {maxRate}, these are multiplied by 15.")
					SoC += c / 60
					if not (0 <= round(SoC, 1) <= capacity):
						print("DEBUG", batteryObj.initialCharge)
						print("DEBUG", batteryCosts[battery])
						raise Exception(
							f"The battery schedule of battery {batteryId} results in a SoC of {SoC} at time {idx}, which in not within 0,{capacity}")
			print("All batteries succesfully tested!")
		if int(sum(staticCosts[:96 * (self.alpgConf.numDays - 2)])) != sum(
				self.situation.staticCosts[:96 * (self.alpgConf.numDays - 2)]):
			raise Exception(
				f"The static costs are {int(sum(staticCosts[:96 * (self.alpgConf.numDays - 2)]))} which is not the same as {sum(self.situation.staticCosts[:96*(self.alpgConf.numDays-2)])}!")
		print("Successfully tested the static costs!")
		performanceCost = 0
		for i in range(96 * (self.alpgConf.numDays - 2)):
			combinedCost = 0
			combinedCost += sum([ts[i] for _, ts in tsCosts.items()])
			combinedCost += sum([ev[i] for _, ev in evCosts.items()])
			combinedCost += sum([battery[i] for _, battery in batteryCosts.items()])
			combinedCost += staticCosts[i]
			performanceCost += (combinedCost / (60 * self.alpgConf.numHouses)) ** 2
		return performanceCost

	def runDEMKit(self):
		self.createDEMKitComposer()
		self.createDEMKitConfig()
		if self.force:
			dbClient = InfluxDBClient('localhost', '8086', 'admin', 'admin', 'dem')
			for resName in ['controllers', 'devices', 'environment', 'host']:
				dbClient.query(f'drop measurement "{self.alpgConf.configName}-{resName}"')
		p = subprocess.Popen(" ".join(
			["/home/wowditi/anaconda3/bin/python", self.demkitConf.demkit, '-f', self.demkitConf.folder, '-m',
			 self.alpgConf.configName]), shell=True, universal_newlines=True,
			cwd=self.demkitConf.demkit.rsplit('/', 1)[0])
		p.wait()
		tsCosts, evCosts, batteryCosts, staticCosts = self.retrieveDEMKitResults()
		perfCost = self.validate_results(tsCosts, evCosts, batteryCosts, staticCosts)
		self.plot(self.demkitConf.plot, self.demkitConf.plotValues, tsCosts, evCosts, batteryCosts, staticCosts)
		return perfCost

	def rewriteActivationToCosts(self, activationData: Sequence[SimPoint], costs: List[float]) -> List[float]:
		result: List[float] = [0.0] * (96 * (self.alpgConf.numDays - 2))
		for point in filter(lambda x: x.value, activationData):
			result[point.time:point.time + len(costs)] = costs
		return result

	def UppaalListToList(self, data: Sequence[SimPoint]) -> List[float]:
		result: List[float] = [0.0] * (96 * (self.alpgConf.numDays - 2) + 1)
		prev = 0
		prevIdx = 0
		for point in data:
			while prevIdx < point.time - 1:
				prevIdx += 1
				result[prevIdx] = prev
			result[point.time] = float(point.value)
			prev = point.value
			prevIdx = point.time
		return result

	@staticmethod
	def fixUppaalEV(fillValues: List[float], charges: List[float]):
		for j, (idx, charge) in enumerate([(idx, charge) for idx, charge in enumerate(charges) if
		                                   int(charge) != 0 and int(charges[idx - 1]) == 0]):
			end = next((i + idx + 1 for i, c in enumerate(charges[idx + 1:]) if int(c) == 0))
			steps = [(idx, charge)] + [(h + idx + 1, c) for h, c in enumerate(charges[idx + 1:end]) if
			                           charges[h + idx] == charges[h + idx - 1] != charges[h + idx + 1]]
			for h, (i, c) in enumerate(steps):
				if c - charges[i - 1] > fillValues[i - 1]:
					if charge == fillValues[i - 1] + fillValues[i]:
						fillValues[i - 2:end if h + 1 == len(steps) else steps[h + 1][0]] = fillValues[
						                                                                    i - 1:end + 1 if h + 1 == len(
							                                                                    steps) else
						                                                                    steps[h + 1][0] + 1]
					else:
						fillValues[i - 2] = c - charges[i - 1] - fillValues[i - 1]
		return fillValues

	def rewriteStrategoResult(self, result: Union[None, Dict[str, Sequence[SimPoint]]]):
		tsCosts: Dict[str, List[float]] = dict()
		evCosts: Dict[str, List[float]] = dict()
		batteryCosts: Dict[str, List[float]] = dict()
		for key in result:
			if key[:14] == 'TimeShiftables' and key[-9:] == 'Activated':
				houseId, devId = key.split('_')[1:3]
				if f'House{houseId}' in self.situation.washingmachineConsumption:
					if devId == '0':
						tsCosts[f'WM{houseId}'] = self.rewriteActivationToCosts(result[key],
						                                                        self.situation.washingmachineConsumption[
							                                                        f'House{houseId}'])
					elif f'House{houseId}' in self.situation.dishwasherConsumption:
						tsCosts[f'DW{houseId}'] = self.rewriteActivationToCosts(result[key],
						                                                        self.situation.dishwasherConsumption[
							                                                        f'House{houseId}'])
				elif f'House{houseId}' in self.situation.dishwasherConsumption:
					if devId == '0':
						tsCosts[f'DW{houseId}'] = self.rewriteActivationToCosts(result[key],
						                                                        self.situation.dishwasherConsumption[
							                                                        f'House{houseId}'])
			if key[:10] == 'fillValues':
				evId = re.split('[\[\]]', key)[1]
				if int(evId) in self.situation.EVList:
					evCosts[f'EV{evId}'] = self.UppaalListToList(result[
						                                             key])  # self.fixUppaalEV(self.UppaalListToList(result[key]), self.UppaalListToList(result[f'charges[{evId}]']))
			if key[:16] == 'batteryFillValue':
				batteryId = re.split('[\[\]]', key)[1]
				if int(batteryId) in self.situation.BatteryList:
					batteryCosts[f'BAT{batteryId}'] = self.UppaalListToList(result[key])
		return tsCosts, evCosts, batteryCosts, self.situation.staticCosts[:96 * (self.alpgConf.numDays - 2)]

	def runStratego(self):
		if os.path.exists(self.strategoConf.workingDir):
			if not self.force:
				raise Exception("Startego working directory already exists!")
		else:
			os.makedirs(self.strategoConf.workingDir)
		s = self.strategoConf
		copyfile(s.baseModel, s.workingModel)
		alg = StrategoAlgorithm(s.result, s.query, s.workingModel, s.verifyta,
		                        Model(s.workingModel, f'{self.alpgConf.outputPath}/UppaalDeclarion.txt'),
		                        s.requiredTemplates, s.components, (self.alpgConf.numDays - 2) * 96, optimize=True,
		                        numSimulations=100, iterations=2, seed=1548404740)
		result = alg.runStratego(s.result)
		tsCosts, evCosts, batteryCosts, staticCosts = self.rewriteStrategoResult(result)
		perfCost = self.validate_results(tsCosts, evCosts, batteryCosts, staticCosts)
		self.plot(self.strategoConf.plot, self.strategoConf.plotValues, tsCosts, evCosts, batteryCosts, staticCosts)
		return perfCost, alg.seed


if __name__ == "__main__":
	if os.name == 'nt':
		_base = 'C:/Users/niels/Documents/Afstuderen'
		_verifyta = "%s/uppaal-stratego-4.1.20/bin-Win32/verifyta.exe" % _base
	else:
		_base = '/home/wowditi/Documents/Afstuderen'
		_verifyta = "%s/uppaal-stratego-4.1.20/bin-Linux/verifyta" % _base
	_alpg = "%s/alpg/profilegenerator.py" % _base
	_configName = "test3"
	_resultsDir = _configName  # "results8"
	_results = "%s/UppaalInput/alpg/%s" % (_base, _resultsDir)
	_resultsFolder = f'{_base}/UppaalInput/results/{_configName}'
	_finalResult = f'{_resultsFolder}/result.txt'
	_demkitPlot = f'{_resultsFolder}/DEMKit.eps'
	_strategoPlot = f'{_resultsFolder}/Stratego.eps'
	_demkitValues = f'{_resultsFolder}/DEMKit.pickle'
	_strategoValues = f'{_resultsFolder}/Stratego.pickle'
	_configDir = "%s/UppaalInput/alpg-configs/" % _base
	_config = "%s%s.py" % (_configDir, _configName)
	_demkit = '%s/demkitsim/DEMKit/demkit.py' % _base
	_demkitWorkSpaceFolder = 'DEMKit_example'
	_demkitWorkSpace = '%s/demkitsim/workspace/%s' % (_base, _demkitWorkSpaceFolder)
	spec = importlib.util.spec_from_file_location("config_module", _config)
	_configCode = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(_configCode)
	_numDays = _configCode.numDays
	_startDay = _configCode.startDay
	_numHouses = _configCode.numHouses
	_baseModel = f'{_base}/Models/Final.xml'
	_strategoWorkingDir = f'{_base}/StrategoOnline/{_configName}/'
	_queryL = f'{_strategoWorkingDir}query.q'
	_modelL = f'{_strategoWorkingDir}Model.xml'
	_resultL = f'{_strategoWorkingDir}Result.txt'
	_requiredTemplates: List[str] = ['Time', 'CostCalculator', 'MainLoop']
	_tsComp: Component = Component("TimeShiftable", 1, True, 1, LearningParams(5, 5, 1, 1))
	_evComp: Component = Component("EV", 1, True, 1, LearningParams(3, 3, 1, 1))
	_batComp: Component = Component("Battery", 1, True, 10, LearningParams(50, 50, 1, 1))
	_components: List[Component] = [_tsComp, _evComp, _batComp]
	_devices = ["Battery"]#, "EV", "Battery"]
	_components: List[Component] = [comp for comp in _components if comp.template in _devices]
	_alpgConf = ALPGConfig(_alpg, _config, _configName, _results, _resultsDir, _numDays, _startDay, _numHouses)
	_demkitConf = DEMKitConfig(_demkit, _demkitWorkSpace, _demkitWorkSpaceFolder, _demkitPlot, _demkitValues)
	_strategoConf = StrategoConfig(_verifyta, _strategoWorkingDir, _baseModel, _modelL, _queryL, _requiredTemplates,
	                               _components, _resultL, _strategoPlot, _strategoValues)
	comparator = Comparator(_alpgConf, _demkitConf, _strategoConf, _devices, _resultsFolder, force=True)
	comparator.createSituation()
	comparator.createDeclaration()
	demkitTime = time.time()
	demkitResult = comparator.runDEMKit()
	demkitTime = time.time() - demkitTime
	strategoTime = time.time()
	strategoResult, strategoSeed = comparator.runStratego()
	strategoTime = time.time() - strategoTime
	print("Final comparison:", demkitResult, strategoResult)
	print(
		f"The Stratego schedule puts {round(strategoResult/demkitResult*100, 2)}% as much strain on the grid as the demkit schedule.")
	with open(_finalResult, 'w') as f:
		f.write("\n".join([
			f'NumTS:    {len(comparator.situation.dishwasherTimes)+len(comparator.situation.washingmachineTimes) if "TimeShiftable" in comparator.devices else 0}',
			f'NumEV:    {len(comparator.situation.EVList) if "EV" in comparator.devices else 0}',
			f'NumBat:   {len(comparator.situation.BatteryList) if "Battery" in comparator.devices else 0}',
			f'Length:   {comparator.alpgConf.numDays-2} days',
			f'DEMKit:',
			f'  Duration:   {demkitTime}',
			f'  Score:      {demkitResult}',
			f'Uppaal:',
			f'  Duration:   {strategoTime}',
			f'  Score:      {strategoResult}',
			f'  Seed:       {strategoSeed}',
			f'  Learning:   Logistic Regression',
			f'  Resolution: {(comparator.alpgConf.numDays - 1) * 96+1}',
			f'  TimeShiftables:',
			f'      Amount:         {_tsComp.maxActive}',
			f'      Optimized:      {_tsComp.optimize}',
			f'      Good-runs:      {_tsComp.learningParams.gruns if "TimeShiftable" in _devices else 0}',
			f'      Total-runs:     {_tsComp.learningParams.truns if "TimeShiftable" in _devices else 0}',
			f'      Runs-pr-state:  {_tsComp.learningParams.rprstate if "TimeShiftable" in _devices else 0}',
			f'      Eval-runs:      {_tsComp.learningParams.eruns if "TimeShiftable" in _devices else 0}',
			f'  EVs:',
			f'      Amount:         {_evComp.maxActive}',
			f'      Optimized:      {_evComp.optimize}',
			f'      Good-runs:      {_evComp.learningParams.gruns if "EV" in _devices else 0}',
			f'      Total-runs:     {_evComp.learningParams.truns if "EV" in _devices else 0}',
			f'      Runs-pr-state:  {_evComp.learningParams.rprstate if "EV" in _devices else 0}',
			f'      Eval-runs:      {_evComp.learningParams.eruns if "EV" in _devices else 0}',
			f'  Batteries:',
			f'      Amount:         {_batComp.maxActive}',
			f'      Optimized:      {_batComp.optimize}',
			f'      Good-runs:      {_batComp.learningParams.gruns if "Battery" in _devices else 0}',
			f'      Total-runs:     {_batComp.learningParams.truns if "Battery" in _devices else 0}',
			f'      Runs-pr-state:  {_batComp.learningParams.rprstate if "Battery" in _devices else 0}',
			f'      Eval-runs:      {_batComp.learningParams.eruns if "Battery" in _devices else 0}',
			'',
			f"The Stratego schedule puts {round(strategoResult/demkitResult*100, 2)}% as much strain on the grid as the demkit schedule."
		]))
