from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import math
import numpy as np
import pandas as pd
import re
from typing import Dict, List


class Interval_t(object):
	def __init__(self, name, interval, offset=0):
		self.name = name
		self.start = interval.start - offset
		self.end = interval.end - offset

	def __str__(self):
		return "const %s %s = {%s, %s};" % (self.__class__.__name__, self.name, self.start, self.end)


class IntervalList_t(object):
	def __init__(self, name, intervals, longestList):
		self.name = name
		self.length = len(intervals)
		self.intervals = [Interval_t("%si%d" % (name, idx), interval) for idx, interval in
		                  enumerate(intervals)]  # type: List[Interval_t]
		self.longestList = longestList

	def __str__(self):
		interval_names = ['empty%s' % Interval_t.__name__] * self.longestList
		for idx, interval in enumerate(self.intervals):
			interval_names[idx] = interval.name
		return "const %s %s = {%d, {%s}};" % (self.__class__.__name__, self.name, self.length,
		                                      str(interval_names)[1:-1].replace('\'', ''))


class ConsumptionProfile_t(object):
	def __init__(self, name, consumption_profile, longestConsumptionList):
		self.name = name
		self.consumption_profile = consumption_profile
		self.longestConsumptionList = longestConsumptionList
		self.cost = sum(self.consumption_profile)

	def __str__(self):
		consumptions = [0] * self.longestConsumptionList
		for idx, consumption in enumerate(self.consumption_profile):
			consumptions[idx] = consumption
		return "const %s %s = {%d, {%s}};" % (
			self.__class__.__name__, self.name, len(self.consumption_profile),
			str([int(c) for c in consumptions])[1:-1])


class TimeShiftable_t(object):
	def __init__(self, name, intervals, consumption_profile, longestList, longestConsumptionList):
		self.name = name
		self.intervalList = IntervalList_t("%slist" % name, intervals, longestList)
		self.consumption_profile = ConsumptionProfile_t("%scp" % name, consumption_profile, longestConsumptionList)
		self.cost = self.consumption_profile.cost * self.intervalList.length

	def __str__(self):
		return "const %s %s = {%s, %s};" % (
			self.__class__.__name__, self.name, self.consumption_profile.name, self.intervalList.name)


class Battery_t(object):
	def __init__(self, name, chargeRate, capacity, SoC):
		self.name = name
		self.chargeRate = chargeRate
		self.capacity = capacity * 60
		self.SoC = SoC * 60

	def __str__(self):
		return "const %s %s = {%s, %s, %s};" % (
			self.__class__.__name__, self.name, self.chargeRate, self.capacity, self.SoC)


class ElectricalVehicle_t(object):
	discretization_factor = 0

	def __init__(self, name, ev, longestList):
		self.name = name
		self.requiredCharge = ev.requiredCharge
		self.capacity = ev.capacity
		self.maximumRate = int(ev.maximumRate) * ElectricalVehicle_t.discretization_factor
		self.longestList = longestList
		self.intervalList = IntervalList_t("%slist" % name,
		                                   [Interval(*x) for x in list(zip(ev.startTimes, ev.endTimes))], longestList)
		self.maxTSCostList = [0] * longestList

	def setMaxTSCosts(self, timeShiftables):
		for i in range(self.intervalList.length):
			start = self.intervalList.intervals[i].start
			end = self.intervalList.intervals[i].end
			self.maxTSCostList[i] = int(
				sum([ts.consumption_profile.cost for ts in timeShiftables for interval in ts.intervalList.intervals \
				     if start <= interval.start <= end or start <= interval.end <= end]) / (end - start))

	def __str__(self):
		chargeList = [0] * self.longestList
		for idx, charge in enumerate(self.requiredCharge):
			chargeList[idx] = charge
		return "const %s %s = {%s, %s, %s, %s, %s};" % (
			self.__class__.__name__, self.name, self.maximumRate, self.capacity, arrayToUppaal(chargeList),
			arrayToUppaal(self.maxTSCostList), self.intervalList.name)


class House_t(object):
	def __init__(self, name, maxDevices, longestList, longestConsumptionList):
		self.name = name
		self.timeShiftables = list()  # type: List[TimeShiftable_t]
		self.longestList = longestList
		self.longestConsumptionList = longestConsumptionList
		self.maxDevices = maxDevices
		self.length = 0
		self.ev = None
		self.battery = None

	def createTimeShiftables(self, devices, consumption_profile):
		l = len(self.timeShiftables)
		self.timeShiftables += [
			(TimeShiftable_t("%sd%d" % (self.name, idx + l), intervals, consumption_profile, self.longestList,
			                 self.longestConsumptionList)) for idx, intervals in
			enumerate(devices)]  # type: List[TimeShiftable_t]
		self.length = len(self.timeShiftables)

	def createEv(self, ev):
		self.ev = ElectricalVehicle_t(self.name + "ev", ev, self.longestList)

	def createBattery(self, battery: Battery):
		self.battery = Battery_t(self.name + "battery", battery.maximumRate, battery.capacity, battery.initialCharge)

	def get_declaration(self):
		result = ""
		if not self.ev:
			self.createEv(EV(0))
		for timeShiftable in self.timeShiftables:
			for interval in timeShiftable.intervalList.intervals:
				result += str(interval) + '\n'
			result += str(timeShiftable.intervalList) + '\n'
			result += str(timeShiftable.consumption_profile) + '\n'
			result += str(timeShiftable) + '\n'
		if self.ev:
			for interval in self.ev.intervalList.intervals:
				result += str(interval) + '\n'
			result += str(self.ev.intervalList) + '\n'
			result += str(self.ev) + '\n'
		if self.battery:
			result += str(self.battery) + '\n'
		result += str(self) + '\n'
		return result

	def __str__(self):
		interval_names = ['empty%s' % TimeShiftable_t.__name__] * self.maxDevices
		for idx, timeShiftable in enumerate(self.timeShiftables):
			interval_names[idx] = timeShiftable.name
		battery = 'emptyBattery_t' if not self.battery else self.battery.name
		return "const %s %s = {%d, {%s}, %s, %s};" % (
			self.__class__.__name__, self.name, self.length, str(interval_names)[1:-1].replace('\'', ''), battery,
			self.ev.name)


class Battery(object):
	discretization_factor = 0

	def __init__(self, house, maximumRate, capacity, initialCharge):
		self.house = int(house)
		self.initialCharge = int(initialCharge)
		self.capacity = int(capacity)
		self.maximumRate = int(maximumRate) * Battery.discretization_factor


class EV(object):
	def __init__(self, house):
		self.house = house
		self.requiredCharge = list()
		self.capacity = 0
		self.maximumRate = 0
		self.startTimes = list()
		self.endTimes = list()

	def getLongestInterval(self):
		return max([self.endTimes[idx]-start for idx, start in enumerate(self.startTimes)])



def arrayToUppaal(array):
	if type(array) == list:
		return '{%s}' % ','.join([arrayToUppaal(elem) for elem in array])
	else:
		return str(array)


Interval = namedtuple('Interval', ['start', 'end'])


def parseStatic(base_dir: str, days_processed: int, offset: int, discretization_factor: int) -> (
		List[str], pd.DataFrame):
	with open(base_dir + 'HeatingSettings.txt') as f:
		num_houses = len(f.readlines())
	houseNames = ['House%d' % i for i in range(num_houses)]
	ep = base_dir + "Electricity_Profile%s.csv"
	eps = [
		"GroupElectronics", "GroupFridges", "GroupInductive", "GroupLighting",
		"GroupOther", "GroupStandby", "PVProduction"
	]
	ep_data = dict(
		Base=pd.read_csv(
			ep % "", error_bad_lines=False, sep=';', names=houseNames))
	for Group in eps:
		ep_data[Group] = pd.read_csv(
			ep % "_%s" % Group, error_bad_lines=False, sep=';', names=houseNames)
	summed_ep = ep_data['Base']
	summed_ep = summed_ep.add(ep_data["PVProduction"], fill_value=0)
	max_days = len(summed_ep) / 60 / 24
	if not days_processed or days_processed == 0:
		days_processed = max_days
	end_time = offset * discretization_factor * 60 + days_processed * 3600 * 24
	discretized_list = [0] * int(days_processed * 60 * 24 / discretization_factor)
	for i in range(len(discretized_list)):
		discretized_list[i] = summed_ep[i * discretization_factor:(i + 1) * discretization_factor].sum()
	discretized_ep = pd.DataFrame(data=discretized_list, columns=summed_ep.columns)
	summed_ep = discretized_ep
	return houseNames, summed_ep


def createWashingMachine(base_dir: str, days_calculated: int, offset: int, discretization_factor: int) -> (
		Dict[str, List[Interval]], Dict[str, List[int]]):
	washingmachine_consumption = dict()
	with open(base_dir + 'WashingMachine_Profile.txt') as f:
		for line in f.readlines():
			washingmachine_consumption['House%s' % line.split(':', 1)[0]] = [x for x in (
				list(eval(re.sub(r"complex\((\d+\.\d+),\s[0-9.]+\)", r"\1", line.split(':', 1)[1]))))]
	for key in washingmachine_consumption.keys():
		washingmachine_consumption[key] = [
			sum(washingmachine_consumption[key][i * discretization_factor:(i + 1) * discretization_factor]) for i in
			range(
				math.floor((len(washingmachine_consumption[key]) - 1) / discretization_factor) + 1)]
	washingmachine_start_times = dict()
	with open(base_dir + 'WashingMachine_Starttimes.txt') as f:
		for line in f.readlines():
			washingmachine_start_times['House%s' % line.split(':', 1)[0]] = [int(x) for x in
			                                                                 (line.split(':', 1)[1].rstrip().split(
				                                                                 ','))]
	for key in washingmachine_start_times:
		for i in range(len(washingmachine_start_times[key])):
			washingmachine_start_times[key][i] = int(
				washingmachine_start_times[key][i] / discretization_factor / 60) - offset
	washingmachine_end_times = dict()
	with open(base_dir + 'WashingMachine_Endtimes.txt') as f:
		for line in f.readlines():
			washingmachine_end_times['House%s' % line.split(':', 1)[0]] = [int(x) for x in
			                                                               (line.split(':', 1)[1].rstrip().split(','))]
	for key in washingmachine_end_times:
		for i in range(len(washingmachine_end_times[key])):
			washingmachine_end_times[key][i] = int(
				washingmachine_end_times[key][i] / discretization_factor / 60) - offset
	washingmachine_times = {key: list(zip(washingmachine_start_times[key], washingmachine_end_times[key])) for key in
	                        washingmachine_end_times.keys()}
	for interval_list in washingmachine_times.values():
		for i in range(len(interval_list)):
			interval_list[i] = Interval(*interval_list[i])
	for house in washingmachine_times:
		for idx, interval in enumerate(washingmachine_times[house]):
			if interval.end > days_calculated * 60 * 24 / discretization_factor:
				washingmachine_times[house] = washingmachine_times[house][0:idx]
				break
	return washingmachine_times, washingmachine_consumption


def createDishwasher(base_dir: str, days_calculated: int, offset: int, discretization_factor: int) -> (
		Dict[str, List[Interval]], Dict[str, List[int]]):
	dishwasher_consumption = dict()
	with open(base_dir + 'Dishwasher_Profile.txt') as f:
		for line in f.readlines():
			dishwasher_consumption['House%s' % line.split(':', 1)[0]] = [x for x in (
				list(eval(re.sub(r"complex\((\d+\.\d+),\s[0-9.]+\)", r"\1", line.split(':', 1)[1]))))]
	for key in dishwasher_consumption.keys():
		dishwasher_consumption[key] = [
			sum(dishwasher_consumption[key][i * discretization_factor:(i + 1) * discretization_factor]) for i in
			range(math.floor((len(dishwasher_consumption[key]) - 1) / discretization_factor) + 1)]
	dishwasher_start_times = dict()
	with open(base_dir + 'Dishwasher_Starttimes.txt') as f:
		for line in f.readlines():
			dishwasher_start_times['House%s' % line.split(':', 1)[0]] = [int(x) for x in
			                                                             (line.split(':', 1)[1].rstrip().split(','))]
	for key in dishwasher_start_times:
		for i in range(len(dishwasher_start_times[key])):
			dishwasher_start_times[key][i] = int(dishwasher_start_times[key][i] / discretization_factor / 60) - offset
	dishwasher_end_times = dict()
	with open(base_dir + 'Dishwasher_Endtimes.txt') as f:
		for line in f.readlines():
			dishwasher_end_times['House%s' % line.split(':', 1)[0]] = [int(x) for x in
			                                                           (line.split(':', 1)[1].rstrip().split(','))]
	print(dishwasher_end_times)
	for key in dishwasher_end_times:
		for i in range(len(dishwasher_end_times[key])):
			dishwasher_end_times[key][i] = int(dishwasher_end_times[key][i] / discretization_factor / 60) - offset
	print(dishwasher_end_times)
	dishwasher_times = {key: list(zip(dishwasher_start_times[key], dishwasher_end_times[key])) for key in
	                    dishwasher_end_times.keys()}
	for interval_list in dishwasher_times.values():
		for i in range(len(interval_list)):
			interval_list[i] = Interval(*interval_list[i])
	for house in dishwasher_times:
		for idx, interval in enumerate(dishwasher_times[house]):
			if interval.end > days_calculated * 60 * 24 / discretization_factor:
				dishwasher_times[house] = dishwasher_times[house][0:idx]
				break
	return dishwasher_times, dishwasher_consumption


def createEVList(base_dir: str, days_calculated: int, offset: int, discretization_factor: int) -> Dict[int, EV]:
	EVList = dict()  # type: Dict[int, EV]
	with open(base_dir + 'ElectricVehicle_RequiredCharge.txt') as f:
		for line in f.readlines():
			evId = int(line.split(':', 1)[0])
			ev = EV(evId)
			EVList[evId] = ev
			ev.requiredCharge = [int(x) for x in line.split(':')[1].split(',')]

	with open(base_dir + 'ElectricVehicle_Specs.txt') as f:
		for idx, line in enumerate(f.readlines()):
			evId = int(line.split(':', 1)[0])
			EVList[evId].capacity, EVList[evId].maximumRate = line.split(':')[1].split(',')

	with open(base_dir + 'ElectricVehicle_Starttimes.txt') as f:
		for idx, line in enumerate(f.readlines()):
			evId = int(line.split(':', 1)[0])
			EVList[evId].startTimes = [int(int(x) / discretization_factor / 60) - offset for x in
			                           line.split(':')[1].split(',')]

	with open(base_dir + 'ElectricVehicle_Endtimes.txt') as f:
		for idx, line in enumerate(f.readlines()):
			evId = int(line.split(':', 1)[0])
			EVList[evId].endTimes = [int(int(x) / discretization_factor / 60) - offset for x in
			                         line.split(':')[1].split(',')]

	for ev in EVList.values():
		for idx, endtime in enumerate(ev.endTimes):
			if endtime > days_calculated * 60 * 24 / discretization_factor:
				ev.endTimes = ev.endTimes[0:idx]
				ev.startTimes = ev.startTimes[0:idx]
				break
	return EVList


def createBatteryList(base_dir: str) -> Dict[int, Battery]:
	BatteryList = dict()  # type: Dict[int, Battery]
	with open(base_dir + 'BatterySettings.txt') as f:
		for line in f.readlines():
			newBattery = Battery(*re.split('[:,]', line))
			BatteryList[int(newBattery.house)] = newBattery
	return BatteryList


def write_structs(f):
	f.write(
		"""
typedef struct {
	int start;
	int end;
} Interval_t;

typedef struct {
	int length;
	Interval_t intervalList[longestIntervalList];
} IntervalList_t;

typedef struct {
	int length;
	int data[longestConsumption];
} ConsumptionProfile_t;

typedef struct {
	ConsumptionProfile_t consumption;
	IntervalList_t data;
} TimeShiftable_t;

typedef struct {
	UInt32_t maximumChargeRate;
	UInt32_t capacity;
	UInt32_t initialSoc;
} Battery_t;

typedef struct {
	UInt32_t maximumChargeRate;
	UInt32_t capacity;
	UInt32_t requiredCharge[longestIntervalList];
	Int32_t maxTsCosts[longestIntervalList];
	IntervalList_t intervals;
} ElectricalVehicle_t;

typedef struct {
	int timeShiftableLength;
	TimeShiftable_t timeShiftables[maxNumTimeShiftables];
	Battery_t battery;
	ElectricalVehicle_t ev;
} House_t;

""")


def write_houses(dest, houseList, houseNames, discretization_factor, summed_ep, battery_List, ev_list, max_devices,
                 longest_interval, longest_consumption):
	with open(dest, 'w+') as f:
		f.write('\n'.join(["""typedef int[-8388608, 8388607] Int24_t;
typedef int[0, 8388607] UInt24_t;
typedef int[-2147483648, 2147483647] Int32_t;
typedef int[0, 2147483647] UInt32_t;
typedef int[-128, 127] Int8_t;
typedef int[0, 127] UInt8_t;
typedef int[0, 65535] UInt_t;
UInt32_t square(Int24_t a) {return a*a;}
Int32_t abs(Int32_t a) {return a < 0 ? -a : a;}
clock time;
hybrid clock cost;
UInt_t intTime;
Int24_t HouseDynamicCosts;
Int24_t HouseBatteryCosts;
broadcast chan step;
broadcast chan batteryStep;
broadcast chan evStep;
broadcast chan tsStep;
Int8_t chargingState;""",
		                   "const UInt_t reductionFactor = %s;" % (len(houseNames) * 60),
		                   "const UInt_t num_entries = %s;" % len(summed_ep.House0),
		                   "const UInt_t end_time = num_entries;",
		                   "const int maxNumTimeShiftables = %s;" % max_devices,
		                   "const int longestIntervalList = %s;" % longest_interval,
		                   "const int longestConsumption = %s;" % longest_consumption,
		                   "const int numHouses = %s;" % len(houseList),
		                   """
typedef int[0, maxNumTimeShiftables-1] timeShiftable_t;
typedef int[0, numHouses-1] house_t;
typedef int[0, longestIntervalList-1] interval_t;
typedef int[0, num_entries-1] entries_t;
broadcast chan batteryStartSwitch;
broadcast chan batteryStopSwitch;
			   """,
		                   f'const int longestInterval = {max([ev.getLongestInterval() for _, ev in ev_list.items()]+[0])};',
		                   "Int32_t batteryFillLimit[house_t] = %s;" % arrayToUppaal([-2147480000] * len(houseList)),
		                   "meta Int24_t batteryFillValue[house_t];",
		                   "UInt32_t SoC[house_t] = %s;" % arrayToUppaal(
			                   [battery_List[i].initialCharge*60 if i in battery_List.keys() else 0 for i in
			                    range(len(houseList))]),
		                   """
Int24_t latestCost;

broadcast chan tsStartChan[house_t][timeShiftable_t];
						   """,
		                   "Int32_t fillLimits[house_t] = %s;" % arrayToUppaal([-2147480000] * len(houseList)),
		                   """
UInt32_t fillValues[house_t];
UInt32_t charges[house_t];
int[0, longestIntervalList] evIntervals[house_t];

house_t order[house_t];
						   """,
		                   "int[-1, end_time] tsStartTimes[house_t][timeShiftable_t] = %s;" % arrayToUppaal(
			                   [[-1] * max_devices] * len(houseList))
		                   ]) + '\n')
		write_structs(f)
		# for houseName in houseNames:
		#    f.write("const int %sBase[num_entries] = {%s};\n" % (houseName, str(list(summed_ep[houseName]))[1:-1]))
		# f.write("const int HouseBases[house_t][num_entries] = {%s};\n" % str(["%sBase" % houseName for houseName in houseNames])[1:-1].replace('\'', ''))
		f.write("const int HouseBases[num_entries] = {%s};\n" % (
			','.join([str(sum([summed_ep[name][i] for name in houseNames])) for i in range(len(summed_ep.House0))])))
		f.write("const int HouseAvg = %s;\n" % (int(np.mean([np.mean(summed_ep[name]) for name in houseNames])) * 10))
		f.write("const %s empty%s = {0,0};\n" % (Interval_t.__name__, Interval_t.__name__))
		f.write("const %s empty%s = {0, {%s}};\n" % (ConsumptionProfile_t.__name__, ConsumptionProfile_t.__name__, str(
			([0] * longest_consumption))[1:-1]))
		f.write("const %s empty%s = {0, {%s}};\n" % (IntervalList_t.__name__, IntervalList_t.__name__, str(
			(['empty%s' % Interval_t.__name__] * longest_interval))[1:-1].replace('\'', '')))
		f.write("const %s empty%s = {empty%s, empty%s};\n" % (
			TimeShiftable_t.__name__, TimeShiftable_t.__name__, ConsumptionProfile_t.__name__, IntervalList_t.__name__))
		f.write("const %s empty%s = {0, 0, 0};\n" % (Battery_t.__name__, Battery_t.__name__))
		for house in houseList:
			f.write(house.get_declaration())
		f.write('const %s houses[house_t] = {%s};' % (
			House_t.__name__, str(['h%d' % i for i in range(len(houseList))])[1:-1].replace('\'', '')))
		f.write("""
Int24_t getHouseCost(int offset) {
    return HouseBases[intTime+offset]+HouseDynamicCosts+HouseBatteryCosts;
}""")


@dataclass
class Situation:
	staticCosts: List[int]
	EVList: Dict[int, EV]
	BatteryList: Dict[int, BatteryList]
	dishwasherTimes: Dict[str, List[Interval]]
	dishwasherConsumption: Dict[str, List[float]]
	washingmachineTimes: Dict[str, List[Interval]]
	washingmachineConsumption: Dict[str, List[float]]


def createHouses(base_dir: str, dest: str, days_processed: int, days_calculated: int, offset: int,
                 discretization_factor: int):
	ElectricalVehicle_t.discretization_factor = discretization_factor
	Battery.discretization_factor = discretization_factor
	houseNames, summed_ep = parseStatic(base_dir, days_processed, offset, discretization_factor)
	dishwasher_times, dishwasher_consumption = createDishwasher(base_dir, days_calculated, offset,
	                                                            discretization_factor)
	washingmachine_times, washingmachine_consumption = createWashingMachine(base_dir, days_calculated, offset,
	                                                                        discretization_factor)
	EVList = createEVList(base_dir, days_calculated, offset, discretization_factor)
	BatteryList = createBatteryList(base_dir)
	lst = (list(dishwasher_times.keys()) + list(washingmachine_times.keys()))
	max_devices = lst.count(max(set(lst), key=lst.count))
	lst = (list(dishwasher_times.values()) + list(washingmachine_times.values()))
	longest_interval = len(max(lst, key=len))
	lst = (list(washingmachine_consumption.values()) + list(dishwasher_consumption.values()))
	longest_consumption = len(max(*lst, key=len))

	houseList = list()
	for i in range(len(houseNames)):
		house = House_t("h%d" % i, max_devices, longest_interval, longest_consumption)
		if 'House%d' % i in washingmachine_times.keys():
			house.createTimeShiftables([washingmachine_times['House%d' % i]], washingmachine_consumption['House%d' % i])
		if 'House%d' % i in dishwasher_times.keys():
			house.createTimeShiftables([dishwasher_times['House%d' % i]], dishwasher_consumption['House%d' % i])
		if i in EVList.keys():
			house.createEv(EVList[i])
		if i in BatteryList.keys():
			house.createBattery(BatteryList[i])
		houseList.append(house)
	timeShiftables = [timeShiftable for house in houseList for timeShiftable in house.timeShiftables]
	for house in [h for h in houseList if h.ev]:
		house.ev.setMaxTSCosts(timeShiftables)
	write_houses(dest, houseList, houseNames, discretization_factor, summed_ep, BatteryList, EVList, max_devices,
	             longest_interval, longest_consumption)
	return Situation([sum([summed_ep[name][i] for name in houseNames]) for i in range(len(summed_ep.House0))], EVList,
	                 BatteryList, dishwasher_times, dishwasher_consumption, washingmachine_times,
	                 washingmachine_consumption)


if __name__ == "__main__":
	_base_dir = "/home/wowditi/Documents/Afstuderen/UppaalInput/alpg/results8/"
	_dest = "/home/wowditi/Documents/Afstuderen/Models/input-parsed2.txt"
	_discretization_factor = 15
	_offset_days = 180
	_offset = int(_offset_days * 24 * 60 / _discretization_factor)
	_days_processed = 4
	_days_calculated = 2
	createHouses(_base_dir, _dest, _days_processed, _days_calculated, _offset, _discretization_factor)
