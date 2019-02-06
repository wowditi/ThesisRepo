#!/usr/bin/python3    

	#Artifical load profile generator v1.1, generation of artificial load profiles to benchmark demand side management approaches
    #Copyright (C) 2018 Gerwin Hoogsteen

    #This program is free software: you can redistribute it and/or modify
    #it under the terms of the GNU General Public License as published by
    #the Free Software Foundation, either version 3 of the License, or
    #(at your option) any later version.

    #This program is distributed in the hope that it will be useful,
    #but WITHOUT ANY WARRANTY; without even the implied warranty of
    #MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #GNU General Public License for more details.

    #You should have received a copy of the GNU General Public License
    #along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    

    
#This is an example configuration file!

# Select the output writer
try:
	import writer as writer
except:
	pass

#Random seed
seed = 9001

#input files:
weather_irradiation = 'input/weather/solarirradiation_twenthe.csv'
weather_timebaseDataset = 3600 #in seconds per interval


#Simulation:
#number of days to simulate and skipping of initial days. Simulation starts at Sunday January 1.
numDays = 5		# number of days
startDay = 180			# Initial day
numHouses = 30


#Select the geographic location. Refer to the Astral plugin to see available locations (or give a lon+lat)
# Use e.g. https://www.latlong.net/
from astral import Location

location = Location()
location.solar_depression = 'civil'
location.latitude = 52.239095
location.longitude = 6.857018
location.timezone = 'Europe/Amsterdam'
location.elevation = 0

#Select the devices in the neighbourhood

#Devices
#Scale overall consumption:
consumptionFactor = 1.0 #consumption was a bit too high

# Penetration of emerging technology in percentages
# all values must be between 0-100
# These indicate what percentage of the houses has a certain device

# Electric mobility, restriction that the sum <= 100
# Note, households with larger driving distances will receive EVs first
penetrationEV 				= 26
penetrationPHEV 			= 32

# PV and storage, restriction that Battery <= PV
# Note PV and battery size depend on the annual household consumption
# This emulates the Dutch "nul-op-the-meter regime (net zero annual electricity usage)
penetrationPV				= 75
penetrationBattery 			= 20	#Note only houses with PV will receive a battery!

# Heating systems, with restriction that the sum <= 100
penetrationHeatPump 		= 25
penetrationCHP				= 5		# Combined heat and power

penetrationInductioncooking = 25


#Device parameters:
#EV
capacityEV = 	42000	#Wh
powerEV = 		9000	#W
capacityPHEV = 	12000	#Wh
powerPHEV = 	3000	#W

#PV
PVProductionPerYear = 	750		#average kWh per m2 solar panel on annual basis
PVAngleMean = 			35 		#degrees, 0 is horizontal to earth surface
PVAngleSigma = 			10		#degrees
PVAzimuthMean = 		180 	#degrees, 0 is north, 90 is east
PVAzimuthSigma = 		90 		#degrees
PVEfficiencyMin = 		15		#% of theoretical max
PVEfficiencyMax = 		20		#% of theoretical max

#Driving distances
commuteDistanceMean = 	45		#km
commuteDistanceSigma = 	20		#km


#Battery
capacityBatteryLarge = 	11000 	#Wh
capacityBatteryMedium = 6000  	#Wh
capacityBatterySmall = 	3000 	#Wh
powerBatteryLarge = 	2500 	#W
powerBatteryMedium = 	4500  	#W
powerBatterySmall = 	3000 	#W


#Kitchen
#Consumption of devices
ConsumptionOven = 				2500	#W
ConsumptionMicroWave = 			800		#W
ConsumptionStoveVentilation = 	130 	#W #But this is maximum, usually set lower!
ConsumptionInductionStove = 	2700 	#W #http://homeguides.sfgate.com/many-watts-induction-stove-85380.html

ConsumptionFridgeBigMin = 		50		#W
ConsumptionFridgeBigMax = 		100		#W
ConsumptionFridgeSmallMin = 	30		#W
ConsumptionFridgeSmallMax = 	80		#W

ConsumptionKettle = 			2600	#W

#White goods
ConsumptionIron = 				2200	#W
ConsumptionVacuumcleaner = 		1200	#W

#House
ConsumptionHouseVentilation = 	50 		#W


#Household randomization
#all values must be between 0-1000
familyOutingChanceMin = 			15 	#percentage
familyOutingChanceMax = 			25 	#percentage
personWeekdayActivityChanceMin = 	10 	#percentage
personWeekdayActivityChanceMax = 	20 	#percentage
personWeekendActivityChanceMin = 	25 	#percentage
personWeekendActivityChanceMax = 	45 	#percentage



householdList = []

#Select the types of households
try:
	import households

	for i in range(0,5):
		householdList.append(households.HouseholdSingleWorker())

	for i in range(0,1):
		householdList.append(households.HouseholdSingleRetired())
		
	for i in range(0,4):
		householdList.append(households.HouseholdDualWorker(True))
		
	for i in range(0,2):
		householdList.append(households.HouseholdDualWorker(False))
		
	for i in range(0,3):
		householdList.append(households.HouseholdDualRetired())	
		
	for i in range(0,6):
		householdList.append(households.HouseholdFamilyDualWorker(True))
		
	for i in range(0,9):
		householdList.append(households.HouseholdFamilyDualWorker(False))
except:
	pass
