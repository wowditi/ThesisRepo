from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass, astuple

import math
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import time
from lxml import etree
from typing import List, Sequence, Dict

from Grammar import *


@dataclass
class DataPoint:
	x: int
	y: int


@dataclass
class LearningParams:
	gruns: int
	truns: int
	rprstate: int
	eruns: int


@dataclass
class Component:
	template: str
	maxActive: int
	optimize: bool
	simulations: int
	learningParams: LearningParams


@dataclass
class SimPoint:
	time: int
	value: int

	def __repr__(self):
		return "(%d, %d)" % (self.time, self.value)


@dataclass
class Interval:
	start: SimPoint
	end: Optional[SimPoint]


@dataclass
class IntInterval:
	start: int
	end: int


class Timer:
	def __init__(self, tname):
		self.name = tname

	def __enter__(self):
		self.timer = time.time()

	def __exit__(self, exc_type, exc_val, exc_tb):
		print("%s took: " % self.name, time.time() - self.timer)


class Data(list):
	def __init__(self, data=None):
		if data is None:
			data = list()
		super(list, self).__init__()
		self.active_periods = list()
		for x in data:
			self.append(x)

	def get_cost(self):
		return self[-1].y

	def update_active_periods(self):
		i = 0
		intervals = list()
		while i < len(self):
			if self[i].y == 1:
				start = math.ceil(self[i].x)
				while i < len(self) and self[i].y == 1:
					i += 1
				intervals.append(IntInterval(int(start), math.floor(self[i].x) - 1))
			i += 1
		self.active_periods = intervals

	def get_costs(self, ctime, costs):
		for interval in self.active_periods:
			if interval.start <= ctime < interval.end:
				return costs[ctime - interval.start]
		return 0


class Transition:

	def __init__(self, transition: etree.ElementTree, parent: 'Template'):
		self.transition = transition
		self.parent = parent
		self.parentNode = self.parent.template
		self.active = True
		self.source = next(l for l in parent.locations if l.id == transition.find("source").get("ref"))
		self.target = next(l for l in parent.locations if l.id == transition.find("target").get("ref"))
		self.attributeDict = dict()
		for label in transition.findall("label"):
			self.attributeDict[label.get("kind")] = label.text

	def delete(self):
		self.parentNode.remove(self.transition)
		self.active = False

	def restore(self):
		self.parentNode.insert(-1, self.transition)
		self.active = True

	def getLabel(self, kind):
		return self.transition.find("label[@kind='%s']" % kind).text if self.transition.find(
			"label[@kind='%s']" % kind) is not None else None

	def setLabel(self, kind, value):
		if self.getLabel(kind) is not None:
			self.transition.find("label[@kind='%s']" % kind).text = value
		else:
			newTrans = etree.SubElement(self.transition, 'label')
			newTrans.set('kind', kind)
			newTrans.text = value

	@property
	def controllable(self):
		return self.transition.get("controllable")

	@controllable.setter
	def controllable(self, controllable):
		self.transition.set("controllable", "true" if controllable else "false")


class Location:

	def __init__(self, location: etree.ElementTree, parent: 'Template'):
		self.parent = parent  # type: Template
		self.location = location  # type: etree.Element
		self.parentNode = self.parent.template
		self.id = location.get("id")
		self.committed = location.find("committed") is not None
		self.oldInvariant = ""
		if location.find("name") is not None:
			self.name = location.find("name").text
		else:
			self.name = self.id
			nameElement = etree.Element('name')
			nameElement.text = self.name
			self.location.insert(0, nameElement)
		self.active = True

	@property
	def invariant(self):
		return self.location.find("label[@kind='invariant']").text if self.location.find(
			"label[@kind='invariant']") is not None else None

	@invariant.setter
	def invariant(self, invariant):
		self.oldInvariant = self.invariant
		self.location.find("label[@kind='invariant']").text = invariant

	@property
	def comment(self):
		return self.location.find("label[@kind='comments']").text if self.location.find(
			"label[@kind='comments']") is not None else None

	@comment.setter
	def comment(self, comment):
		self.location.find("label[@kind='comments']").text = comment

	def delete(self):
		self.parentNode.remove(self.location)
		[t.delete() for t in self.parent.transitions if t.source == self or t.target == self]
		self.active = False

	def restore(self):
		self.parentNode.insert(3, self.location)
		[t.restore() for t in self.parent.transitions if t.source == self or t.target == self]
		self.active = True


class Template:

	def __init__(self, template: etree.ElementTree, global_decl: Decl):
		self.template = template
		self.locations = [Location(location, self) for location in template.findall("location")]
		self.transitions = [Transition(transition, self) for transition in template.findall("transition")]
		self.global_decl = global_decl
		self.declaration = Decl(values=global_decl.values)
		self.executedTransitions = []
		if template.find("declaration") is not None:
			self.declaration.parse(template.find("declaration").text)
		self.systemName = self.name
		self.templateName = self.name
		self.usefull = True
		self.active = True
		self.StoredTemplate = copy.deepcopy(template)
		self.staticFuncs = dict(TimeShiftable=self.modifyTS, TimeShiftableR=self.revertTS, EV=self.modifyEV,
		                        EVR=self.revertEV, Battery=self.modifyBattery, BatteryR=self.revertBattery)
		self.neededVars = dict(TimeShiftable=self.tsVars, EV=self.evVars, Battery=self.batteryVars)
		self.dynamic = True
		self.previousVarValues = None
		self.startTime = None  # used to sort EVs

	@property
	def name(self):
		return self.template.find('name').text

	@name.setter
	def name(self, value):
		self.template.find('name').text = value

	@property
	def init(self):
		return next(l for l in self.locations if l.id == self.template.find('init').get('ref'))

	@init.setter
	def init(self, location: Location):
		self.template.find('init').set("ref", location.id)

	@property
	def parameters(self):
		if self.template.find('parameter') is not None:
			return self.declaration.parse(self.template.find('parameter').text, Parameters)
		return None

	@parameters.setter
	def parameters(self, parameters: Parameters):
		self.template.find('parameter').text = str(parameters)

	def insertConstant(self, ctype, cname, value):
		newConst = self.declaration.parse("const %s %s = %s;" % (ctype, cname, value), ConstantDef)
		self.declaration.declaration.insert(0, newConst)

	def getVars(self, const=False):
		result = [x for x in self.declaration.declaration if type(x) == VariableDef]
		if const:
			result = result + [x for x in self.declaration.declaration if type(x) == ConstantDef]
		return result

	def getLocation(self, lname: str):
		return next(l for l in self.locations if l.name == lname)

	def getTransition(self, source: Location, target: Location, active=True) -> Union[Transition, None]:
		return next((t for t in self.transitions if t.source == source and t.target == target and t.active == active),
		            None)

	def getTransitions(self, source: Location, target: Location, active=True):
		return [t for t in self.transitions if t.source == source and t.target == target and t.active == active]

	def getLocationNames(self):
		return [x.name for x in self.locations if not x.committed]

	def addTransition(self, source: Location, target: Location, controllable: bool = False, append=True) -> Transition:
		transition = etree.SubElement(self.template, "transition",
		                              attrib={'controllable': "true" if controllable else "false"})
		etree.SubElement(transition, 'source', attrib={'ref': source.id})
		etree.SubElement(transition, 'target', attrib={'ref': target.id})
		trans = Transition(transition, self)
		if append:
			self.transitions.append(trans)
		return trans

	def removeParameter(self, param):
		self.parameters = Parameters([p for p in self.parameters if p.name != param.name])

	def removeParameterByIndex(self, index):
		self.parameters = Parameters([p for idx, p in enumerate(self.parameters) if idx != index])

	def alterDecl(self):
		if self.template.find("declaration") is not None:
			self.template.find("declaration").text = str(self.declaration)

	def getStateVariables(self):
		if self.template.find("declaration") is None:
			return []
		stateList = []
		for i in [x for x in self.declaration.getVars() if x.type != "chan"]:
			if i.isList:
				length = i.length  # type: lengthDef
				tempList = ["%s[%d]" % (i.name, z) for z in range(self.declaration.evalExpression(length.value))]
				while length.hasNext:
					length = length.next
					tempList = ["%s[%d]" % (y, z) for z in range(self.declaration.evalExpression(length.value)) for y in
					            tempList]
				stateList += tempList
			else:
				stateList.append(i.name)
		return stateList

	def getNeededVars(self):
		return self.neededVars[self.templateName]()

	def makeStatic(self, varValues: Dict[str, Sequence[SimPoint]]):
		if self.dynamic:
			self.previousVarValues = varValues
			self.staticFuncs[self.templateName](varValues)
			self.dynamic = False

	def makeDynamic(self):
		if not self.dynamic:
			self.staticFuncs[self.templateName + "R"]()
			self.dynamic = True

	def revertDynamic(self):
		self.makeStatic(self.previousVarValues)

	def modifyTS(self, varValues: Dict[str, Sequence[SimPoint]]):
		startTimes = [val.time for val in varValues["%s.Activated" % self.systemName] if val.value]
		self.getLocation("Waiting").invariant = ""
		self.getLocation("Choice").delete()
		self.getTransition(self.getLocation("Waiting"), self.getLocation("Waiting")).delete()
		# self.getTransition(self.getLocation("Choice"), self.getLocation("Waiting")).delete()
		# self.getTransition(self.getLocation("Choice"), self.getLocation("Activated")).controllable = False
		# trans = self.getTransition(self.getLocation("Waiting"), self.getLocation("Choice"))
		# trans.delete()
		for startTime in startTimes:
			newTransition = self.addTransition(self.getLocation("Waiting"), self.getLocation("Activated"),
			                                   controllable=False, append=False)
			newTransition.setLabel('synchronisation', "tsStep?")
			newTransition.setLabel('guard', "time == %d" % startTime)
			newTransition.setLabel('assignment', 'startTime = intTime')

	# newTransition = copy.deepcopy(trans.transition)
	# newTransition.find("label[@kind='synchronisation']").text = "tsStep?"
	# newTransition.find("label[@kind='guard']").text = "time == %d" % startTime
	# self.locations[0].location.getparent().insert(-1, newTransition)

	def revertTS(self):
		self.getLocation("Waiting").invariant = self.getLocation("Waiting").oldInvariant
		[self.transitions[0].parentNode.remove(x) for x in
		 self.template.xpath("transition[source/@ref='%s']" % self.getLocation("Waiting").id)]
		self.getLocation("Choice").restore()
		self.getTransition(self.getLocation("Waiting"), self.getLocation("Waiting"), False).restore()

	# self.getTransition(self.getLocation("Choice"), self.getLocation("Waiting"), False).restore()
	# self.getTransition(self.getLocation("Choice"), self.getLocation("Activated")).controllable = True
	# self.getTransition(self.getLocation("Waiting"), self.getLocation("Choice"), False).restore()

	def modifyEV(self, varValues: Dict[str, Sequence[SimPoint]]):
		trans = self.getTransition(self.getLocation("Choice"), self.getLocation("Working"))
		fillLimits = varValues["fillLimits[%s]" % self.name.split("_", 1)[1]]
		limits = []
		nextVal = True
		for limit in fillLimits:
			if int(limit.value) == -2147480000:
				nextVal = True
			else:
				if nextVal:
					limits.append(limit)
					nextVal = False
		if not limits:
			self.getLocation("Choice").committed = False
		trans.delete()
		for limit in limits:
			newTrans = self.addTransition(self.getLocation("Choice"), self.getLocation("Working"),
			                              controllable=False, append=False)
			newTrans.setLabel("assignment", "fillLimit = %s, \n  interval++" % int(limit.value))
			newTrans.setLabel("guard", "time == %s" % limit.time)

	def revertEV(self):
		[self.transitions[0].parentNode.remove(x) for x in
		 self.template.xpath("transition[source/@ref='%s']" % self.getLocation("Choice").id)]
		self.getTransition(self.getLocation("Choice"), self.getLocation("Working"), active=False).restore()
		self.getLocation("Choice").committed = True

	def modifyBattery(self, varValues: Dict[str, Sequence[SimPoint]]):
		fillLimits = varValues["batteryFillLimit[%s]" % self.name.split("_", 1)[1]]
		nextVal = None
		limits = []  # type: List[Interval]
		for limit in fillLimits:
			if int(limit.value) == -2147480000:
				if nextVal:
					limits.append(Interval(nextVal, limit))
					nextVal = None
			else:
				if not nextVal:
					nextVal = limit
		if nextVal:
			limits.append(Interval(nextVal, None))
		self.getLocation("StartChoice").delete()
		self.getTransition(self.getLocation("Working"), self.getLocation("Inactive")).delete()
		for interval in [i for i in limits]:
			activateTrans = self.addTransition(self.getLocation("Inactive"), self.getLocation("Working"), append=False)
			activateTrans.setLabel("guard", "time == %s" % interval.start.time)
			activateTrans.setLabel("assignment", "batteryFillLimit=%s" % interval.start.value)
			activateTrans.setLabel("synchronisation", "batteryStep?")
			if interval.end:
				deactivateTrans = self.addTransition(self.getLocation("Working"), self.getLocation("Inactive"),
				                                     append=False)
				deactivateTrans.setLabel("guard", "time == %s" % interval.end.time)
				deactivateTrans.setLabel("assignment",
				                         "batteryFillLimit=-2147480000, batteryFillValue=0")
				deactivateTrans.setLabel("synchronisation", "tsStep?")

	def revertBattery(self):
		[self.transitions[0].parentNode.remove(x) for x in
		 self.template.xpath(
			 "transition[source/@ref='%s' and label/@kind='synchronisation' and label='tsStep?']" % self.getLocation(
				 "Working").id)]
		[self.transitions[0].parentNode.remove(x) for x in
		 self.template.xpath("transition[source/@ref='%s']" % self.getLocation("Inactive").id)]
		self.getLocation("StartChoice").restore()
		self.getTransition(self.getLocation("Working"), self.getLocation("Inactive"), active=False).restore()

	def tsVars(self):
		return ["%s.Activated" % self.systemName]

	def evVars(self):
		return ["fillLimits[%s]" % self.name.split("_", 1)[1]]

	def batteryVars(self):
		return ["batteryFillLimit[%s]" % self.name.split("_", 1)[1]]


class Model:

	def __init__(self, modelPath: str, newDeclaration: str = None):
		self.doc = etree.parse(modelPath)  # type: etree.ElementTree
		if newDeclaration is not None:
			with open(newDeclaration) as f:
				self.doc.find("declaration").text = "\n".join(f.readlines())
		self.declaration = Decl()
		self.declaration.parse(self.doc.find("declaration").text)
		self.templates = [Template(template, self.declaration) for template in self.doc.findall("template")]
		[TemplateG.addTemplate(t.name) for t in self.templates]
		self.system = self.declaration.parse(self.doc.find("system").text, System)  # System
		self.modelEndTime = 0

	def disable(self, template: Template):
		desc = next(s for s in self.system if type(s) == SystemDescription)
		if template.systemName in desc:
			desc.remove(template.systemName)
			template.active = False

	def enable(self, template):
		if not template.active:
			next(s for s in self.system if type(s) == SystemDescription).append(template.systemName)
			template.active = True

	def alterDecl(self):
		self.doc.find("declaration").text = str(self.declaration)

	def splitTemplate(self, template: Template, newNames, valueList, systemNames=None):
		index = self.templates.index(template) + 1
		for idx, nname in enumerate(newNames):
			newTemplate = Template(copy.deepcopy(template.template), template.global_decl)
			newTemplate.name = nname
			if systemNames:
				newTemplate.systemName = systemNames[idx]
			for values in valueList:
				for param in values[nname]:
					newTemplate.insertConstant(values[nname][param][0], param, values[nname][param][1])
			self.templates.append(newTemplate)
			self.doc.getroot().insert(index, newTemplate.template)
			index += 1
		self.templates.remove(template)
		self.doc.getroot().remove(template.template)

	def expand(self):
		newSystemList = []
		newTemplateDeclarations = []
		baseList = []
		for templateName in next(s for s in self.system if type(s) == SystemDescription):
			tname = templateName
			base = next((t for t in self.templates if t.name == tname), None)
			templates = []
			while base is None:
				decl = next((s for s in self.system if type(s) == TemplateDeclaration and s.name == tname), None)
				if decl:
					templates.append(decl)
					tname = decl.base
				else:
					base = next((t for t in self.templates if t.name == tname), None)
			if not templates:
				if not base.parameters:
					baseList.append(base.name)
					newSystemList.append(templateName)
					continue
				stack = [templateName]
				for param in base.parameters:
					pType = self.declaration.getType(param.type)
					minValue = self.declaration.evalExpression(pType.start)
					maxValue = self.declaration.evalExpression(pType.end)
					newStack = []
					for entry in stack:
						for i in range(minValue, maxValue + 1):
							newStack.append("%s_%d" % (entry, i))
					stack = newStack
				valuesStack = [
					{t: {base.parameters[p - 1].name: (base.parameters[p - 1].type, t.split("_")[-p]) for p in
					     range(1, len(base.parameters) + 1)} for t in stack}]
				[base.removeParameter(param) for param in base.parameters]
				self.splitTemplate(base, stack, valuesStack)
				for x in stack:
					baseList.append(x)
					newSystemList.append(TemplateG(x))
			else:
				if not templates[0].params:
					baseList.append(base.name)
					base.systemName = templateName
					newSystemList.append(templateName)
					continue
				stack = [""]
				for param in templates[0].params:
					pType = self.declaration.getType(param.type)
					minValue = self.declaration.evalExpression(pType.type.min)
					maxValue = self.declaration.evalExpression(pType.type.max)
					# print(param, minValue, maxValue)
					newStack = []
					for entry in stack:
						for i in range(minValue, maxValue + 1):
							newStack.append("%s_%d" % (entry, i))
					stack = newStack
				self.splitTemplate(base, [base.name + x for x in stack], [],
				                   [templates[0].name + x + "_T" for x in stack])
				for x in [base.name + x for x in stack]:
					baseList.append(x)
				for idx, template in enumerate(templates):
					templateName = template.name
					baseName = template.base
					for x in stack:
						template.name = templateName + x
						template.base = TemplateG(str(baseName) + x)
						newTemplateDeclarations.append(copy.deepcopy(template))
						TemplateG.addTemplate(template.name)
						if not idx:
							newTemplateDeclarations.append(self.declaration.parse(
								"%s%s_T = %s%s(%s);" % (templateName, x, templateName, x, ",".join(x.split("_")[1:])),
								TemplateDeclaration))
							newSystemList.append(TemplateG("%s%s_T" % (templateName, x)))
					template.name = templateName
					template.base = baseName
					self.system.remove(template)
		self.system[next(idx for idx, s in enumerate(self.system) if type(s) == SystemDescription)] = SystemDescription(
			newSystemList)
		for decl in newTemplateDeclarations:
			self.system.insert(-1, decl)
		self.doc.find("system").text = str(self.system)
		for x in [t for t in self.templates if t.name not in baseList]:
			self.templates.remove(x)

	# self.doc.getroot().remove(x.template)

	def getStateVariables(self):
		stateList = []
		for i in [x for x in self.declaration.getVars() if x.type != "chan"]:
			if i.isList:
				length = i.length  # type: lengthDef
				tempList = ["%s[%d]" % (i.name, z) for z in range(self.declaration.evalExpression(length.value))]
				while length.hasNext:
					length = length.next
					tempList = ["%s[%d]" % (y, z) for z in range(self.declaration.evalExpression(length.value)) for y in
					            tempList]
				stateList += tempList
			else:
				stateList.append(i.name)
		for t in self.templates:
			if t.active:
				stateList += ["%s.%s" % (t.systemName, state) for state in t.getStateVariables()]
		return stateList

	def getLocationNames(self):
		return ["%s.%s" % (template.systemName, location.name) for template in self.templates if template.active for
		        location in template.locations if location.active]

	def learningQuery(self, end_time):
		return "strategy Min = minE(cost) [time<=%s+1] : <> time == %s" % (
			end_time, end_time if end_time < self.modelEndTime else "%s && Time.Finished" % end_time)

	def simulateQuery(self, num, end_time, optimize, variables):
		return "simulate %s [time<=%s] {%s} %s" % (num, end_time if end_time < self.modelEndTime else end_time + 1,
		                                           ",".join(variables) if variables else ",".join(
			                                           self.getStateVariables() + self.getLocationNames()),
		                                           "under Min" if optimize else "")

	def generateQuery(self, simulations, learning_end_time, variables, optimize=False):
		return (self.learningQuery(learning_end_time) + "\n" if optimize else "") + \
		       self.simulateQuery(simulations, learning_end_time, optimize, variables)

	def updateValues(self, varValues, end_time):
		# for var, values in varValues.items():
		#	if var in results:
		#		results[var] += values
		for var in varValues:
			splitVar = var.split('.', 1)
			if len(splitVar) == 2:
				template = next(t for t in self.templates if t.systemName == splitVar[0])
				loc = next((l for l in template.locations if
				            l.name == splitVar[1]), None)  # type: Location
				if loc:
					if varValues[var][-1].value == 1:
						template.init = loc
				else:
					next(t for t in self.templates if t.systemName == splitVar[0]).declaration.setVar(splitVar[1],
					                                                                                  varValues[var][
						                                                                                  -1].value)
			else:
				if var in ['time', 'intTime']:
					self.declaration.setVar(var, end_time)
				elif var == 'Time.c':
					self.declaration.setVar(var, 0)
				else:
					try:
						self.declaration.setVar(var, varValues[var][-1].value)
					except Exception as e:
						print(var, varValues[var])
						raise e

	def save(self, location):
		self.doc.find("system").text = str(self.system)
		self.alterDecl()
		for template in self.templates:
			template.alterDecl()
		# template.getLocation()
		# self.doc.find("template[name='TSController_0_0']/location/label[@kind='invariant']").text = ""
		# print(self.doc.find("template[name='TSController_0_0']/location/label[@kind='invariant']").text)
		self.doc.write(location)

def evSortHelper(template: Template):
	return template.startTime


class StrategoAlgorithm:
	def __init__(self, resultFile: str, queryFile: str, modelFile: str, verifyta: str, model: Model,
	             requiredTemplates: List[str], components: Sequence[Component], endTime: int, stepSize: int = 96,
	             learningTime: int = 192, numSimulations: int = 25, optimize: bool = True, iterations: int = 2,
	             seed: int = None):
		self.seed = seed
		if self.seed is None:
			from datetime import datetime
			self.seed = int(datetime.utcnow().timestamp())
		self.resultFile = resultFile
		self.verifyta = verifyta
		self.queryFile = queryFile
		self.model = model
		self.gruns = "20"
		self.truns = "20"
		self.rprstate = "10"
		self.eruns = "10"
		self.endTime = endTime
		self.stepSize = stepSize
		self.learningTime = learningTime
		self.numSimulations = numSimulations
		self.iterations = iterations
		self.optimization = optimize
		self.requiredTemplates = requiredTemplates
		self.components: Sequence[Component] = components
		self.modelFile = modelFile
		model.modelEndTime = model.declaration.evalExpression(model.declaration.getConst("end_time").value)
		self.model.expand()
		self.model.save(self.modelFile)
		self.createQuery(1, False, 1, end_time=1)
		with Timer("Running the model"):
			self.runVerifyTa()
		result, cost = self.parseSimulationFile(resultFile, 1, 3, 0, split=True)
		model.updateValues(result, 0)
		for template in model.templates:
			model.disable(template)
			if template.templateName == "TimeShiftable" and template.init.name == "Finished":
				template.usefull = False
			if template.templateName == "EV" and template.init.name == "Inactive":
				template.usefull = False
			if template.templateName == "Battery" and template.init.name == "Unavailable":
				template.usefull = False
		for template in [t for t in model.templates if t.templateName in requiredTemplates]:
			model.enable(template)
		model.save(self.modelFile)

	def runVerifyTa(self, gruns=0, truns=0, rprstate=0, eruns=0):
		with open(self.resultFile, "w+") as f:
			p = subprocess.Popen(" ".join(
				[self.verifyta, "-q", '--learning-method', '2', '--filter', '2', '--good-runs', str(gruns),
				 '--total-runs', str(truns),
				 '--runs-pr-state',
				 str(rprstate),
				 '--eval-runs', str(eruns), '-R', str(self.endTime + 1 + self.stepSize), self.modelFile,
				 self.queryFile]), shell=True, universal_newlines=True,
				stdout=f)
			p.wait()

	def createQuery(self, ctime, optimize, simulations, variables=None, end_time=None):
		with open(self.queryFile, 'w+') as f:
			f.write(self.model.generateQuery(simulations,
			                                 end_time if end_time else min(ctime + self.learningTime - self.stepSize,
			                                                               self.model.modelEndTime),
			                                 variables,
			                                 optimize))

	@staticmethod
	def parseSimulationFile(file: str, simulations: int, skipLines: int, end_time: int, split: bool = False) -> (
			Dict[str, Sequence[SimPoint]], int):
		varValuesFull: Dict[str, Sequence[SimPoint]] = dict()
		costs = None
		with open(file) as f:
			minIndex = 0
			lines = f.readlines()[skipLines:]
			for i in range(0, len(lines), simulations + 1):
				var = lines[i][:-2]
				if var == "cost":
					costs = [eval(lines[h + i].rsplit(" ", 1)[1])[1] for h in range(1, simulations + 1)]
					minIndex = costs.index(min(costs)) + 1
					print("minCost:", costs[minIndex - 1])
					break
			for i in range(0, len(lines), simulations + 1):
				var = lines[i][:-2]
				varValuesFull[var] = [SimPoint(*eval(x)) for x in (lines[i + minIndex].split(" ")[1:])]
		if not split:
			return varValuesFull, costs[minIndex - 1]
		# Variables increasing at a steady rate of increase can have an incorrect value atm, however this only effects the cost
		# varValues: Dict[str, Sequence[SimPoint]] = dict()
		# for key, values in varValuesFull.items():
		#	beforeSplit = [v for v in values if v.time <= end_time]
		#	if beforeSplit[-1].time != end_time:
		#		nextVal = values[len(beforeSplit)]
		#		beforeSplit.append(SimPoint(end_time, int(
		#			(nextVal.value - beforeSplit[-1].value) / (nextVal.time - beforeSplit[-1].time) * (
		#						end_time - beforeSplit[-1].time))))
		#		varValues[key] = beforeSplit
		varValues: Dict[str, Sequence[SimPoint]] = {key: [v for v in values if v.time <= end_time] for key, values in
		                                            varValuesFull.items()}
		return varValues, costs[minIndex - 1]

	@staticmethod
	def transformResult(result):
		for key in result:
			result[key] = Data([DataPoint(x, y) for x, y in result[key]])
		return result

	def printResults(self, result):
		simulation = self.transformResult(result)
		idx = 0
		HouseBatteryCosts = []
		HouseDynamicCosts = []
		for i in range(192):
			while idx < len(simulation['HouseBatteryCosts']) and int(
					simulation['HouseBatteryCosts'][idx].x) < i + 1:
				idx += 1
			if len(simulation['HouseBatteryCosts']) > 0:
				HouseBatteryCosts.append(
					int(simulation['HouseBatteryCosts'][idx - 1].y))
		idx = 0
		for i in range(192):
			while idx < len(simulation['HouseDynamicCosts']) and int(
					simulation['HouseDynamicCosts'][idx].x) < i + 1:
				idx += 1
			HouseDynamicCosts.append(
				int(simulation['HouseDynamicCosts'][idx - 1].y))
		print("BatteryCosts:", sum(HouseBatteryCosts))
		print("DynamicCosts:", sum(HouseDynamicCosts))
		base_dir = "/home/wowditi/Documents/Afstuderen/UppaalInput/alpg/results8/"
		discretization_factor = 15
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
		discretized_list = [0] * int(len(summed_ep.index) / discretization_factor)
		for i in range(int(len(summed_ep.index) / discretization_factor)):
			discretized_list[i] = summed_ep[i * discretization_factor:(i + 1) *
			                                                          discretization_factor].sum()
		discretized_ep = pd.DataFrame(data=discretized_list, columns=summed_ep.columns)
		summed_ep = discretized_ep
		combined_ep = [(sum([summed_ep[hname][i] for hname in houseNames]))
		               for i in range(len(summed_ep.House0))]
		# result = [(int((combined_ep[idx] + HouseDynamicCosts[idx] + HouseBatteryCosts[idx]) / 900)) ** 2
		#          for idx in range(len(HouseDynamicCosts))]
		plt.figure(figsize=(20, 5))
		plt.bar(
			range(len(HouseDynamicCosts)), [(int(
				(combined_ep[idx] + HouseDynamicCosts[idx] + HouseBatteryCosts[idx])))
				for idx in range(len(HouseDynamicCosts))],
			linewidth=6.0,
			color='limegreen',
			label="TimeShiftable Costs")

	# plt.savefig("%s/StrategoOnline/test.png" % self.base)
	@staticmethod
	def findTSMistake(result: Dict[str, Sequence[SimPoint]]):
		if [x for i, x in enumerate(result['tsStartTimes[0][0]']) if
		    x.value != -1 and x.time != x.value and result['tsStartTimes[0][0]'][i - 1].value == -1]:
			exit()

	@staticmethod
	def UppaalListToList(data: Sequence[SimPoint]):
		base = data[0].time
		result: List[float] = [0.0] * int(data[-1].time + 1 - data[0].time)
		prev = 0
		prevIdx = base
		for point in data:
			while prevIdx < point.time - 1:
				prevIdx += 1
				result[prevIdx - base] = prev
			result[point.time - base] = float(point.value)
			prev = point.value
			prevIdx = point.time
		return result

	def findEVMistake(self, result: Dict[str, Sequence[SimPoint]]):
		fillValues = self.UppaalListToList(result['fillValues[0]'])
		charges = self.UppaalListToList(result['charges[0]'])
		if int(charges[0]) == 0 and [(i, x, charges[i], fillValues[i - 1]) for i, x in enumerate(charges) if
		                             int(x) != 0 and int(charges[i - 1]) == 0 and fillValues[i - 1] != x]:
			exit()

	def runStratego(self, output: str) -> Union[None, Dict[str, Sequence[SimPoint]]]:
		finalResult: Union[None, Dict[str, Sequence[SimPoint]]] = None
		with Timer("Outer loop"):
			for i in list(range(self.stepSize, self.endTime, self.stepSize)) + (
					[] if self.endTime - 1 % self.stepSize == 0 else [self.endTime]):
				print(i)
				minCost = 0
				with Timer("Iterations loop"):
					for iteration in range(self.iterations):
						print(iteration)
						with Timer("Components loop"):
							for component in self.components:
								templates = [t for t in self.model.templates if
								             t.templateName == component.template and t.usefull]
								if not templates:
									continue
								if component.template == "EV":
									currentIntervals = self.model.declaration.evalExpression(
										self.model.declaration.getVar('evIntervals').value)
									for template in templates:
										evId = int(template.name.split("_")[1])
										template.startTime = self.model.declaration.evalExpression(
											self.model.declaration.getConst(
												f'h{evId}evlist').value)[1][currentIntervals[evId]][0]
									templates.sort(key=evSortHelper)
								numComp = math.ceil(
									len(templates) * 1.0 / math.ceil(len(templates) * 1.0 / component.maxActive))
								for j in range(0, len(templates), numComp):
									neededVars = ["cost"]
									for template in templates[j:j + numComp]:
										template.makeDynamic()
										self.model.enable(template)
										neededVars += template.getNeededVars()
										self.model.save(self.modelFile)
									self.createQuery(i, self.optimization and component.optimize, component.simulations,
									                 variables=neededVars)
									#if i==288 and component.template=='EV':
									#	print("DEBUG")
									#    exit()
									with Timer("Running the %s model" % component.template):
										self.runVerifyTa(*astuple(component.learningParams))
									result, cost = self.parseSimulationFile(self.resultFile, component.simulations,
									                                        12 if self.optimization and component.optimize else 3,
									                                        i)
									# self.findEVMistake(result)
									if iteration > 0 and cost > minCost:
										print("reverting")
										for template in templates[j:j + numComp]:
											template.revertDynamic()
									else:
										minCost = cost
										for template in templates[j:j + numComp]:
											template.makeStatic(result)
									self.model.save(self.modelFile)
				self.createQuery(i, False, 1)
				with Timer("Running the model"):
					self.runVerifyTa()
				result, _ = self.parseSimulationFile(self.resultFile, 1, 3, i, split=True)
				if not finalResult:
					finalResult = {key: [val for val in value if val.time <= i] for key, value in result.items()}
				else:
					finalResult = {key: list(finalResult[key]) + [val for val in value if val.time <= i] for key, value
					               in result.items()}
				self.model.updateValues(result, i)
				self.model.declaration.setVar("cost", 0.0)
				for template in self.model.templates:
					self.model.disable(template)
				for template in [t for t in self.model.templates if t.templateName in self.requiredTemplates]:
					self.model.enable(template)
				self.model.save(self.modelFile)
		print("result:", [p for p in finalResult['cost'] if p.time <= self.endTime][-1])
		with open(output, 'wb') as file:
			pickle.dump(finalResult, file, pickle.HIGHEST_PROTOCOL)
		return finalResult


if __name__ == "__main__":
	if os.name == 'nt':
		_base = 'C:/Users/niels/Documents/Afstuderen'
		_verifyta = "%s/uppaal-stratego-4.1.20/bin-Win32/verifyta.exe" % _base
	else:
		_base = '/home/wowditi/Documents/Afstuderen'
		_verifyta = "%s/uppaal-stratego-4.1.20/bin-Linux/verifyta" % _base
	_baseL = '%s/Models/OnlineTSv7.xml' % _base
	_queryL = '%s/StrategoOnline/Test.q' % _base
	_modelL = '%s/StrategoOnline/Test.xml' % _base
	_resultL = '%s/StrategoOnline/Test.txt' % _base
	_requiredTemplates: List[str] = ['Time', 'CostCalculator', 'MainLoop']
	_components: List[Component] = [Component("TimeShiftable", 2, LearningParams(2, 2, 2, 2)),
	                                Component("EV", 2, LearningParams(2, 2, 2, 2)),
	                                Component("Battery", 1, LearningParams(20, 20, 10, 10))]
	alg = StrategoAlgorithm(_resultL, _queryL, _modelL, _verifyta, Model(_baseL), _requiredTemplates, _components,
	                        endTime=192, optimize=True)
	alg.runStratego('%s/StrategoOnline/result' % _base)
