from typing import Union, Optional

from pypeg2 import *

double = re.compile(r"\d+\.\d+")
listLength = re.compile(r"\[\d+\]")

Var = namedtuple('Variable', ["const", "type", "name"])


def insertTab(myStr):
	return '\t'.join(myStr.splitlines(True))


class TemplateG(Keyword):
	grammar = Enum()

	@staticmethod
	def addTemplate(template: str):
		TemplateG.grammar.data[Symbol(template)] = Keyword(template)

	@staticmethod
	def reset():
		TemplateG.grammar = Enum()

	@staticmethod
	def assign(grammar):
		TemplateG.grammar = grammar


class TemplateList:
	templates = []

	@staticmethod
	def addTemplate(template):
		TemplateList.templates.append(template)

	@staticmethod
	def reset():
		TemplateList.templates = []

	@staticmethod
	def assign(templates):
		TemplateList.templates = templates


class TypeList:
	types = [double, int, str]

	@staticmethod
	def addType(typeN, pos=None):
		if pos:
			TypeList.types.insert(0, typeN)
		else:
			TypeList.types.append(typeN)

	@staticmethod
	def reset():
		TypeList.types = [double, int, str]

	@staticmethod
	def assign(types):
		TypeList.types = types


class VariableList(Keyword):
	grammar = Enum()
	variables = []
	types = []

	@staticmethod
	def addVariable(variable):
		VariableList.variables.append(variable.name)
		VariableList.types.append(variable.type)
		VariableList.grammar.data[Symbol(variable.name)] = Keyword(variable.name)

	@staticmethod
	def reset():
		VariableList.variables = []
		VariableList.types = []
		VariableList.grammar = Enum()

	@staticmethod
	def assign(variables):
		VariableList.variables = [v.name for v in variables]
		VariableList.types = [v.type for v in variables]
		for var in VariableList.variables:
			VariableList.grammar.data[Symbol(var)] = Keyword(var)

	@staticmethod
	def getVars():
		return [Var(False, VariableList.types[i], VariableList.variables[i]) for i in range(len(VariableList.types))]


class ConstantList(Keyword):
	grammar = Enum()
	constants = []
	types = []

	@staticmethod
	def addConstant(constant):
		ConstantList.constants.append(constant.name)
		ConstantList.types.append(constant.type)
		ConstantList.grammar.data[Symbol(constant.name)] = Keyword(constant.name)

	@staticmethod
	def reset():
		ConstantList.constants = []
		ConstantList.types = []
		ConstantList.grammar = Enum()

	@staticmethod
	def assign(constants):
		ConstantList.constants = [c.name for c in constants]
		ConstantList.types = [c.type for c in constants]
		for var in ConstantList.constants:
			ConstantList.grammar.data[Symbol(var)] = Keyword(var)

	@staticmethod
	def getVars():
		return [Var(True, ConstantList.types[i], ConstantList.constants[i]) for i in range(len(ConstantList.types))]


class Type(Keyword):
	grammar = Enum(K("int"), K('double'), K('clock'), K('chan'), K('bool'))

	@staticmethod
	def addType(typeN: str):
		Type.grammar.data[Symbol(typeN)] = Keyword(typeN)

	@staticmethod
	def reset():
		Type.grammar = Enum(K("int"), K('double'), K('clock'), K('chan'), K('bool'))

	@staticmethod
	def assign(grammar):
		Type.grammar = grammar


operator = re.compile(r"!=|==|\*|/|-|\+|<\?|>\?|>=|<=|&&|\|\||>|<")


class OneComment:
	grammar = "//", attr("comment", re.compile(r".*"))

	def __repr__(self):
		return "//%s" % self.comment


class BlockComment:
	grammar = "/*", attr("comment", re.compile(r".*?(?=\*/)", flags=re.DOTALL)), "*/"

	def __repr__(self):
		return "/*%s*/" % self.comment


class Comment:
	grammar = attr("comment", [BlockComment, OneComment])

	def __repr__(self):
		return str(self.comment)


class Expression:
	grammar = None

	@property
	def hasRhs(self):
		return hasattr(self, "op")

	def __repr__(self):
		if self.hasRhs:
			return "%s %s %s" % (self.lhs, self.op, self.rhs)
		else:
			try:
				return "%s" % self.lhs
			except Exception as e:
				print(type(self.lhs))
				raise e


class ParExpression:
	grammar = "(", attr("exp", Expression), ")"

	def __repr__(self):
		return "(%s)" % self.exp


class lengthDef:
	value: Expression
	next: Optional["lengthDef"]
	grammar = None

	@property
	def hasNext(self):
		return hasattr(self, "next")

	def __repr__(self):
		return "[%s]%s" % (self.value, self.next if self.hasNext else "")


lengthDef.grammar = "[", attr("value", Expression), "]", optional(attr("next", lengthDef))


class Sum:
	grammar = "sum(", attr("name", str), ":", attr("type", Type), ")", attr("exp", ParExpression)

	def __repr__(self):
		return "sum(%s:%s)(%s)" % (self.name, self.type, self.exp)


class DotExpr:
	grammar = None

	@property
	def isArray(self):
		return hasattr(self, "idx")

	@property
	def hasNext(self):
		return hasattr(self, "next")

	def __repr__(self):
		return ".%s%s%s" % (self.value, self.idx if self.isArray else "",
		                    self.next if self.hasNext else "")


DotExpr.grammar = ".", attr("value", word), optional(attr("idx", lengthDef)), optional(attr("next", DotExpr))

minF = flag("neg", "-")
optEnd = optional(attr("idx", lengthDef)), optional(attr("next", DotExpr))


class Constant:
	grammar = minF, attr("value", ConstantList), optEnd

	@property
	def isArray(self):
		return hasattr(self, "idx")

	@property
	def hasNext(self):
		return hasattr(self, "next")

	def __repr__(self):
		return "%s%s%s%s" % ("-" if self.neg else "", self.value, self.idx if self.isArray else "",
		                     self.next if self.hasNext else "")


class Variable:
	grammar = minF, attr("value", VariableList), optEnd

	@property
	def isArray(self):
		return hasattr(self, "idx")

	@property
	def hasNext(self):
		return hasattr(self, "next")

	def __repr__(self):
		return "%s%s%s%s" % ("-" if self.neg else "", self.value, self.idx if self.isArray else "",
		                     self.next if self.hasNext else "")


class Value:
	grammar = minF, attr("value", [TypeList.types]), optEnd

	@property
	def isArray(self):
		return hasattr(self, "idx")

	@property
	def hasNext(self):
		return hasattr(self, "next")

	def __repr__(self):
		return "%s%s%s%s" % ("-" if self.neg else "", self.value, self.idx if self.isArray else "",
		                     self.next if self.hasNext else "")


class FunctionCall(List):
	grammar = attr("func", word), "(", optional(csl(Expression)), ")"

	def __repr__(self):
		return "%s(%s)" % (self.func, ",".join([str(x) for x in self]))


class SpecialExpression:
	grammar = attr("lhs", [Sum, FunctionCall, ParExpression, Constant, Variable, Value]), optional(
		attr("op", operator), attr("rhs", [Expression]))

	@property
	def hasRhs(self):
		return hasattr(self, "op")

	def __repr__(self):
		if self.hasRhs:
			return "%s %s %s" % (self.lhs, self.op, self.rhs)
		else:
			return "%s" % self.lhs


class InlineIf:
	grammar = attr("cond", SpecialExpression), "?", attr("true", Expression), ":", attr("false", Expression)

	def __repr__(self):
		return "%s ? %s : %s" % (self.cond, self.true, self.false)


class customInt:
	grammar = "int[", attr("min", Expression), ",", attr("max", Expression), "]"

	def __repr__(self):
		return "int[%s,%s]" % (self.min, self.max)


TypeList.addType(customInt)


class uList(List):
	grammar = None

	def __repr__(self):
		return "{%s}" % str(list(self))[1:-1]

	def getValues(self):
		return list(self)

	#def initChildLists(self, lengths):
	#	print("not yet implemented")
		#for x in enumerate(self):
		#	self[x] = uList([Expression()]*lengths[0])
		#	if len(lengths) > 1:
		#		self[x].lhs.initChildLists(lengths[1:])

	def setValue(self, value, indexList):
		if len(indexList) == 1:
			self[indexList[0]] = value
		else:
			self[indexList[0]].lhs.setValue(value, indexList[1:])



uList.grammar = '{', csl([Expression, uList]), '}'
TypeList.addType(uList, 0)

Expression.grammar = attr("lhs", [uList, Sum, FunctionCall, ParExpression, InlineIf, Constant, Variable, Value]), optional(
	attr("op", operator), attr("rhs", [Expression]))


class VariableDef:
	length: lengthDef
	value: Expression
	grammar = optional(attr("meta", "meta")), optional(attr("special", re.compile("broadcast|hybrid"))), [customInt, Type], str, optional(
		attr("length", lengthDef)), optional(
		"=", attr("value", Expression)), ";"

	def __init__(self, parser):
		self.type = parser[0]
		self.name = parser[1]
		VariableList.addVariable(self)

	def getVar(self):
		return Var(False, self.type, self.name)

	def getValue(self):
		try:
			return self.value
		except:
			return None

	def initialize(self, listLengths):
		newList = uList
		#for i in listLengths:
		self.value = Expression

	def setValue(self, value, indexList=None):
		if self.isList:
			self.value.lhs.setValue(value, indexList)
		else:
			self.value = value

	@property
	def isSpecial(self):
		return hasattr(self, "special")

	@property
	def isMeta(self):
		return hasattr(self, "meta")

	@property
	def isList(self):
		return hasattr(self, 'length')

	@property
	def hasValue(self):
		return hasattr(self, "value")

	def __repr__(self):
		return f'{"meta " if self.isMeta else ""}{self.special+" " if self.isSpecial else ""}{self.type} {self.name}{self.length if self.isList else ""}{" = "+str(self.value) if self.hasValue else ""};'
		#return "%s %s%s%s;" % ("%s %s" % (self.special, self.type) if self.isSpecial else self.type, self.name,
		#                       self.length if self.isList else "", " = %s" % self.value if self.hasValue else "")


class ConstantDef:
	length: lengthDef
	value: Expression
	grammar = "const", VariableDef.grammar

	def __init__(self, parser):
		self.type = parser[0]
		self.name = parser[1]
		ConstantList.addConstant(self)

	def getVar(self):
		return Var(True, self.type, self.name)

	def getValue(self):
		try:
			return self.value
		except:
			return None

	@property
	def isList(self):
		return hasattr(self, 'length')

	@property
	def hasValue(self):
		return hasattr(self, "value")

	def __repr__(self):
		return "const %s %s%s%s;" % (self.type, self.name, self.length if self.isList else "",
		                             " = %s" % self.value if self.hasValue else "")


class TypeDef:
	grammar = "typedef", customInt, word, ";"

	@property
	def hasValue(self):
		return True

	def __init__(self, parser):
		self.name = parser[1]
		Type.addType(parser[1])
		TypeList.addType(parser[1])
		self.type = parser[0]
		ConstantList.addConstant(self)
		valueExpr = parse("1+"+ str(self.type.max), Expression)
		self.value = valueExpr

	def __repr__(self):
		return "typedef int[%s,%s] %s;" % (self.type.min, self.type.max, self.name)


class StructVariables(List):
	grammar = csl([ConstantDef, VariableDef], separator=endl)


class StructDef:
	grammar = "typedef", "struct", "{", attr("var", StructVariables), "}", word, ";"

	def __init__(self, parser):
		# print(parser)
		self.name = parser
		Type.addType(parser)
		TypeList.addType(parser)

	def __repr__(self):
		return """typedef struct {
	%s
} %s;""" % (insertTab("\n".join([str(x) for x in self.var])), self.name)


class Parameter:
	grammar = flag("const", "const"), optional(attr("special", re.compile("broadcast|hybrid"))), \
	          attr("type", [customInt, Type]), flag("reference", "&"), name(), optional(attr("length", lengthDef))

	@property
	def isSpecial(self):
		return hasattr(self, "special")

	@property
	def isArray(self):
		return hasattr(self, "length")

	def __repr__(self):
		return "%s%s %s" % ("const " if self.const else "", self.type if not self.isSpecial else "%s %s" % (self.special, self.type), self.name)


class Parameters(List):
	grammar = optional(csl(Parameter))

	def __repr__(self):
		return ",".join([str(x) for x  in self])


class Instructions(List):
	grammar = None

	def __repr__(self):
		return insertTab("\n".join([str(x) for x in self]))


class Else:
	grammar = "else {", attr("instr", Instructions), "}"

	def __repr__(self):
		return """else {
	%s	
}""" % self.instr


class If:
	grammar = "if (", attr("condition", Expression), ")", "{", \
	          attr("instr", Instructions), "}", optional(attr("elseO", Else))

	def __repr__(self):
		try:
			self.elseO
		except:
			self.elseO = ""
		return """if (%s) {
	%s
} %s""" % (self.condition, self.instr, self.elseO)


class While:
	grammar = "while (", attr("condition", Expression), ")", "{", attr("instr", Instructions), "}"

	def __repr__(self):
		return """while (%s) {
	%s		
}""" % (self.condition, self.instr)


class Assignment:
	grammar = attr("lhs", [Variable, Constant, Value]), optional(attr("op", re.compile(r"\+|-|/|\*"))), "=", attr("rhs",
	                                                                                                              Expression), ';'

	@property
	def hasAdditionalOp(self):
		return hasattr(self, "op")

	def __repr__(self):
		return "%s %s= %s;" % (self.lhs, self.op if self.hasAdditionalOp else "", self.rhs)


class Inc:
	grammar = attr("var", Value), "++"

	def __repr__(self):
		return "%s++" % self.var


class Increment:
	grammar = Inc.grammar, ";"

	def __repr__(self):
		return "%s++;" % self.var


class Dec:
	grammar = attr("var", Value), "--"

	def __repr__(self):
		return "%s--" % self.var


class ForSimple:
	grammar = "for", "(", attr("name", word), ":", attr("type", Type), ")", "{", attr("instr", Instructions), "}"

	def __repr__(self):
		return """for (%s:%s) {
	%s		
}""" % (self.name, self.type, self.instr)


class ForComplex:
	grammar = "for", "(", attr("assign", Assignment), \
	          attr("condition", Expression), ";", \
	          attr("action", [Dec, Inc, Expression]), ")", "{", attr("instr", Instructions), "}"

	def __repr__(self):
		return """for (%s%s;%s) {
	%s		
}""" % (self.assign, self.condition, self.action, self.instr)


class Return:
	grammar = "return", attr("exp", Expression), ';'

	def __repr__(self):
		return "return %s;" % self.exp


class Instruction:
	grammar = attr("instr", [Increment, Return, Assignment, VariableDef, TypeDef, StructDef, ConstantDef])

	def __repr__(self):
		return str(self.instr)


Instructions.grammar = maybe_some([BlockComment, Comment, If, Else, While, ForSimple, ForComplex, Instruction])


class Void:
	grammar = "void"

	def __repr__(self):
		return "void"


class Function:
	grammar = attr("type", [Void, customInt, Type]), name(), "(", attr("params", Parameters), ")", '{', attr(
		"instr", Instructions), '}'

	def __repr__(self):
		return """%s %s(%s) { 
	%s
}""" % (self.type, self.name, self.params, self.instr)


class Declaration(List):
	grammar = maybe_some([Comment, TypeDef, StructDef, ConstantDef, VariableDef, Function])

	def __repr__(self):
		return "\n".join([str(x) for x in self])


class SystemDescription(List):
	grammar = "system", csl(TemplateG), ';'

	def __repr__(self):
		return "system %s;" % ",".join([str(x) for x in self])


class TemplateArguments(List):
	grammar = csl(Expression)

	def __repr__(self):
		return ",".join([str(x) for x in self])


class TemplateDeclaration:
	grammar = word, optional("(", attr("params", Parameters), ")"), '=', attr("base", TemplateG), '(', attr("args",
	                                                                                              TemplateArguments), ')', ';'

	@property
	def hasParams(self):
		return hasattr(self, "params")

	def __init__(self, parser):
		self.name = parser
		TemplateG.addTemplate(self.name)

	def __repr__(self):
		return "%s%s = %s(%s);" % (self.name, "(%s)" % self.params if self.hasParams else "", self.base, self.args)


class System(List):
	grammar = maybe_some([Comment, SystemDescription, TemplateDeclaration])

	def __repr__(self):
		return "\n".join([str(x) for x in self])


class Decl:
	def __init__(self, values=None):
		if not values:
			values = dict()
		self.values = values
		self.setValues()
		self.evalMap = dict(Sum=self.notImplemented, FunctionCall=self.notImplemented,
		                    ParExpression=self.evalParExpression,
		                    InlineIf=self.notImplemented, Constant=self.evalConstant, Variable=self.evalValue,
		                    Value=self.evalValue, uList=self.evalUList)

	def getVars(self, const=False):
		"""

		:param const:
		:type: bool
		:return: List of variables
		:rtype: typing.List[VariableDef]
		"""
		result = [x for x in self.declaration if type(x) == VariableDef]
		if const:
			result = result + [x for x in self.declaration if type(x) == ConstantDef]
		return result

	def getConst(self, name: str) -> Union[ConstantDef, TypeDef, None]:
		return next((x for x in self.declaration if type(x) in [ConstantDef, TypeDef] and x.name == name), None)

	def getVar(self, name: str) -> Union[VariableDef, None]:
		return next((x for x in self.declaration if type(x) == VariableDef and x.name == name), None)

	def getType(self, typeName):
		return next((x for x in self.declaration if type(x) == TypeDef and x.name == typeName), None)

	def notImplemented(self, expression):
		raise Exception("The expression %s cannot yet be evaluated." % expression)

	def evalExpression(self, expression: Expression):
		self.setValues()
		lhs = self.evalMap[str(type(expression.lhs)).split('\'')[1].split('.')[-1]](expression.lhs)
		if expression.hasRhs:
			return eval(str(lhs) + expression.op + str(self.evalExpression(expression.rhs)))
		return lhs

	def evalParExpression(self, expression: ParExpression):
		return self.evalExpression(expression.exp)

	def evalConstant(self, constant):
		definition = self.getConst(constant.value)  # type: ConstantDef
		if definition.hasValue:
			return self.evalExpression(definition.value)
		else:
			if constant.type == 'double':
				return 0.0
			else:
				return 0

	def evalVariable(self, variable):
		definition = self.getVar(variable.name)
		if definition.hasValue:
			return self.evalExpression(variable.value)
		else:
			if variable.type == 'double':
				return 0.0
			else:
				return 0

	def evalValue(self, value):
		return int(value.value) if not value.neg else -int(value.value)

	def evalUList(self, value: uList):
		return [self.evalExpression(exp) if type(exp) == Expression else exp for exp in list(value)]


	def setValues(self):
		ConstantList.reset()
		VariableList.reset()
		TypeList.reset()
		Type.reset()
		if "Types" in self.values:
			Type.grammar = self.values["Types"]
		if "TypeList" in self.values:
			TypeList.types = self.values["TypeList"]
		if "Variables" in self.values:
			VariableList.assign([x for x in self.values["Variables"] if x.const == False])
			ConstantList.assign([x for x in self.values["Variables"] if x.const == True])

	def parse(self, declaration, type=Declaration):
		self.setValues()
		if type != Declaration:
			return parse(declaration, type)
		self.declaration = parse(declaration, type)
		self.values = dict()
		self.values["Types"] = Type.grammar
		self.values["TypeList"] = TypeList.types
		self.values["Variables"] = VariableList.getVars() + ConstantList.getVars()

	def getVariables(self):
		return self.values["Variables"]

	def getValues(self):
		return self.values

	def setVar(self, var, value):
		decl = next(x for x in self.declaration if type(x) == VariableDef and x.name == var.split('[')[0]) # type: VariableDef
		if decl.type in ["double", "clock"]:
			castType = float
		else:
			castType = int
		if '[' in var:
			indexList = []
			remaining = var
			while remaining:
				idx, remaining = remaining.split('[', 1)[1].split(']', 1)
				indexList.append(int(idx))
			if not decl.hasValue and decl.isList:
				length = decl.length  # type: lengthDef
				lengths = [self.evalExpression(length.value)]
				while length.hasNext:
					length = length.next
					lengths.append(self.evalExpression(length.value))
				decl.value = Expression()
				decl.value.lhs = uList([Expression()]*lengths[0])
				if len(lengths) > 1:
					decl.value.lhs.initChildLists(lengths[1:])
			decl.setValue(castType(value), indexList)
		else:
			decl.setValue(castType(value))

	def __repr__(self):
		return str(self.declaration)
