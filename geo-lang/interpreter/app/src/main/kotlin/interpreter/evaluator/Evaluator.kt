package interpreter.evaluator

import com.google.gson.GsonBuilder
import interpreter.tokenizer.Position
import interpreter.tokenizer.TokenInfo
import interpreter.tokenizer.TokenType

fun error(): Value {
    return Value.NumberType(0)
}
fun evaluate(tokens: List<TokenInfo>): String {
    val evaluatorInfo = EvaluatorInfo(
        tokens,
        TokenInfo(TokenType.BoolType, "", Position(1, 1)),
        0,
        hashMapOf()
    )

    while (!evaluatorInfo.matchToken(TokenType.EOF)) {
        bitwise(evaluatorInfo)
        if (evaluatorInfo.matchToken(TokenType.EOF)) {
            break
        } else {
            evaluatorInfo.matchToken(TokenType.Semicolon)
        }
    }

    val gson = GsonBuilder().setPrettyPrinting().create()
    return gson.toJson(evaluatorInfo.featureCollection)
}

fun bitwise(evaluatorInfo: EvaluatorInfo): Value {
    var res = addition(evaluatorInfo)
    while (evaluatorInfo.matchToken(TokenType.BWAnd) || evaluatorInfo.matchToken(TokenType.BWOr)) {
        if (evaluatorInfo.currentTokenInfo.type == TokenType.BWAnd) {
            res = Value.NumberType((res as Value.NumberType).value.toLong() and (addition(evaluatorInfo) as Value.NumberType).value.toLong())
        } else if (evaluatorInfo.currentTokenInfo.type == TokenType.BWOr) {
            res = Value.NumberType((res as Value.NumberType).value.toLong() or (addition(evaluatorInfo) as Value.NumberType).value.toLong())
        }
    }

    return res
}

private fun addition(evaluatorInfo: EvaluatorInfo): Value {
    var res = multiplication(evaluatorInfo)
    while (evaluatorInfo.matchToken(TokenType.Addition) || evaluatorInfo.matchToken(TokenType.Subtraction)) {
        if (evaluatorInfo.currentTokenInfo.type == TokenType.Addition) {
            res = Value.NumberType((res as Value.NumberType).value + (multiplication(evaluatorInfo) as Value.NumberType).value)
        } else if (evaluatorInfo.currentTokenInfo.type == TokenType.Subtraction) {
            res = Value.NumberType((res as Value.NumberType).value - (multiplication(evaluatorInfo) as Value.NumberType).value)
        }
    }

    return res
}

private operator fun Number.plus(value: Number): Number {
    return when {
        this is Double && value is Int -> this + value.toDouble()
        this is Int && value is Double -> this.toDouble() + value
        this is Int && value is Int -> this + value
        this is Double && value is Double -> this + value
        else -> throw IllegalArgumentException("Unsupported.")
    }
}

private operator fun Number.minus(value: Number): Number {
    return when {
        this is Double && value is Int -> this - value.toDouble()
        this is Int && value is Double -> this.toDouble() - value
        this is Int && value is Int -> this - value
        this is Double && value is Double -> this - value
        else -> throw IllegalArgumentException("Unsupported.")
    }
}

private operator fun Number.times(value: Number): Number {
    return when {
        this is Double && value is Int -> this * value.toDouble()
        this is Int && value is Double -> this.toDouble() * value
        this is Int && value is Int -> this * value
        this is Double && value is Double -> this * value
        else -> throw IllegalArgumentException("Unsupported.")
    }
}

private operator fun Number.div(value: Number): Number {
    return when {
        this is Double && value is Int -> this / value.toDouble()
        this is Int && value is Double -> this.toDouble() / value
        this is Int && value is Int -> {
            if (value != 0) {
                this.toDouble() / value.toDouble()
            } else {
                throw ArithmeticException("Division by zero")
            }
        }
        this is Double && value is Double -> this / value
        else -> throw IllegalArgumentException("Unsupported.")
    }
}

private operator fun Number.compareTo(value: Number): Int {
    return when {
        this is Double && value is Int -> this.compareTo(value.toDouble())
        this is Int && value is Double -> this.toDouble().compareTo(value)
        this is Int && value is Int -> this.compareTo(value)
        this is Double && value is Double -> this.compareTo(value)
        else -> throw IllegalArgumentException("Unsupported.")
    }
}

private operator fun Number.unaryMinus(): Number {
    return when (this) {
        is Double -> -this
        is Int -> -this
        else -> throw IllegalArgumentException("Unsupported.")
    }
}

private fun multiplication(evaluatorInfo: EvaluatorInfo): Value {
    var res = comparisonOperators(evaluatorInfo)
    while (evaluatorInfo.matchToken(TokenType.Multiplication) || evaluatorInfo.matchToken(TokenType.Division)) {
        if (evaluatorInfo.currentTokenInfo.type == TokenType.Multiplication) {
            res = Value.NumberType((res as Value.NumberType).value * (comparisonOperators(evaluatorInfo) as Value.NumberType).value)
        } else if (evaluatorInfo.currentTokenInfo.type == TokenType.Division) {
            res = Value.NumberType((res as Value.NumberType).value / (comparisonOperators(evaluatorInfo) as Value.NumberType).value)
        }
    }

    return res
}

private fun comparisonOperators(evaluatorInfo: EvaluatorInfo): Value {
    var res = unary(evaluatorInfo)
    while (evaluatorInfo.matchToken(TokenType.GreaterThan) || evaluatorInfo.matchToken(TokenType.LowerThan) || evaluatorInfo.matchToken(TokenType.Comparison)) {
        if (evaluatorInfo.currentTokenInfo.type == TokenType.GreaterThan) {
            res = if ((res as Value.NumberType).value > (unary(evaluatorInfo) as Value.NumberType).value) Value.NumberType(1) else Value.NumberType(0)
        } else if (evaluatorInfo.currentTokenInfo.type == TokenType.LowerThan) {
            res = if ((res as Value.NumberType).value < (unary(evaluatorInfo) as Value.NumberType).value) Value.NumberType(1) else Value.NumberType(0)
        } else if (evaluatorInfo.currentTokenInfo.type == TokenType.Comparison) {
            res = if ((res as Value.NumberType).value == (unary(evaluatorInfo) as Value.NumberType).value) Value.NumberType(1) else Value.NumberType(0)
        }
    }

    return res
}

private fun unary(evaluatorInfo: EvaluatorInfo): Value {
    return if (evaluatorInfo.matchToken(TokenType.Addition)) {
        primary(evaluatorInfo)
    } else if (evaluatorInfo.matchToken(TokenType.Subtraction)) {
        Value.NumberType(-(primary(evaluatorInfo) as Value.NumberType).value)
    } else {
        primary(evaluatorInfo)
    }
}

private fun primary(evaluatorInfo: EvaluatorInfo): Value {
    return when {
        evaluatorInfo.matchToken(TokenType.True) -> Value.NumberType(1)
        evaluatorInfo.matchToken(TokenType.False) -> Value.NumberType(0)
        evaluatorInfo.matchToken(TokenType.Int) -> Value.NumberType(evaluatorInfo.currentTokenInfo.lexeme.toInt())
        evaluatorInfo.matchToken(TokenType.Float) -> Value.NumberType(evaluatorInfo.currentTokenInfo.lexeme.toDouble())
        evaluatorInfo.matchToken(TokenType.Identifier) -> evaluateIdentifier(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.String) -> Value.StringType(evaluatorInfo.currentTokenInfo.lexeme)
        evaluatorInfo.matchToken(TokenType.LeftBracket) -> evaluateArray(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.VarDeclaration) -> createVar(evaluatorInfo, false)
        evaluatorInfo.matchToken(TokenType.ConstDeclaration) -> createVar(evaluatorInfo, true)
        evaluatorInfo.matchToken(TokenType.Println) -> evaluatePrintln(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.If) -> evaluateIf(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.Function) -> storeFunction(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.For) -> evaluateFor(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.ElseIf) -> evaluateIf(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.Group) -> evaluateGroup(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.Fetch) -> Value.ArrayType(fetch())
        evaluatorInfo.matchToken(TokenType.LeftParenthesis) -> {
            val result = bitwise(evaluatorInfo)
            evaluatorInfo.matchToken(TokenType.RightParenthesis)
            result
        }
        else -> Value.NumberType(0.0)
    }
}

fun storeFunction(evaluatorInfo: EvaluatorInfo): Value {
    evaluatorInfo.matchToken(TokenType.Identifier)
    val name = evaluatorInfo.currentTokenInfo.lexeme
    evaluatorInfo.matchToken(TokenType.LeftParenthesis)
    val parameters = LinkedHashMap<String, Variable>()
    while(!evaluatorInfo.matchToken(TokenType.RightParenthesis)) {
        evaluatorInfo.matchToken(TokenType.Identifier)
        val parameterName = evaluatorInfo.currentTokenInfo.lexeme
        evaluatorInfo.matchToken(TokenType.Colon)
        val parameterType = type(evaluatorInfo)
        parameters[parameterName] = Variable(true, parameterType, when(parameterType){
            TokenType.I32 -> Value.NumberType(0)
            TokenType.F32 -> Value.NumberType(0.0)
            TokenType.StringType -> Value.StringType("")
            TokenType.Function -> Value.FunctionType(Function("", LinkedHashMap(), TokenType.NoOp, 0))
            TokenType.ArrayType -> Value.ArrayType(ArrayList<Any?>())
            else -> {
                evaluatorPrintError(EvaluatorError.AssignmentTypeError(evaluatorInfo.currentTokenInfo, evaluatorInfo.lastNTokensLexemes(3)))
                Value.NumberType(0)
            }
        })
        evaluatorInfo.matchToken(TokenType.Comma)
    }

    evaluatorInfo.matchToken(TokenType.Colon)
    type(evaluatorInfo)
    val returnType = evaluatorInfo.currentTokenInfo.type
    evaluatorInfo.matchToken(TokenType.LeftBraces)
    val start = evaluatorInfo.i
    evaluatorInfo.skipCurrentScope()
    evaluatorInfo.matchToken(TokenType.RightBraces)

    evaluatorInfo.variables[name] =
        Variable(true, TokenType.FunctionType, Value.FunctionType(Function(name, parameters, returnType, start)))
    return Value.FunctionType(Function(name, parameters, returnType, start))
}
fun runFunction(evaluatorInfo: EvaluatorInfo, name: String): Value {
    val fn = evaluatorInfo.variables[name]
    if (fn != null) {
        val function = fn.value as? Value.FunctionType
        if (function != null) {
            for ((key, value) in function.value.parameters) {
                evaluatorInfo.variables[key] = value
                evaluatorInfo.variables[key]?.value = bitwise(evaluatorInfo)
                evaluatorInfo.matchToken(TokenType.Comma)
                if(evaluatorInfo.matchToken(TokenType.RightParenthesis)) {
                    break;
                }
            }

            val i = evaluatorInfo.i
            evaluatorInfo.i = function.value.start
            val res = scope(evaluatorInfo)
            for ((key, _) in function.value.parameters) {
                evaluatorInfo.variables.remove(key)
            }
            evaluatorInfo.i = i
            return res
        } else {

        }
    } else {
        evaluatorPrintError(EvaluatorError.FunctionNotFound(evaluatorInfo.currentTokenInfo, evaluatorInfo.lastNTokensLexemes(3)))
    }

    return Value.NumberType(0)
}

fun toNumericArray(value: Value): List<Number> {
    val result = mutableListOf<Number>()

    if (value is Value.ArrayType) {
        for (item in value.value) {
            if (item is Value.NumberType) {
                result.add(item.value)
            } else if (item is Value.ArrayType) {
                result.addAll(toNumericArray(item))
            }
        }
    }

    return result
}

fun toNumericArray2(value: Value): List<List<Number>> {
    val result = mutableListOf<List<Number>>()

    if (value is Value.ArrayType) {
        if (value.value.isNotEmpty() && value.value[0] is Value.ArrayType) {
            for (item in value.value) {
                if (item is Value.ArrayType) {
                    result.add(toNumericArray(item))
                }
            }
        }
    }

    return result
}

fun evaluateGroup(evaluatorInfo: EvaluatorInfo): Value {
    evaluatorInfo.matchToken(TokenType.LeftBraces)
    while(!evaluatorInfo.matchToken(TokenType.RightBraces)) {
        if (evaluatorInfo.matchToken(TokenType.Building)) {
            evaluatorInfo.matchToken(TokenType.String)
            val name = evaluatorInfo.currentTokenInfo.lexeme
            Box(toNumericArray(bitwise(evaluatorInfo)), toNumericArray(bitwise(evaluatorInfo)), evaluatorInfo, name)
        }

        if (evaluatorInfo.matchToken(TokenType.Point)) {
            evaluatorInfo.matchToken(TokenType.String)
            val name = evaluatorInfo.currentTokenInfo.lexeme
            Point(toNumericArray(bitwise(evaluatorInfo)), name, evaluatorInfo)
        }

        if (evaluatorInfo.matchToken(TokenType.Line)) {
            evaluatorInfo.matchToken(TokenType.String)
            val name = evaluatorInfo.currentTokenInfo.lexeme
            Line(toNumericArray(bitwise(evaluatorInfo)), toNumericArray(bitwise(evaluatorInfo)), name, evaluatorInfo)
        }

        if (evaluatorInfo.matchToken(TokenType.Curve)) {
            evaluatorInfo.matchToken(TokenType.String)
            val name = evaluatorInfo.currentTokenInfo.lexeme
            Curve(toNumericArray(bitwise(evaluatorInfo)), toNumericArray(bitwise(evaluatorInfo)), (bitwise(evaluatorInfo) as Value.NumberType).value.toDouble(), name, evaluatorInfo)
        }

        if (evaluatorInfo.matchToken(TokenType.Circle)) {
            evaluatorInfo.matchToken(TokenType.String)
            val name = evaluatorInfo.currentTokenInfo.lexeme
            val ev = bitwise(evaluatorInfo)
            val center = toNumericArray(ev)

            val radius = (bitwise(evaluatorInfo) as Value.NumberType).value.toDouble()
            Circle(center, radius, name, evaluatorInfo)
        }

        if (evaluatorInfo.matchToken(TokenType.Polyline)) {
            evaluatorInfo.matchToken(TokenType.String)
            val name = evaluatorInfo.currentTokenInfo.lexeme
            val points = toNumericArray2(bitwise(evaluatorInfo))

            Polyline(points, name, evaluatorInfo)
        }

        if (evaluatorInfo.matchToken(TokenType.NPolygon)) {
            evaluatorInfo.matchToken(TokenType.String)
            val name = evaluatorInfo.currentTokenInfo.lexeme
            val points = toNumericArray2(bitwise(evaluatorInfo))

            NPolygon(points, name, evaluatorInfo)
        }
    }

    return Value.NumberType(0)
}

private inline fun <reified T> getArrayType(): ArrayList<T> {
    return ArrayList()
}
fun evaluateArray(evaluatorInfo: EvaluatorInfo): Value {
    if(evaluatorInfo.matchToken(TokenType.RightBracket)) {
        return Value.ArrayType(ArrayList<Any?>())
    } else {
        val first = bitwise(evaluatorInfo)
        val list = Value.ArrayType(when (first) {
            is Value.StringType -> getArrayType<String>()
            is Value.NumberType -> {
                if (first.value is Double) {
                    getArrayType<Double>()
                } else {
                    getArrayType<Int>()
                }
            }
            is Value.ArrayType -> getArrayType<ArrayList<*>>()
            is Value.FunctionType -> getArrayType<Function>()
            is Value.NewsType -> getArrayType<News>()
        })

        (list.value as ArrayList<Any?>).add(first)
        while (!evaluatorInfo.matchToken(TokenType.RightBracket)) {
            evaluatorInfo.matchToken(TokenType.Comma)
            val value = bitwise(evaluatorInfo)
            if (first::class == value::class) {
                list.value.add(value)
            } else {
                evaluatorPrintError(
                    EvaluatorError.ArrayTypeError(
                        evaluatorInfo.currentTokenInfo,
                        evaluatorInfo.lastNTokensLexemes(3)
                    )
                )
            }
        }

        return list
    }
}

private fun evaluateIf(evaluatorInfo: EvaluatorInfo): Value {
    var value: Value

    val cond = bitwise(evaluatorInfo)
    val conditional = when(cond) {
        is Value.StringType -> cond.value.isNotEmpty()
        is Value.NumberType ->
            if(cond.value is Double && cond.value != 0.0) {
                true
            } else { cond.value is Int && cond.value != 0 }
        is Value.ArrayType -> cond.value.isNotEmpty()
        else -> false
    }

    if(conditional) {
        value = scope(evaluatorInfo)
        while(evaluatorInfo.matchToken(TokenType.ElseIf) || evaluatorInfo.matchToken((TokenType.Else))) {
            if(evaluatorInfo.currentTokenInfo.type == TokenType.ElseIf) {
                evaluatorInfo.skipRemainingIfElse()
                evaluatorInfo.matchToken(TokenType.LeftBraces)
            }
            evaluatorInfo.matchToken(TokenType.LeftBraces)
            evaluatorInfo.skipCurrentScope()
            evaluatorInfo.matchToken(TokenType.RightBraces)
        }
        return value
    } else {
        evaluatorInfo.matchToken(TokenType.LeftBraces)
        evaluatorInfo.skipCurrentScope()
        evaluatorInfo.matchToken(TokenType.RightBraces)
        if(evaluatorInfo.matchToken(TokenType.ElseIf)) {
            return evaluateIf(evaluatorInfo)
        } else {
            if(evaluatorInfo.matchToken((TokenType.Else))) {
                return scope(evaluatorInfo)
            }
            return Value.NumberType(0)
        }
    }
}

private fun evaluateFor(evaluatorInfo: EvaluatorInfo): Value {
    evaluatorInfo.matchToken(TokenType.Identifier)
    var objName = evaluatorInfo.currentTokenInfo.lexeme

    if(evaluatorInfo.variables.containsKey(objName)) {
        evaluatorPrintError(EvaluatorError.ShadowedVariable(evaluatorInfo.currentTokenInfo))
    }

    var returnValue: Value = Value.NumberType(0)
    if(evaluatorInfo.matchToken(TokenType.In)) {
        var array: Value = Value.NumberType(0)
        if(evaluatorInfo.matchToken(TokenType.Identifier)) {
            array = evaluateIdentifier(evaluatorInfo)
        } else if(evaluatorInfo.matchToken(TokenType.LeftBracket)) {
            array = evaluateArray(evaluatorInfo)
        }
        val i = evaluatorInfo.i
        var endi = evaluatorInfo.i
        val ktArr = (array as Value.ArrayType).value as ArrayList<Value>
        for(current: Value in ktArr) {
            val type = when (current) {
                is Value.StringType -> {
                    TokenType.StringType
                }
                is Value.NumberType -> {
                    if (current.value is Int) {
                        TokenType.I32
                    } else {
                        TokenType.F32
                    }
                }
                is Value.ArrayType -> {
                    TokenType.ArrayType
                }
                is Value.FunctionType -> {
                    TokenType.FunctionType
                }
                is Value.NewsType -> {
                    TokenType.NewsType
                }
            }

            evaluatorInfo.variables[objName] = Variable(true, type, current)
            evaluatorInfo.matchToken(TokenType.LeftBraces)
            returnValue = scope(evaluatorInfo)
            evaluatorInfo.variables.remove(objName)
            endi = evaluatorInfo.i
            evaluatorInfo.i = i
        }
        evaluatorInfo.i = endi
        return returnValue
        //add for i in 1..3 {} (range)
    }

    return Value.NumberType(0)
}
private fun scope(evaluatorInfo: EvaluatorInfo): Value {
    var value: Value = Value.NumberType(0)
    evaluatorInfo.matchToken(TokenType.LeftBraces)
    while (!evaluatorInfo.matchToken(TokenType.RightBraces)) {
        value = bitwise(evaluatorInfo)
    }

    return value
}
private fun evaluatePrintln(evaluatorInfo: EvaluatorInfo): Value {
    val formatString = StringBuilder()
    val args = mutableListOf<String>()
    evaluatorInfo.matchToken(TokenType.LeftParenthesis)
    while (!evaluatorInfo.matchToken(TokenType.RightParenthesis)) {
        if (evaluatorInfo.matchToken(TokenType.String)) {
            val lexeme = evaluatorInfo.currentTokenInfo.lexeme
            val interpolatedString = lexeme.replace(Regex("\\$\\{([^}]+)}")) { matchResult ->
                val variableName = matchResult.groupValues[1]
                val variable = evaluatorInfo.variables[variableName]
                val value = when (variable?.type) {
                    TokenType.I32 -> (variable.value as Value.NumberType).value.toInt().toString()
                    TokenType.F32 -> (variable.value as Value.NumberType).value.toDouble().toString()
                    TokenType.StringType -> (variable.value as Value.StringType).value.substring(1, (variable.value as Value.StringType).value.length - 1)
                    TokenType.NewsType -> (variable.value as Value.NewsType).value.toString()
                    TokenType.ArrayType -> (variable.value as Value.ArrayType).value.joinToString(separator = ", ") { element ->
                        when (element) {
                            is Value.StringType -> element.value
                            is Value.NumberType -> element.value.toString()
                            else -> element.toString()
                        }
                    }
                    else -> ""
                }

                value
            }
            formatString.append(interpolatedString)
        } else {
            val arg = bitwise(evaluatorInfo)
            args.add(arg.toString())
            formatString.append("%s")
        }
    }

    val formattedOutput = formatString.toString()
    val formattedArgs = args.toTypedArray()
    var output: String = ""
    try {
        output = formattedOutput.format(*formattedArgs)
        println(output.substring(1, output.length - 1))
    } catch (e: Exception) {
        output = "invalid"
        println(output)
    }

    return Value.NumberType(0.0)
}

private fun evaluateIdentifier(evaluatorInfo: EvaluatorInfo): Value {
    var name = evaluatorInfo.currentTokenInfo.lexeme
    var variable = evaluatorInfo.variables[evaluatorInfo.currentTokenInfo.lexeme]
    if (variable != null) {
        if (evaluatorInfo.matchToken(TokenType.Dot)) {
            if(evaluatorInfo.matchToken(TokenType.Location)) {
                val news = (variable.value as? Value.NewsType)?.value
                val list = ArrayList<Value.NumberType>()
                if (news != null) {
                    list.add(news.location.coordinates.first)
                }
                if (news != null) {
                    list.add(news.location.coordinates.second)
                }

                if(list.isEmpty()) {
                    evaluatorPrintError(EvaluatorError.AssignmentTypeError(evaluatorInfo.currentTokenInfo, evaluatorInfo.lastNTokensLexemes(3)))
                }

                return Value.ArrayType(list)
            }
            if (evaluatorInfo.matchToken(TokenType.Push) || evaluatorInfo.matchToken(TokenType.Pop)) {
                var isPush = evaluatorInfo.currentTokenInfo.type == TokenType.Push
                if (!variable.isConst) {
                    evaluatorInfo.matchToken(TokenType.LeftParenthesis)
                    val arrayValue = variable.value as? Value.ArrayType
                    if (arrayValue != null) {
                        val array = arrayValue.value as ArrayList<Value>
                        val value = if(!isPush) { Value.NumberType(0) } else { bitwise(evaluatorInfo) }
                        if(!isPush) {
                            if(array.isNotEmpty()) {
                                array.removeAt(array.size - 1)
                            } else {
                                evaluatorPrintError(EvaluatorError.PopOnEmptyArray(evaluatorInfo.currentTokenInfo, evaluatorInfo.lastNTokensLexemes(3)))
                            }
                        } else if(array.isEmpty()) {
                            array.add(value)
                        } else if(array[0]::class == value::class) {
                            array.add(value)
                        } else {
                            evaluatorPrintError(
                                EvaluatorError.ArrayTypeError(
                                    evaluatorInfo.currentTokenInfo,
                                    evaluatorInfo.lastNTokensLexemes(3)
                                )
                            )
                        }
                        evaluatorInfo.matchToken(TokenType.RightParenthesis)
                    } else {
                        evaluatorPrintError(EvaluatorError.ArrayTypeError(evaluatorInfo.currentTokenInfo, evaluatorInfo.lastNTokensLexemes(3)))
                    }
                } else {
                    evaluatorPrintError(
                        EvaluatorError.AssignmentConstError(
                            evaluatorInfo.currentTokenInfo,
                            evaluatorInfo.lastNTokensLexemes(3)
                        )
                    )
                }
            }

            if(evaluatorInfo.matchToken(TokenType.Get)) {
                evaluatorInfo.matchToken(TokenType.LeftParenthesis)
                val arrayValue = variable.value as? Value.ArrayType
                if (arrayValue != null) {
                    val array = arrayValue.value as ArrayList<Value>
                    val n = bitwise(evaluatorInfo)
                    val getN = (n as Value.NumberType).value.toInt()
                    evaluatorInfo.matchToken(TokenType.RightParenthesis)
                    return array[getN]
                } else {
                    evaluatorPrintError(EvaluatorError.ArrayTypeError(evaluatorInfo.currentTokenInfo, evaluatorInfo.lastNTokensLexemes(3)))
                }
            }
        }

        if(evaluatorInfo.matchToken(TokenType.Equals)) {
            val eval = bitwise(evaluatorInfo)
            if(variable.value::class == eval::class) {
                if(!variable.isConst) {
                    variable.value = eval
                } else {
                    evaluatorPrintError(EvaluatorError.AssignmentConstError(evaluatorInfo.currentTokenInfo, evaluatorInfo.lastNTokensLexemes(3)))
                }
            } else {
                evaluatorPrintError(EvaluatorError.AssignmentTypeError(evaluatorInfo.currentTokenInfo, evaluatorInfo.lastNTokensLexemes(3) + " Types are: $eval::class and $variable.value::class"))
            }
        }

        if(evaluatorInfo.matchToken(TokenType.LeftParenthesis)) {
            return runFunction(evaluatorInfo, name)
        }

        return variable.value
    } else {
        evaluatorPrintError(EvaluatorError.UndefinedVariable(evaluatorInfo.currentTokenInfo))
        return error()
    }
}

private fun createVar(evaluatorInfo: EvaluatorInfo, isConst: Boolean): Value {
    evaluatorInfo.matchToken(TokenType.Identifier)
    val name = evaluatorInfo.currentTokenInfo.lexeme
    if(evaluatorInfo.matchToken(TokenType.Colon)) {
        val type_ = type(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.Equals)
        val value = bitwise(evaluatorInfo)
        evaluatorInfo.variables[name] = Variable(isConst, type_, value)
        return value
    } else {
        evaluatorInfo.matchToken(TokenType.Equals)
        val value = bitwise(evaluatorInfo)
        val type_ = when(value) {
            is Value.StringType -> TokenType.StringType
            is Value.NumberType ->
                if(value.value is Double) {
                    TokenType.F32
                } else { TokenType.I32 }
            is Value.ArrayType -> TokenType.ArrayType
            is Value.FunctionType -> TokenType.FunctionType
            is Value.NewsType -> TokenType.NewsType
        }

        evaluatorInfo.variables[name] = Variable(isConst, type_, value)
        return value
    }
}

private fun isNumericType(evaluatorInfo: EvaluatorInfo): Boolean {
    return evaluatorInfo.matchToken(TokenType.I32)
            || evaluatorInfo.matchToken(TokenType.F32)
            || evaluatorInfo.matchToken(TokenType.BoolType)
}

private fun type(evaluatorInfo: EvaluatorInfo): TokenType {
    if (evaluatorInfo.matchToken(TokenType.I32)
        || evaluatorInfo.matchToken(TokenType.F32)
        || evaluatorInfo.matchToken(TokenType.BoolType)
        || evaluatorInfo.matchToken(TokenType.CharType)
        || evaluatorInfo.matchToken(TokenType.StringType)
        || evaluatorInfo.matchToken((TokenType.NewsType))
        || evaluatorInfo.matchToken(TokenType.ArrayType)) {
        return evaluatorInfo.currentTokenInfo.type
    }

    if(evaluatorInfo.matchToken(TokenType.LeftBracket)) {
        type(evaluatorInfo)
        evaluatorInfo.matchToken(TokenType.RightBracket)
        return TokenType.ArrayType
    }

    return evaluatorInfo.currentTokenInfo.type
}