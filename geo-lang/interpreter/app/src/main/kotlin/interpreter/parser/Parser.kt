package interpreter.parser
import interpreter.tokenizer.*

fun parse(tokens: List<TokenInfo>): Boolean {
    val parserInfo = ParserInfo(
        tokens,
        TokenInfo(TokenType.BoolType, "", Position(1, 1)),
        0
    )

    while (!parserInfo.matchToken(TokenType.EOF)) {
        if(!bitwise(parserInfo)) {
            return false
        }
        if (parserInfo.matchToken(TokenType.EOF)) {
            break
        } else {
            endOfStatement(parserInfo)
        }
    }

    return true
}
private fun bitwise(parserInfo: ParserInfo): Boolean {
    var res = addition(parserInfo)
    while (parserInfo.matchToken(TokenType.BWAnd) || parserInfo.matchToken(TokenType.BWOr)) {
        res = addition(parserInfo)
    }

    return res
}

private fun addition(parserInfo: ParserInfo): Boolean {
    var res = multiplication(parserInfo)
    while (parserInfo.matchToken(TokenType.Addition) || parserInfo.matchToken(TokenType.Subtraction)) {
        res = multiplication(parserInfo)
    }

    return res
}

private fun multiplication(parserInfo: ParserInfo): Boolean {
    var res = comparisonOperators(parserInfo)
    while (parserInfo.matchToken(TokenType.Multiplication) || parserInfo.matchToken(TokenType.Division)) {
        res = comparisonOperators(parserInfo)
    }

    return res
}

private fun comparisonOperators(parserInfo: ParserInfo): Boolean {
    var res = unary(parserInfo)
    while (parserInfo.matchToken(TokenType.GreaterThan) || parserInfo.matchToken(TokenType.LowerThan) || parserInfo.matchToken(TokenType.Comparison)) {
        res = unary(parserInfo)
    }

    return res
}

private fun assignment(parserInfo: ParserInfo): Boolean {
    return if (parserInfo.matchToken(TokenType.Identifier) && parserInfo.matchToken(TokenType.Equals)) {
        bitwise(parserInfo)
    } else {
        parserPrintError(ParserError.InvalidAssignment(parserInfo.currentTokenInfo, parserInfo.lastNTokensLexemes(3)))
        false
    }
}

private fun endOfStatement(parserInfo: ParserInfo): Boolean {
    if (!parserInfo.matchToken(TokenType.Semicolon)) {
        parserPrintError(ParserError.MissingSemicolon(parserInfo.currentTokenInfo))
        return false
    }

    return true
}

private fun unary(parserInfo: ParserInfo): Boolean {
    return if (parserInfo.matchToken(TokenType.Addition) || parserInfo.matchToken(TokenType.Subtraction)) {
        primary(parserInfo)
    } else {
        primary(parserInfo)
    }
}

private fun primary(parserInfo: ParserInfo): Boolean {
    if (parserInfo.matchToken(TokenType.Int) || parserInfo.matchToken(TokenType.Hex)) {
        // Do something
    } else if (parserInfo.matchToken(TokenType.Function)){
        return function(parserInfo)
    } else if (parserInfo.matchToken(TokenType.Identifier)) {
        if (parserInfo.matchToken(TokenType.Equals)) {
            return bitwise(parserInfo)
        } else {
            // Do something
        }
    } else if (parserInfo.matchToken(TokenType.LeftParenthesis)) {
        if(!bitwise(parserInfo)) {
            return false
        }
        if (!parserInfo.matchToken(TokenType.RightParenthesis)) {
            parserPrintError(ParserError.MissingClosingParentheses(parserInfo.currentTokenInfo))
            return false
        }
    } else if (parserInfo.matchToken(TokenType.For)) {
        if (parserInfo.matchToken(TokenType.LeftParenthesis)) {
            if(!assignment(parserInfo)) {
                return false
            }
            if (!parserInfo.matchToken(TokenType.Range)) {
                parserPrintError(ParserError.InvalidFor(parserInfo.currentTokenInfo))
                return false
            }

            bitwise(parserInfo)

            if (!parserInfo.matchToken(TokenType.RightParenthesis)) {
                parserPrintError(ParserError.MissingClosingParentheses(parserInfo.currentTokenInfo))
                return false
            }

            return scope(parserInfo)
        } else {
            parserPrintError(ParserError.ExpectedStartingParentheses(parserInfo.currentTokenInfo))
            return false
        }
    } else if (parserInfo.matchToken(TokenType.Loop)) {
        if (!parserInfo.matchToken(TokenType.LeftBraces)) {
            parserPrintError(ParserError.ExpectedStartingBraces(parserInfo.currentTokenInfo))
            return false
        }
        while (!parserInfo.matchToken(TokenType.RightBraces)) {
            if (parserInfo.matchToken(TokenType.EOF)) {
                parserPrintError(ParserError.MissingClosingBraces(parserInfo.currentTokenInfo))
                return false
            }
            if(!bitwise(parserInfo)){
                return false
            }
        }
    } else if (parserInfo.matchToken(TokenType.LeftBraces)) {
        while (!parserInfo.matchToken(TokenType.RightBraces)) {
            if (parserInfo.matchToken(TokenType.EOF)) {
                parserPrintError(ParserError.MissingClosingBraces(parserInfo.currentTokenInfo))
                return false
            }
            if(!bitwise(parserInfo)){
                return false
            }
        }
    } else {
        parserPrintError(ParserError.Generic(parserInfo.currentTokenInfo, parserInfo.lastNTokensLexemes(3)))
        return false
    }

    return true
}

private fun parameter(parserInfo: ParserInfo): Boolean {
    if(!parserInfo.matchToken(TokenType.Identifier)) {
        parserPrintError(ParserError.InvalidParameterSyntax(parserInfo.currentTokenInfo))
        return false
    }

    if(!parserInfo.matchToken(TokenType.Colon)) {
        parserPrintError(ParserError.InvalidParameterSyntax(parserInfo.currentTokenInfo))
        return false
    }

    if(!type(parserInfo)) {
        return false
    }

    return true
}

private fun scope(parserInfo: ParserInfo): Boolean {
    if (!parserInfo.matchToken(TokenType.LeftBraces)) {
        parserPrintError(ParserError.MissingClosingParentheses(parserInfo.currentTokenInfo.copy()))
        return false
    }

    while (!parserInfo.matchToken(TokenType.RightBraces)) {
        if(!bitwise(parserInfo)) {
            return false
        }

        if (parserInfo.matchToken(TokenType.RightBraces)) {
            break
        } else {
            if(!endOfStatement(parserInfo)) {
                return false
            }
        }
    }

    return true
}
private fun type(parserInfo: ParserInfo): Boolean {
    if(parserInfo.matchToken(TokenType.I32)
        || parserInfo.matchToken(TokenType.F32)
        || parserInfo.matchToken(TokenType.BoolType)
        || parserInfo.matchToken(TokenType.CharType)
        || parserInfo.matchToken(TokenType.StringType)) {
        return true
    }

    parserPrintError(ParserError.NotAType(parserInfo.currentTokenInfo))
    return false
}
private fun function(parserInfo: ParserInfo): Boolean {
    if(!parserInfo.matchToken(TokenType.Identifier)) {
        parserPrintError(ParserError.ExpectedFunctionName(parserInfo.currentTokenInfo))
        return false
    }

    if(!parserInfo.matchToken(TokenType.LeftParenthesis)) {
        parserPrintError(ParserError.ExpectedStartingParentheses(parserInfo.currentTokenInfo))
        return false
    }

    while(!parserInfo.matchToken(TokenType.RightParenthesis)) {
        if(!parameter(parserInfo)) {
            return false
        }

        if(!parserInfo.matchToken(TokenType.RightParenthesis)) {
            if(!parserInfo.matchToken(TokenType.Comma)) {
                parserPrintError(ParserError.ExpectedComma(parserInfo.currentTokenInfo))
                return false
            }
        }
    }

    return scope(parserInfo)
}