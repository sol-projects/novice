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
        } /*else {
            endOfStatement(parserInfo)
        }*/
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

private fun unary(parserInfo: ParserInfo): Boolean {
    return if (parserInfo.matchToken(TokenType.Addition) || parserInfo.matchToken(TokenType.Subtraction)) {
        primary(parserInfo)
    } else {
        primary(parserInfo)
    }
}

private fun primary(parserInfo: ParserInfo): Boolean {
    if (parserInfo.matchToken(TokenType.Int)
        || parserInfo.matchToken(TokenType.Hex)
        || parserInfo.matchToken(TokenType.Float)
        || parserInfo.matchToken(TokenType.String)) {
        return true
    } else if (parserInfo.matchToken(TokenType.Function)){
        return function(parserInfo)
    } else if (parserInfo.matchToken(TokenType.Identifier)) {
        if (parserInfo.matchToken(TokenType.Equals)) {
            return bitwise(parserInfo)
        } else if (parserInfo.matchToken(TokenType.LeftParenthesis)) {
            return functionCall(parserInfo)
        }

        else if (parserInfo.matchToken(TokenType.Dot)) {
            if(!(parserInfo.matchToken(TokenType.Identifier)
                        || parserInfo.matchToken(TokenType.Push)
                        || parserInfo.matchToken(TokenType.Remove)
                        || parserInfo.matchToken(TokenType.Pop))) {
                parserPrintError(ParserError.ExpectedFunctionName(parserInfo.currentTokenInfo))
                return false
            }
            if(!parserInfo.matchToken(TokenType.LeftParenthesis)) {
                parserPrintError(ParserError.ExpectedStartingParentheses(parserInfo.currentTokenInfo))
                return false
            }
            functionCall(parserInfo)
        }
    } else if (parserInfo.matchToken(TokenType.Println)) {
        return if (parserInfo.matchToken(TokenType.LeftParenthesis)) {
            functionCall(parserInfo)
        } else {
            parserPrintError(ParserError.ExpectedStartingParentheses(parserInfo.currentTokenInfo))
            false
        }
    } else if (parserInfo.matchToken(TokenType.Loop)) {
        return scope(parserInfo)
    } else if (parserInfo.matchToken(TokenType.Return)) {
        return bitwise(parserInfo)
    } else if (parserInfo.matchToken((TokenType.ConstDeclaration)) || parserInfo.matchToken(TokenType.VarDeclaration)) {
        return variable(parserInfo)
    } else if(parserInfo.matchToken(TokenType.If)) {
        return if_(parserInfo)
    } else if(parserInfo.matchToken(TokenType.Group)) {
        return group(parserInfo)
    } else if(parserInfo.matchToken(TokenType.LeftBracket)) {
        return array(parserInfo)
    } else if (parserInfo.matchToken(TokenType.LeftParenthesis)) {
        if (!bitwise(parserInfo)) {
            return false
        }
        if (!parserInfo.matchToken(TokenType.RightParenthesis)) {
            parserPrintError(ParserError.MissingClosingParentheses(parserInfo.currentTokenInfo))
            return false
        }
    } else if (parserInfo.matchToken(TokenType.For)) {
        return for_(parserInfo)
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

fun group(parserInfo: ParserInfo): Boolean {
    if(!parserInfo.matchToken(TokenType.LeftBraces)) {
        parserPrintError(ParserError.ExpectedStartingParentheses(parserInfo.currentTokenInfo))
        return false
    }

    while(!parserInfo.matchToken(TokenType.RightBraces)) {
        if(parserInfo.matchToken(TokenType.Building) || parserInfo.matchToken(TokenType.Line)) {
            if(!parserInfo.matchToken(TokenType.String)) {
                parserPrintError(ParserError.ExpectedName(parserInfo.currentTokenInfo))
                return false
            }

            if(!bitwise(parserInfo)) {
                return false
            }

            if(!bitwise(parserInfo)) {
                return false
            }
        }

        if(parserInfo.matchToken(TokenType.Curve)) {
            if(!parserInfo.matchToken(TokenType.String)) {
                parserPrintError(ParserError.ExpectedName(parserInfo.currentTokenInfo))
                return false
            }

            if(!bitwise(parserInfo)) {
                return false
            }

            if(!bitwise(parserInfo)) {
                return false
            }

            if(!bitwise(parserInfo)) {
                return false
            }
        }

        if(parserInfo.matchToken(TokenType.Point)) {
            if(!parserInfo.matchToken(TokenType.String)) {
                parserPrintError(ParserError.ExpectedName(parserInfo.currentTokenInfo))
                return false
            }

            if(!bitwise(parserInfo)) {
                return false
            }
        }
    }

    return true
}

private fun variable(parserInfo: ParserInfo): Boolean {
    if(!parserInfo.matchToken(TokenType.Identifier)) {
        parserPrintError(ParserError.InvalidAssignment(parserInfo.currentTokenInfo, parserInfo.lastNTokensLexemes(3)))
        return false
    }

    if(parserInfo.matchToken(TokenType.Colon)) {
        if(!type(parserInfo)) {
            return false
        }

        if(!parserInfo.matchToken(TokenType.Equals)) {
            parserPrintError(ParserError.InvalidAssignment(parserInfo.currentTokenInfo, parserInfo.lastNTokensLexemes(3)))
            return false
        }

        if(!bitwise(parserInfo)) {
            return false
        }
    } else if(parserInfo.matchToken(TokenType.Equals)) {
        if(!bitwise(parserInfo)) {
            return false
        }
    } else {
        parserPrintError(ParserError.InvalidAssignment(parserInfo.currentTokenInfo, parserInfo.lastNTokensLexemes(3)))
        return false
    }

    return true
}

private fun if_(parserInfo: ParserInfo): Boolean {
    if(!bitwise(parserInfo)) {
        return false
    }

    if(!scope(parserInfo)) {
        return false
    }

    while(parserInfo.matchToken(TokenType.ElseIf)) {
        if(!bitwise(parserInfo)) {
            return false
        }

        if(!scope(parserInfo)) {
            return false
        }
    }

    if(parserInfo.matchToken(TokenType.Else)) {
        if(!scope(parserInfo)) {
            return false
        }
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

    return type(parserInfo)
}

private fun array(parserInfo: ParserInfo): Boolean {
    while(!parserInfo.matchToken(TokenType.RightBracket)) {
        if(!bitwise(parserInfo)) {
            return false
        }

        if(!parserInfo.matchToken(TokenType.RightBracket)) {
            if(!parserInfo.matchToken(TokenType.Comma)) {
                parserPrintError(ParserError.ExpectedComma(parserInfo.currentTokenInfo))
                return false
            }
        } else {
            break
        }
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
        }/* else {
            if(!endOfStatement(parserInfo)) {
                return false
            }
        }*/
    }

    return true
}
private fun type(parserInfo: ParserInfo): Boolean {
    if(parserInfo.matchToken(TokenType.I32)
        || parserInfo.matchToken(TokenType.F32)
        || parserInfo.matchToken(TokenType.BoolType)
        || parserInfo.matchToken(TokenType.CharType)
        || parserInfo.matchToken(TokenType.StringType)
        || parserInfo.matchToken((TokenType.NewsType))) {
        return true
    }

    if(parserInfo.matchToken(TokenType.LeftBracket)) {
        if(type(parserInfo)) {
            if(parserInfo.matchToken(TokenType.RightBracket)) {
                return true
            }
        }
    }
    parserPrintError(ParserError.NotAType(parserInfo.currentTokenInfo))
    return false
}

private fun for_(parserInfo: ParserInfo): Boolean {
    if(!range(parserInfo)) {
        parserPrintError(ParserError.InvalidFor(parserInfo.currentTokenInfo))
    }

    return true
}

private fun range(parserInfo: ParserInfo): Boolean {
    if(parserInfo.matchToken(TokenType.Identifier)) {
        if(parserInfo.matchToken(TokenType.In)) {
            return if(parserInfo.matchToken(TokenType.Identifier)) {
                scope(parserInfo)
            } else {
                false
            }
        } else if(parserInfo.matchToken(TokenType.Range)) {
            return parserInfo.matchToken(TokenType.Identifier) || parserInfo.matchToken(TokenType.Int)
        }
    }

    if(!parserInfo.matchToken(TokenType.Int)) {
        return false
    }

    if(!parserInfo.matchToken(TokenType.Range)) {
        return false
    }

    return parserInfo.matchToken(TokenType.Identifier) || parserInfo.matchToken(TokenType.Int)
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
        } else {
            break
        }
    }

    if(parserInfo.matchToken(TokenType.Colon)) {
        if(!type(parserInfo)) {
            return false
        }
    }
    return scope(parserInfo)
}

private fun functionCall(parserInfo: ParserInfo): Boolean {
    while(!parserInfo.matchToken(TokenType.RightParenthesis)) {
        if(!bitwise(parserInfo)) {
            return false
        }

        if(!parserInfo.matchToken(TokenType.RightParenthesis)) {
            if(!parserInfo.matchToken(TokenType.Comma)) {
                parserPrintError(ParserError.ExpectedComma(parserInfo.currentTokenInfo))
                return false
            }
        } else {
            break
        }
    }

    return true
}