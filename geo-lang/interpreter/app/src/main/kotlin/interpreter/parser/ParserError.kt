package interpreter.parser
import interpreter.tokenizer.*

sealed class ParserError{
    data class Generic(val tokenInfo: TokenInfo, val string: String) : ParserError()
    data class InvalidFor(val tokenInfo: TokenInfo) : ParserError()
    data class InvalidAssignment(val tokenInfo: TokenInfo, val string: String) : ParserError()
    data class MissingClosingBraces(val tokenInfo: TokenInfo) : ParserError()
    data class MissingClosingParentheses(val tokenInfo: TokenInfo) : ParserError()
    data class ExpectedStartingBraces(val tokenInfo: TokenInfo) : ParserError()
    data class ExpectedComma(val tokenInfo: TokenInfo) : ParserError()
    data class ExpectedFunctionName(val tokenInfo: TokenInfo) : ParserError()
    data class ExpectedStartingParentheses(val tokenInfo: TokenInfo) : ParserError()
    data class MissingSemicolon(val tokenInfo: TokenInfo) : ParserError()
    data class UndefinedVariable(val tokenInfo: TokenInfo) : ParserError()
    data class InvalidParameterSyntax(val tokenInfo: TokenInfo) : ParserError()
    data class NotAType(val tokenInfo: TokenInfo) : ParserError()
}

fun parserPrintError(parserError: ParserError) {
    when (parserError) {
        is ParserError.Generic -> println("Syntax error: unexpected token '${parserError.tokenInfo.lexeme}' of type ${parserError.tokenInfo.type} after ${parserError.string} on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.InvalidFor -> println("Syntax error: invalid for loop structure, unexpected token '${parserError.tokenInfo.lexeme}' of type ${parserError.tokenInfo.type} on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.InvalidAssignment -> println("Syntax error: invalid assignment; found '${parserError.tokenInfo.lexeme}' of type ${parserError.tokenInfo.type} after ${parserError.string} on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.MissingClosingBraces -> println("Syntax error: missing closing braces on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.MissingClosingParentheses -> println("Syntax error: missing closing parentheses on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.ExpectedStartingBraces -> println("Syntax error: expected {, found '${parserError.tokenInfo.lexeme}' on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.ExpectedComma -> println("Syntax error: expected ',', found '${parserError.tokenInfo.lexeme}' on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.ExpectedStartingParentheses -> println("Syntax error: expected (, found '${parserError.tokenInfo.lexeme}' on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.MissingSemicolon -> println("Syntax error: missing semicolon ';' on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.UndefinedVariable -> println("Evaluation error: variable '${parserError.tokenInfo.lexeme}' on line ${parserError.tokenInfo.startPosition.row} undefined")
        is ParserError.ExpectedFunctionName -> println("Syntax error: expected function name, found '${parserError.tokenInfo.lexeme}' on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.InvalidParameterSyntax -> println("Syntax error: invalid parameter syntax, found '${parserError.tokenInfo.lexeme}' on line ${parserError.tokenInfo.startPosition.row}")
        is ParserError.NotAType -> println("Syntax error: '${parserError.tokenInfo.lexeme}' on line ${parserError.tokenInfo.startPosition.row} is not a valid type")

    }
}