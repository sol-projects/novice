class Parser(private val tokens: List<Token>) {
    private var currentTokenIndex = 0
    private val currentToken: Token?
        get() = if (currentTokenIndex < tokens.size) tokens[currentTokenIndex] else null

    fun parse(): Boolean {
        try {
            parseStatements()
            return true
        } catch (e: ParseException) {
            println("Syntax Error: ${e.message}")
            return false
        }
    }

    // Productions
    private fun parseStatements() {
        while (isNextTokenIn(TokenType.STATEMENT)) {
            parseStatement()
        }
    }

    private fun parseStatement() {
        when (currentToken?.type) {
            TokenType.EXPRESSION -> {
                expect(TokenType.EXPRESSION)
                expect(TokenType.SEMICOLON)
            }
            TokenType.FOR -> parseForLoop()
            TokenType.FUNCTION -> parseFunction()
            TokenType.LOOP -> parseLoop()
            TokenType.ASSIGNMENT -> parseAssignment()
            TokenType.COMMAND -> parseCommand()
            else -> throw unexpectedTokenError()
        }
    }

    private fun parseForLoop() {
        expect(TokenType.FOR)
        expect(TokenType.IDENTIFIER)
        expect(TokenType.IN)
        parseExpression()
        expect(TokenType.SCOPE)
        parseStatements()
        expect(TokenType.SCOPE)
    }

    private fun parseFunction() {
        expect(TokenType.FUNCTION)
        expect(TokenType.IDENTIFIER)
        expect(TokenType.SCOPE)
        parseParameters()
        expect(TokenType.SCOPE)
    }

    private fun parseParameters() {
        if (isNextToken(TokenType.PARAMETER)) {
            expect(TokenType.PARAMETER)
            parseParameters()
        }
    }

    private fun parseLoop() {
        expect(TokenType.LOOP)
        if (isNextToken(TokenType.RANGE)) {
            expect(TokenType.RANGE)
            parseExpression()
            expect(TokenType.RANGE)
        }
        expect(TokenType.SCOPE)
        parseStatements()
        expect(TokenType.SCOPE)
    }

    private fun parseAssignment() {
        expect(TokenType.ASSIGNMENT)
        expect(TokenType.VAR_DECLARATION)
    }

    private fun parseExpression() {
        expect(TokenType.EXPRESSION)
        if (isNextTokenIn(TokenType.EXPRESSION_TERM, TokenType.EXPRESSION_POWER)) {
            expect(TokenType.EXPRESSION_TERM, TokenType.EXPRESSION_POWER)
            parseExpression()
        }
    }

    private fun parseCommand() {
        when (currentToken?.type) {
            TokenType.LINE -> {
                expect(TokenType.LINE)
                expect(TokenType.POINT)
                expect(TokenType.COMMA)
                expect(TokenType.POINT)
            }
            TokenType.BEND -> {
                expect(TokenType.BEND)
                expect(TokenType.POINT)
                expect(TokenType.COMMA)
                expect(TokenType.POINT)
                expect(TokenType.COMMA)
                expect(TokenType.ANGLE)
            }
            TokenType.BOX -> {
                expect(TokenType.BOX)
                expect(TokenType.POINT)
                expect(TokenType.COMMA)
                expect(TokenType.POINT)
            }
            TokenType.CIRC -> {
                expect(TokenType.CIRC)
                expect(TokenType.POINT)
                expect(TokenType.COMMA)
                if (isNextToken(TokenType.INT, TokenType.FLOAT)) {
                    expect(TokenType.INT, TokenType.FLOAT)
                }
            }
            else -> throw unexpectedTokenError()
        }
    }

    // Helper functions

    private fun expect(vararg expectedTypes: TokenType) {
        val token = currentToken
        val expectedTypeString = expectedTypes.joinToString(" or ")
        if (token != null && token.type in expectedTypes) {
            currentTokenIndex++
        } else {
            val errorMessage = if (token != null) {                "Expected $expectedTypeString, but found ${token.type} '${token.value}'"
            } else {
                "Expected $expectedTypeString, but reached the end of input"
            }
            throw ParseException(errorMessage)
        }
    }

    private fun isNextToken(vararg expectedTypes: TokenType): Boolean {
        val token = currentToken
        return token != null && token.type in expectedTypes
    }

    private fun isNextTokenIn(vararg expectedTypes: TokenType): Boolean {
        val token = currentToken
        return token != null && token.type in expectedTypes
    }

    private fun unexpectedTokenError(): ParseException {
        val token = currentToken
        val errorMessage = if (token != null) {
            "Unexpected token '${token.type}' '${token.value}'"
        } else {
            "Unexpected end of input"
        }
        return ParseException(errorMessage)
    }
}

class ParseException(message: String) : Exception(message)


