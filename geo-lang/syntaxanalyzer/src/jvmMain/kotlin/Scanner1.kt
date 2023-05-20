enum class TokenType {
    INT,
    EXPRESSION,
    EXPRESSION_TERM,
    EXPRESSION_POWER,
    FACTOR,
    EXPRESSION_VALID,
    FUNCTION,
    FUNCTION_HEAD,
    PARAMETERS,
    PARAMETER,
    FUNCTION_BODY,
    MAYBE_RETURNS,
    COMMAND,
    LINE,
    BEND,
    BOX,
    CIRC,
    STRING,
    CHARS,
    SCOPE,
    RANGE,
    TYPE,
    ARRAY_TYPE,
    ARRAY,
    ARRAY_ELEMENTS,
    FOR,
    LOOP,
    ANGLE,
    CITY_ELEMENT,
    BUILDING_ELEMENTS,
    BUILDING_TYPES,
    STATEMENTS,
    STATEMENT,
    CITY,
    CITY_ELEMENTS,
    FLOAT,
    CHAR,
    ASSIGNMENT,
    VAR_DECLARATION,
    MAIN,
    POINT,
    POINT_COMPONENT,
    IF,
    IF_EXPR,
    ELSE,
    ELSE_IF,
    IDENTIFIER,
    LETTER,
    CHARACTERS
}

data class Token(val type: TokenType, val value: String)

class Scanner(private val input: String) {
    private var currentPosition = 0

    fun scanTokens(): List<Token> {
        val tokens = mutableListOf<Token>()
        while (!isAtEnd()) {
            val token = scanToken()
            if (token != null) {
                tokens.add(token)
            }
        }
        return tokens
    }

    private fun scanToken(): Token? {
        val char = advance()
        return when (char) {
            in '0'..'9' -> scanInt(char)
            '+' -> Token(TokenType.EXPRESSION, "+")
            '-' -> Token(TokenType.EXPRESSION, "-")
            '*' -> Token(TokenType.EXPRESSION_TERM, "*")
            '/' -> Token(TokenType.EXPRESSION_TERM, "/")
            '^' -> Token(TokenType.EXPRESSION_POWER, "^")
            '(' -> Token(TokenType.FACTOR, "(")
            ')' -> Token(TokenType.FACTOR, ")")
            '{' -> Token(TokenType.SCOPE, "{")
            '}' -> Token(TokenType.SCOPE, "}")
            '[' -> Token(TokenType.ARRAY, "[")
            ']' -> Token(TokenType.ARRAY, "]")
            ',' -> Token(TokenType.ARRAY_ELEMENTS, ",")
            ';' -> Token(TokenType.STATEMENT, ";")
            ':' -> Token(TokenType.PARAMETER, ":")
            '.' -> Token(TokenType.CHAR, ".")
            '=' -> Token(TokenType.ASSIGNMENT, "=")
            '"' -> scanString()
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' -> scanIdentifier(char)
            else -> null
        }
    }

    private fun scanInt(firstChar: Char): Token {
        val intVal = StringBuilder().append(firstChar)
        while (peek().isDigit()) {
            intVal.append(advance())
        }
        return Token(TokenType.INT, intVal.toString())
    }

    private fun scanString(): Token? {
        val string = StringBuilder()
        while (!isAtEnd()) {
            val char = advance()

            if (char == '"') {
                return Token(TokenType.STRING, string.toString())
            } else {
                string.append(char)
            }
        }
        return null
    }

    private fun scanIdentifier(firstChar: Char): Token {
        val identifier = StringBuilder().append(firstChar)
        while (peek().isLetterOrDigit() || peek() == '_') {
            identifier.append(advance())
        }
        return when (identifier.toString()) {
            "if" -> Token(TokenType.IF, "if")
            "else" -> Token(TokenType.ELSE, "else")
            "fn" -> Token(TokenType.FUNCTION, "fn")
            "let" -> Token(TokenType.ASSIGNMENT, "let")
            "const" -> Token(TokenType.ASSIGNMENT, "const")
            "for" -> Token(TokenType.FOR, "for")
            "loop" -> Token(TokenType.LOOP, "loop")
            "return" -> Token(TokenType.MAYBE_RETURNS, "return")
            "main" -> Token(TokenType.MAIN, "main")
            "true", "false", "bool", "u8", "u16", "u32", "u64", "u128", "i8", "i16",
            "i32", "i64", "i128", "f32", "f64", "char", "string" -> Token(TokenType.TYPE, identifier.toString())
            else -> Token(TokenType.IDENTIFIER, identifier.toString())
        }
    }

    private fun advance(): Char {
        currentPosition++
        return input[currentPosition - 1]
    }

    private fun peek(): Char {
        return if (isAtEnd()) '\u0000' else input[currentPosition]
    }

    private fun isAtEnd(advanceBy: Int = 0): Boolean {
        return currentPosition + advanceBy >= input.length
    }
}
