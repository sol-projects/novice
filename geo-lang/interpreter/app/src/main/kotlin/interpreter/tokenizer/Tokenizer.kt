package interpreter.tokenizer

fun tokenize(input: String): List<TokenInfo>? {
    return Tokenizer(input).tokenize()
}
private class Tokenizer(
    var input: String,
    var current: Int = -1,
    var position: Position = Position(1, 0)
) {
    fun tokenize(): List<TokenInfo>? {
        val tokens = mutableListOf<TokenInfo>()
        while (current < input.length - 1) {
            val token = getToken()
            if (token != null) {
                if (token.type != TokenType.Ignore) {
                    tokens.add(token)
                }
            } else {
                return null;
            }
        }

        tokens.add(TokenInfo(TokenType.EOF, "", Position(0, 0)))
        return tokens
    }

    private fun getToken(): TokenInfo? {
        position.col += 1
        return when (input[++current]) {
            in '0'..'9' -> number()
            '+' -> TokenInfo(TokenType.Addition, "+", position.copy())
            '-' -> TokenInfo(TokenType.Subtraction, "-", position.copy())
            '*' -> TokenInfo(TokenType.Multiplication, "*", position.copy())
            '|' -> TokenInfo(TokenType.BWOr, "|", position.copy())
            '&' -> TokenInfo(TokenType.BWAnd, "&", position.copy())
            '/' -> TokenInfo(TokenType.Division, "/", position.copy())
            '^' -> TokenInfo(TokenType.Power, "^", position.copy())
            '(' -> TokenInfo(TokenType.LeftParenthesis, "(", position.copy())
            ')' -> TokenInfo(TokenType.RightParenthesis, ")", position.copy())
            '{' -> TokenInfo(TokenType.LeftBraces, "{", position.copy())
            '}' -> TokenInfo(TokenType.RightBraces, "}", position.copy())
            '[' -> TokenInfo(TokenType.LeftBracket, "[", position.copy())
            ']' -> TokenInfo(TokenType.RightBracket, "]", position.copy())
            '<' -> TokenInfo(TokenType.LowerThan, "<", position.copy())
            '>' -> TokenInfo(TokenType.GreaterThan, ">", position.copy())
            ',' -> TokenInfo(TokenType.Comma, ",", position.copy())
            ';' -> TokenInfo(TokenType.Semicolon, ";", position.copy())
            ':' -> TokenInfo(TokenType.Colon, ":", position.copy())
            '#' -> hex()
            '.' -> rangeOrDot()
            '\'' -> char()
            '=' -> equals()
            '"' -> string()
            in 'A' .. 'z' -> identifier()
            '\n' -> newline()
            '\t' -> tab()
            ' ' -> space()
            else -> null
        }
    }

    private fun string(): TokenInfo? {
        val token = TokenInfo(TokenType.String, "\"", position.copy())
        while (current < input.length) {
            val c = input[++current]
            position.col += 1
            if (c != '"') {
                token.lexeme += c
            } else {
                token.lexeme += '"'
                return token;
            }
        }

        println("Tokenizer error: invalid string: ${token.lexeme} on line: ${token.startPosition.row}")
        return null
    }

    private fun number(): TokenInfo? {
        val token = TokenInfo(TokenType.Int, input[current].toString(), position.copy())
        while (current < input.length - 1) {
            val c = input[++current]
            position.col += 1
            if (c in '0'..'9') {
                token.lexeme += c
            } else if (c == '.') {
                if (token.lexeme.contains('.')) {
                    println("Tokenizer error: invalid number: ${token.lexeme} on line: ${token.startPosition.row}")
                    return null
                }
                token.type = TokenType.Float
                token.lexeme += c
            } else {
                --current
                --position.col
                return token
            }
        }

        return token
    }

    private fun char(): TokenInfo? {
        val token = TokenInfo(TokenType.Char, input[current].toString(), position.copy())
        return if(current < input.length - 2) {
            token.lexeme += input[++current]
            if(input[++current] != '\'') {
                println("Tokenizer error: expected \"'\" to form a char on line: ${position.row} but unable")
                null
            } else {
                token.lexeme += input[current]
                token
            }
        } else {
            println("Tokenizer error: expected to form a char on line: ${position.row} but unable")
            null
        }
    }

    private fun equals(): TokenInfo {
        return if(current < input.length - 1) {
            if(input[++current] != '=') {
                --current
                TokenInfo(TokenType.Equals, input[current].toString(), position.copy())
            } else {
                TokenInfo(TokenType.Comparison, "==", position.copy())
            }
        } else {
            TokenInfo(TokenType.Equals, input[current].toString(), position.copy())
        }
    }

    private fun rangeOrDot(): TokenInfo? {
        return if(current < input.length - 1) {
            if(input[++current] != '.') {
                --current
                TokenInfo(TokenType.Dot, ".", position.copy())
            } else {
                TokenInfo(TokenType.Range, "..", position.copy())
            }
        } else {
            println("Tokenizer error: expected another '.' to form a range on line: ${position.row}")
            return null
        }
    }

    private fun identifier(): TokenInfo? {
        val token = TokenInfo(TokenType.Identifier, input[current].toString(), position.copy())
        while (current < input.length - 1) {
            val c = input[++current]
            position.col += 1
            if (c in '0'..'9' || c in 'A'..'z') {
                token.lexeme += c
            } else {
                --current
                --position.col
                token.type = checkIdentifier(token.lexeme)
                return token
            }
        }

        token.type = checkIdentifier(token.lexeme)
        return token
    }

    private fun checkIdentifier(lexeme: String): TokenType {
        return when (lexeme) {
            "break" -> TokenType.Break
            "continue" -> TokenType.Continue
            "if" -> TokenType.If
            "elif" -> TokenType.ElseIf
            "else" -> TokenType.Else
            "fn" -> TokenType.Function
            "let" -> TokenType.VarDeclaration
            "const" -> TokenType.ConstDeclaration
            "for" -> TokenType.For
            "in" -> TokenType.In
            "loop" -> TokenType.Loop
            "return" -> TokenType.Return
            "char" -> TokenType.CharType
            "string" -> TokenType.StringType
            "i32" -> TokenType.I32
            "f32" -> TokenType.F32
            "true" -> TokenType.True
            "false" -> TokenType.False
            "road" -> TokenType.Road
            "building" -> TokenType.Building
            "point" -> TokenType.Point
            "group" -> TokenType.Group
            "bool" -> TokenType.BoolType
            "news" -> TokenType.NewsType
            "bend" -> TokenType.Bend
            "line" -> TokenType.Line
            "box" -> TokenType.Box
            "circle" -> TokenType.Circle
            "println" -> TokenType.Println
            "push" -> TokenType.Push
            "remove" -> TokenType.Remove
            "pop" -> TokenType.Pop
            else -> TokenType.Identifier
        }
    }

    private fun newline(): TokenInfo {
        position.col = 1
        position.row += 1

        return TokenInfo(TokenType.Ignore, "", Position(0, 0))
    }

    private fun tab(): TokenInfo {
        position.col += 4
        return TokenInfo(TokenType.Ignore, "", Position(0, 0))
    }

    private fun space(): TokenInfo {
        position.col += 1
        return TokenInfo(TokenType.Ignore, "", Position(0, 0))
    }

    private fun hex(): TokenInfo {
        val token = TokenInfo(TokenType.Hex, "#", position.copy())
        while (current < input.length - 1) {
            val c = input[++current]
            position.col += 1
            if (c in 'A' .. 'F' || c in 'a' .. 'f' || c in '0' .. '9') {
                token.lexeme += c
            } else {
                return token;
            }
        }

        return token
    }
}