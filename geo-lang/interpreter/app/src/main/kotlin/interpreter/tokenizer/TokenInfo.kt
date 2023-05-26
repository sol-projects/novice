package interpreter.tokenizer

data class Position(
    var row: Int,
    var col: Int
)
data class TokenInfo(
    var type: TokenType,
    var lexeme: String,
    val startPosition: Position
)
