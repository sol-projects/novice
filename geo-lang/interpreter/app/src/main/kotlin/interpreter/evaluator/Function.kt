package interpreter.evaluator

import interpreter.tokenizer.TokenType

data class Function(
    val name: String,
    var parameters: LinkedHashMap<String, Variable>,
    val returnType: TokenType,
    val start: Int
)
