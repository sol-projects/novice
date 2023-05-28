package interpreter.evaluator

import interpreter.tokenizer.TokenType

data class Variable(
    val isConst: Boolean,
    val type: TokenType,
    var value: Value
)
