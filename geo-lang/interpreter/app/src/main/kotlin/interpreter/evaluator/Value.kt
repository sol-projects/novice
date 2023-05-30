package interpreter.evaluator
sealed class Value {
    data class StringType(val value: String) : Value()
    data class NumberType(val value: Number) : Value()
    data class ArrayType(val value: ArrayList<*>) : Value()
    data class FunctionType(val value: Function) : Value()
    data class NewsType(val value: News) : Value()
}
