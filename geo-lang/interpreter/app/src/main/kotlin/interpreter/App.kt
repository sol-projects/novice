package interpreter
import interpreter.parser.parse
import interpreter.tokenizer.tokenize
import java.io.File

fun main(args: Array<String>) {
    for (arg in args) {
        tokenize(File(arg).readText())?.let { parse(it) }
    }
}