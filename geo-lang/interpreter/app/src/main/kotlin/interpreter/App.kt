package interpreter
import interpreter.evaluator.evaluate
import interpreter.parser.parse
import interpreter.tokenizer.tokenize
import java.io.File
import interpreter.evaluator.Value

fun main(args: Array<String>) {
    for (arg in args) {
        tokenize(File(arg).readText())?.let {
            if(parse(it)) {
                val eval = evaluate(it)
                if(eval.isNotEmpty()) {
                    File("out.json").writeText(eval)
                }
            }
        }
    }
}