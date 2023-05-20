import java.io.File

fun main(args: Array<String>) {
    val input = "if (x = y+4){ y = x - 4;}"

    val scanner = Scanner(input)
    val tokens = scanner.scanTokens()

    println("Scanned Tokens:")
    for (token in tokens) {
        println("${token.type}: ${token.value}")
    }

    val parser = Parser(tokens)
    val isSyntaxValid = parser.parse()

    if (isSyntaxValid) {
        println("Syntax is valid!")
    }
}
