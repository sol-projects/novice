
fun main() {
    val input = "if( x >0){ print(\"Positive\") } else if (x < 0) { print(\"Negative\") } else { print(\"Zero\") }"
    val scanner = Scanner(input)
    val tokens = scanner.scanTokens()
    for (token in tokens) {
        println("${token.type}: ${token.value}")
    }
}
