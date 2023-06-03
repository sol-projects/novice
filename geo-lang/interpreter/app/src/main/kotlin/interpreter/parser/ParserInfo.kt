package interpreter.parser
import interpreter.tokenizer.*

class ParserInfo(
    val tokens: List<TokenInfo>,
    var currentTokenInfo: TokenInfo,
    private var i: Int
) {
    fun matchToken(expectedToken: TokenType): Boolean {
        currentTokenInfo = tokens[i]
        if (tokens[i].type == expectedToken) {
            i++
            return true
        }
        return false
    }

    fun lastNTokensLexemes(n: Int): String {
        var c = n
        var counter = 1
        var string = ""
        var index = i - counter
        while (n > 0 && index >= 0) {
            string = "${tokens[index].lexeme} $string"
            counter++
            index = i - counter
            c--
        }
        return string.trim()
    }
}