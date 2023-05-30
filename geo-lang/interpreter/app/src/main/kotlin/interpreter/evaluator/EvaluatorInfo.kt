package interpreter.evaluator
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import interpreter.tokenizer.TokenInfo
import interpreter.tokenizer.TokenType
import java.io.ByteArrayInputStream

data class EvaluatorInfo(
    val tokens: List<TokenInfo>,
    var currentTokenInfo: TokenInfo,
    var i: Int,
    var variables: HashMap<String, Variable>,
    var featureCollection: JsonObject = JsonObject(),
    var features: JsonArray = JsonArray()
    ) {
    init {
        featureCollection.addProperty("type", "FeatureCollection")
    }
    fun matchToken(expectedToken: TokenType): Boolean {
        currentTokenInfo = tokens[i]
        if (tokens[i].type == expectedToken) {
            i++
            return true
        }
        return false
    }

    fun skipCurrentScope() {
        var openBracesCount = 1
        while (i < tokens.size && openBracesCount > 0) {
            if (tokens[i].type == TokenType.LeftBraces) {
                openBracesCount++
            } else if (tokens[i].type == TokenType.RightBraces) {
                openBracesCount--
            }

            i++
        }

        i--
    }

    fun skipRemainingIfElse() {
        var openIfElseCount = 1

        while (i < tokens.size && openIfElseCount > 0) {
            if (tokens[i].type == TokenType.LeftBraces) {
                i++
                skipCurrentScope()

                openIfElseCount--
                if((tokens[i].type == TokenType.ElseIf || tokens[i].type == TokenType.Else)) {
                    openIfElseCount++
                }
            } else if ((tokens[i].type == TokenType.ElseIf || tokens[i].type == TokenType.Else)) {
                openIfElseCount++
            }

            i++
        }

        i--
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
