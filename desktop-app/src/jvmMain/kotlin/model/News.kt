package org.example.model
import java.text.SimpleDateFormat
import java.util.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
data class INews(
        var title: String,
        var url: String,
        var date: Date,
        val authors: List<String>,
        var content: String,
        val categories: List<String>,
        val location: Location,
        val _id: String = "0",
        val __v: Int = 0
) {
    override fun toString(): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        val formattedDate = dateFormat.format(date)
        val formattedAuthors = authors.joinToString("\", \"", "[\"", "\"]")
        val formattedCategories = categories.joinToString("\", \"", "[\"", "\"]")
        val coordinates = "[${location.coordinates.first}, ${location.coordinates.second}]"
        val jsonContent = Json.encodeToString(content)

        return "{\n" +
                "    \"title\": \"$title\",\n" +
                "    \"url\": \"$url\",\n" +
                "    \"date\": \"$formattedDate\",\n" +
                "    \"authors\": $formattedAuthors,\n" +
                "    \"content\": $jsonContent,\n" +
                "    \"categories\": $formattedCategories,\n" +
                "    \"location\": {\n" +
                "        \"type\": \"${location.type}\",\n" +
                "        \"coordinates\": $coordinates\n" +
                "    }\n" +
                "}"
    }
}
data class Location(
    val type: String,
    val coordinates: Pair<Double, Double>
)