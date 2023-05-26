package org.example.model
import java.text.SimpleDateFormat
import java.util.*

data class INews(
    val title: String,
    val url: String,
    val date: Date,
    val authors: List<String>,
    val content: String,
    val categories: List<String>,
    val location: Location,
    val _id: String = "0",
    val __v: Int = 0
) {
    override fun toString(): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd")
        val formattedDate = dateFormat.format(date)
        val formattedAuthors = authors.joinToString(", ")
        val formattedCategories = categories.joinToString(", ")

        return """
            Title: $title
            URL: $url
            Date: $formattedDate
            Authors: $formattedAuthors
            Content: $content
            Categories: $formattedCategories
        """.trimIndent().plus("\n")
    }
}

data class Location(
    val type: String,
    val coordinates: Pair<Double, Double>
)