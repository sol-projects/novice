package org.example.model
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.*
data class INews(
    var title: String,
    var url: String,
    var date: java.util.Date,
    val authors: List<String>,
    var content: String,
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