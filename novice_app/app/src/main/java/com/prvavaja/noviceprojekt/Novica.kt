package com.prvavaja.noviceprojekt

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
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

class Novica(var title: String,
             var url: String,
             var date: java.util.Date,
             val authors: List<String>,
             var content: String,
             val categories: List<String>,
             val location: Location,
             val _id: String = "0",
             val __v: Int = 0) {
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