package org.example.model
import java.util.Date

data class INews(
    val title: String,
    val url: String,
    val date: Date,
    val authors: List<String>,
    val content: String,
    val categories: List<String>,
    val location: String,
)