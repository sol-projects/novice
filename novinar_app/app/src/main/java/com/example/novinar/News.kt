package com.example.novinar

import android.os.Parcelable
import kotlinx.parcelize.Parcelize


@Parcelize
data class Location(
    val type: String?,
    val coordinates: List<Double>?
) : Parcelable

@Parcelize
data class News(
    val _id: String?,
    val title: String,
    val content: String,
    val location: Location?,
    val category: String?,
    val timestamp: String?,
    val url: String?,
    val authors: List<String>?,
    val categories: List<String>?,
    val views: List<String>?
) : Parcelable
