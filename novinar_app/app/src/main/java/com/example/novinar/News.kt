package com.example.novinar

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class News(
    val _id: String?,
    val title: String,
    val content: String,
    val category: String,
    val image: String?,
    val latitude: String?,
    val longitude: String?,
    val timestamp: String?
) : Parcelable
