package com.example.novinar.api

import com.example.novinar.News
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Call
import retrofit2.http.*

interface ApiService {

    @Multipart
    @POST("/addNews")
    fun addNews(
        @Part("title") title: RequestBody,
        @Part("content") content: RequestBody,
        @Part("category") category: RequestBody,
        @Part("latitude") latitude: RequestBody,
        @Part("longitude") longitude: RequestBody,
        @Part image: MultipartBody.Part
    ): Call<Void>


    @GET("getNews")
    fun getNews(): Call<List<News>>

    @DELETE("deleteNews/{id}")
    fun deleteNews(@Path("id") id: String): Call<Void>

    @PUT("updateNews/{id}")
    fun updateNews(
        @Path("id") id: String,
        @Body news: News
    ): Call<Void>
}
