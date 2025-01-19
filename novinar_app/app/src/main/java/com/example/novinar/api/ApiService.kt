package com.example.novinar.api

import com.example.novinar.News
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Call
import retrofit2.http.*


interface ApiService {

    @GET("news/")
    fun getNews(): Call<List<News>>

    @Multipart
    @POST("news/")
    fun postNews(
        @Part("title") title: RequestBody,
        @Part("content") content: RequestBody,
        @Part("categories") categories: RequestBody,
        @Part("location") location: RequestBody,
        @Part image: MultipartBody.Part?
    ): Call<Void>

    @POST("news/")
    fun postNews(@Body requestBody: RequestBody): Call<Void>

    @DELETE("news/{id}")
    fun deleteNews(@Path("id") id: String): Call<Void>

    @Multipart
    @POST("news/upload")
    fun uploadNewsWithPhoto(
        @Part("title") title: RequestBody,
        @Part("content") content: RequestBody,
        @Part photo: MultipartBody.Part
    ): Call<News>


    @PUT("news/{id}")
    fun updateNews(
        @Path("id") id: String,
        @Body news: News
    ): Call<Void>
}
