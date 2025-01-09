package com.example.novinar.api
import com.example.novinar.News
import retrofit2.Call
import retrofit2.http.*

interface ApiService {
    @GET("/getNews")
    fun getNews(): Call<List<News>>

    @POST("/addNews")
    fun addNews(@Body news: News): Call<Void>

    @PUT("/editNews/{id}")
    fun editNews(@Path("id") id: String, @Body news: News): Call<Void>

    @DELETE("/deleteNews/{id}")
    fun deleteNews(@Path("id") id: String): Call<Void>
}
