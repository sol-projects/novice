package com.example.novinar.api

import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory


object RetrofitClient {
    private const val BASE_URL = "http://192.168.1.119:8000/"

    private val client = OkHttpClient.Builder()
        .addInterceptor(object : Interceptor {
            override fun intercept(chain: Interceptor.Chain): Response {
                val request = chain.request().newBuilder()
                    .addHeader(
                        "Authorization",
                        "Bearer " +
                                "JWT KEY " /*Run to get JWT KEY uuid is in .env file curl -X POST \ http://192.168.1.119:8000/news/login \ -H "Content-Type: application/json" \ -d '{ "uuid": "" }'*/

                    )
                    .build()
                return chain.proceed(request)
            }
        })
        .build()

    val apiService: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }
}