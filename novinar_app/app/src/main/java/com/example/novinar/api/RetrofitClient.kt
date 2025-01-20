package com.example.novinar.api

import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory


object RetrofitClient {
    private const val BASE_URL = "http://192.168.94.168:8000/"

    private val client = OkHttpClient.Builder()
        .addInterceptor(object : Interceptor {
            override fun intercept(chain: Interceptor.Chain): Response {
                val request = chain.request().newBuilder()
                    .addHeader(
                        "Authorization",
                        "Bearer " +
                                "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YzAwNjE0NC1lZTU4LTExZWQtYTA1Yi0wMjQyYWMxMjAwMDMiLCJyb2xlIjoiYWRtaW4iLCJpYXQiOjE3MzczNjcwNzQsImV4cCI6MTczNzM3MDY3NH0.JiYPWazYL758pio3ueNCWsCLtcQmmwhkmebQ4s4ggErSjaRiAzjS1WNcgqQ4PnXYHxyn9QeKP0JyBkMTuV6pjUM5XsBh7hSWshnHiDrShiP-nM2UWQYWbfyKSpofRv7VJMIIkgzt7rFYHC9nCez-jzb3mxacmHr4pE4a5zsD2cXU7MeEfGcebaccWXUpKsXKJVdusuILGoQvoHBvPBgpI2h-sWCNk9cCZt1bAoIuNyD6RbnnPQtgf5PnrvuSjzEDJM7QPnGtXTtJjHD5hnWosKwMt3krrvINIB2JLZXf8wONZGqlzH8Jru60kAPhZv_rs0MwkKyyR7p0e8ajUl-nLg" /*Run to get JWT KEY uuid is in .env file curl -X POST \ http://192.168.1.119:8000/news/login \ -H "Content-Type: application/json" \ -d '{ "uuid": "" }'*/

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