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
                                "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YzAwNjE0NC1lZTU4LTExZWQtYTA1Yi0wMjQyYWMxMjAwMDMiLCJyb2xlIjoiYWRtaW4iLCJpYXQiOjE3MzczMjU1OTUsImV4cCI6MTczNzMyOTE5NX0.STWb2PQotXjGnxkZb0QyQ5UoQCk4ewzpWtzNpt88j_73oqHbel9uSiFSp1crGz2u-0XpDnY-eTPTjCSDaE_ASIqbPDpYL3nDv4ptAWy46NoRPzl_Q9FtTe187T7iRVf8-je7HbFdyPqH5pmNJGfLPcx5gCqZvNgL7TS3iwj8oY4N4CWVlNYNrCVyMFrV3pXQH1QqgYNoa3j-YCx_BCzw9WnvY2UUC9TJM_smp8_bNc8UHwNpVOOYU_VqOCD1GlZjNU5NDOXAtHDNCpKQNlg2HQYrnbp-dJh-vy5net8sNz-IcTqErORu-ONXuVKM9d94fXJYJ_n2FlqsjjDgvppR8A" /*Run to get JWT KEY uuid is in .env file curl -X POST \ http://192.168.1.119:8000/news/login \ -H "Content-Type: application/json" \ -d '{ "uuid": "" }'*/

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