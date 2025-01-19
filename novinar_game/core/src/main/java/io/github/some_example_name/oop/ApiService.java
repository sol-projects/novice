package io.github.some_example_name.oop;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.*;

import java.util.List;

public interface ApiService {

    // GET Routes
    @GET("news/")
    Call<List<News>> getNews();

    @GET("news/{id}")
    Call<News> getNewsById(@Path("id") String id);

    @GET("news/scrape/{n}")
    Call<Void> scrapeNews(@Path("n") int numberOfArticles);

    @GET("news/scrape/{website}/{n}")
    Call<Void> scrapeNewsFromWebsite(@Path("website") String website, @Path("n") int numberOfArticles);

    @GET("news/categories/{categories}")
    Call<List<News>> getNewsByCategories(@Path("categories") String categories);

    @GET("news/authors/{authors}")
    Call<List<News>> getNewsByAuthors(@Path("authors") String authors);

    @GET("news/location/{location}")
    Call<List<News>> getNewsByLocation(@Path("location") String location);

    @GET("news/website/{website}")
    Call<List<News>> getNewsByWebsite(@Path("website") String website);

    @GET("news/date/before/{date}")
    Call<List<News>> getNewsBeforeDate(@Path("date") String date);

    @GET("news/date/after/{date}")
    Call<List<News>> getNewsAfterDate(@Path("date") String date);

    @GET("news/date/after/{after}/before/{before}")
    Call<List<News>> getNewsBetweenDates(@Path("after") String after, @Path("before") String before);

    @GET("news/title/{title}")
    Call<List<News>> getNewsByTitle(@Path("title") String title);

    @GET("news/content/{content}")
    Call<List<News>> getNewsByContent(@Path("content") String content);

    // POST Routes
    @Multipart
    @POST("news/")
    Call<Void> postNews(
        @Part("title") RequestBody title,
        @Part("content") RequestBody content,
        @Part("categories") RequestBody categories,
        @Part("location") RequestBody location,
        @Part MultipartBody.Part image
    );

    @POST("news/store")
    Call<Void> storeNews(@Body RequestBody requestBody);

    @POST("news/geolang")
    Call<Void> processGeolang(@Body RequestBody requestBody);

    // PUT Routes
    @PUT("news/{id}")
    Call<Void> updateNews(
        @Path("id") String id,
        @Body News news
    );

    @PUT("news/store")
    Call<Void> updateStoredNews(@Body RequestBody requestBody);

    @PUT("news/geolang")
    Call<Void> updateGeolang(@Body RequestBody requestBody);

    // DELETE Routes
    @DELETE("news/{id}")
    Call<Void> deleteNews(@Path("id") String id);
}
