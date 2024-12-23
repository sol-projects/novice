package com.example.novinar

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

object NewsRepository {
    private const val PREFS_NAME = "news_prefs"
    private const val NEWS_KEY = "news_list"
    private val newsList = mutableListOf<News>()

    fun addNews(news: News) {
        newsList.add(news)
        saveNewsList()
    }

    fun getNewsList(): List<News> {
        return newsList
    }

    fun editNews(index: Int, updatedNews: News) {
        if (index in newsList.indices) {
            newsList[index] = updatedNews
            saveNewsList()
        }
    }

    fun deleteNews(index: Int) {
        if (index in newsList.indices) {
            newsList.removeAt(index)
            saveNewsList()
        }
    }

    fun loadNewsList() {
        val prefs = App.context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val json = prefs.getString(NEWS_KEY, null)
        if (!json.isNullOrEmpty()) {
            val type = object : TypeToken<MutableList<News>>() {}.type
            val loadedList: MutableList<News> = Gson().fromJson(json, type)
            newsList.clear()
            newsList.addAll(loadedList)
        }
    }

    private fun saveNewsList() {
        val prefs = App.context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val editor = prefs.edit()
        val json = Gson().toJson(newsList)
        editor.putString(NEWS_KEY, json)
        editor.apply()
    }
}
