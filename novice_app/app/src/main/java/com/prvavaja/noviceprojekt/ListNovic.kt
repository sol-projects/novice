package com.prvavaja.noviceprojekt

class ListNovic {
    private val items = mutableListOf<NewsArticle>()

    fun addItem(novica: NewsArticle) {
        items.add(novica)
    }
    fun updateItem(updatedNovica: NewsArticle) {
        val index = items.indexOfFirst { it._id == updatedNovica._id }
        if (index != -1) {
            items[index] = updatedNovica
        }
    }
    fun removeItem(novica: NewsArticle) {
        items.remove(novica)
    }

    fun getItems(): List<NewsArticle> {
        val sortedItems = items
        for (item in sortedItems) {
            println(item.toString())
        }
        return sortedItems
    }
    fun velikost(): Int {
        return items.size
    }
    fun getLastItem(): NewsArticle {
        return items.last()
    }
}