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
    // Remove a news from the list
    fun removeItem(novica: NewsArticle) {
        items.remove(novica)
    }

    // Get the list of items in the shopping cart
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