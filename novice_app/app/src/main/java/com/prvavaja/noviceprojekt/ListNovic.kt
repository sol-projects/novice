package com.prvavaja.noviceprojekt

class ListNovic {
    private val items = mutableListOf<Novica>()

    fun addItem(novica: Novica) {
        items.add(novica)
    }
    fun updateItem(updatedNovica: Novica) {
        val index = items.indexOfFirst { it._id == updatedNovica._id }
        if (index != -1) {
            items[index] = updatedNovica
        }
    }
    // Remove a news from the list
    fun removeItem(novica: Novica) {
        items.remove(novica)
    }

    // Get the list of items in the shopping cart
    fun getItems(): List<Novica> {
        val sortedItems = items
        for (item in sortedItems) {
            println(item.toString())
        }
        return sortedItems
    }
    fun velikost(): Int {
        return items.size
    }
    fun getLastItem(): Novica {
        return items.last()
    }
}