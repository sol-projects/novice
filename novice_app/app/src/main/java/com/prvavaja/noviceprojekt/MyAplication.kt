package com.prvavaja.noviceprojekt

import android.app.Application
import android.util.Log
import io.realm.kotlin.Realm
import io.realm.kotlin.mongodb.App
import io.realm.kotlin.mongodb.User

class MyAplication:  Application() {
    var list=ListNovic()
    lateinit var realm: Realm
    lateinit var realm_app: App
    lateinit var user: User
    override fun onCreate() {
        super.onCreate()
        Log.d("MyApplication", "Application onCreate called")
        // Create authors and categories lists
        val authors = listOf("Author1", "Author2")
        val categories = listOf("Category1", "Category2")

        // Create a Location object
        val location = Location(type = "Point", coordinates = Pair(123.45, 67.89))

        // Create a Date object (replace this with the actual date you want to use)
        val currentDate = java.util.Date()
        var novica1=NewsArticle("Naslov1","https://example1.com",currentDate,authors,"nekaj se dogaja1",categories,location,"123",1)
        var novica2=NewsArticle("Naslov2","https://example2.com",currentDate,authors,"nekaj se dogaja2",categories,location,"124",2)
        var novica3=NewsArticle("Naslov3","https://example3.com",currentDate,authors,"nekaj se dogaja3",categories,location,"125",3)
        list.addItem(novica1)
        list.addItem(novica2)
        list.addItem(novica3)
    }

}