package com.prvavaja.noviceprojekt

import android.app.Application
import android.util.Log
class MyAplication:  Application() {
    var list=ListNovic()
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
        var novica1=Novica("Naslov1","https://example1.com",currentDate,authors,"nekaj se dogaja1",categories,location,"123",1)
        var novica2=Novica("Naslov2","https://example2.com",currentDate,authors,"nekaj se dogaja2",categories,location,"124",2)
        var novica3=Novica("Naslov3","https://example3.com",currentDate,authors,"nekaj se dogaja3",categories,location,"125",3)
        list.addItem(novica1)
        list.addItem(novica2)
        list.addItem(novica3)
    }

}