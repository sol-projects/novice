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
    }

}