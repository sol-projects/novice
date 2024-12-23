package com.example.novinar

import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

object NotificationHelper {
    private const val CHANNEL_ID = "news_channel"
    private const val CHANNEL_NAME = "News Notifications"
    private const val CHANNEL_DESCRIPTION = "Notifications for news updates"

    fun createNotificationChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = CHANNEL_DESCRIPTION
            }

            val notificationManager: NotificationManager =
                context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    @SuppressLint("MissingPermission")
    fun sendNotification(
        context: Context,
        title: String,
        content: String,
        intent: android.app.PendingIntent
    ) {
        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle(title)
            .setContentText(content)
            .setContentIntent(intent)
            .setAutoCancel(true)
            .build()

        with(NotificationManagerCompat.from(context)) {
            notify(
                System.currentTimeMillis().toInt(),
                notification
            )
        }
    }
}
