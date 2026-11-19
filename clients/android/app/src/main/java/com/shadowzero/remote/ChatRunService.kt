package com.shadowzero.remote

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

/** Keeps one chat turn alive while the phone screen is off. */
class ChatRunService : Service() {
    override fun onBind(intent: Intent?) = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(NotificationChannel("run", "Shadow run", NotificationManager.IMPORTANCE_LOW))
        }
        val notif: Notification = NotificationCompat.Builder(this, "run")
            .setContentTitle("Shadow is answering")
            .setContentText("A turn is running on the PC")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(1, notif, android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(1, notif)
        }
        return START_NOT_STICKY
    }

    companion object {
        fun start(ctx: Context) = ctx.startForegroundService(Intent(ctx, ChatRunService::class.java))
        fun stop(ctx: Context) = ctx.stopService(Intent(ctx, ChatRunService::class.java))
    }
}