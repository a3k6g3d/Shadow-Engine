package com.shadowzero.remote

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "settings")

val KEY_URL = stringPreferencesKey("url")
val KEY_TOKEN = stringPreferencesKey("token")

/** Saved server address + pairing token. */
class Settings(private val ctx: Context) {
    val url: Flow<String> = ctx.dataStore.data.map { it[KEY_URL] ?: "" }
    val token: Flow<String> = ctx.dataStore.data.map { it[KEY_TOKEN] ?: "" }

    suspend fun save(url: String, token: String) {
        ctx.dataStore.edit {
            it[KEY_URL] = url.trimEnd('/')
            it[KEY_TOKEN] = token
        }
    }

    suspend fun clear() {
        ctx.dataStore.edit { it.clear() }
    }
}
