plugins {
    id("com.android.application") version "9.4.0" apply false
    // AGP 9 built-in Kotlin: kotlin-android is gone; compose/serialization plugins apply on top of it.
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.10" apply false
    id("org.jetbrains.kotlin.plugin.serialization") version "2.2.10" apply false
}
