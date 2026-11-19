package com.shadowzero.remote

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val settings = Settings(applicationContext)
        setContent {
            MaterialTheme {
                val api = remember { Api("", "") }
                var ready by remember { mutableStateOf(false) }
                LaunchedEffect(Unit) {
                    api.baseUrl = settings.url.first()
                    api.token = settings.token.first()
                    ready = api.baseUrl.isNotEmpty() && api.token.isNotEmpty()
                }
                if (!ready) {
                    PairScreen(settings, api) { ready = true }
                } else {
                    MainScreen(api, settings) { ready = false }
                }
            }
        }
    }
}

@Composable
fun PairScreen(settings: Settings, api: Api, onPaired: () -> Unit) {
    val scope = rememberCoroutineScope()
    var url by remember { mutableStateOf("http://192.168.1.100:8081") }
    var pin by remember { mutableStateOf("") }
    var err by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text("Connect to ShadowZero", style = MaterialTheme.typography.headlineSmall)
            Text(
                "Start shadowzero-serve on the PC, open http://<PC-ip>:8081 there and press Pair to get a 4-digit PIN. " +
                    "Enter that PIN here to pair this phone.",
                style = MaterialTheme.typography.bodyMedium
            )
            OutlinedTextField(
                value = url, onValueChange = { url = it }, label = { Text("PC address") },
                singleLine = true, modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = pin, onValueChange = { pin = it.filter { c -> c.isDigit() }.take(4) },
                label = { Text("PIN (4 digits)") }, singleLine = true,
                visualTransformation = PasswordVisualTransformation(), modifier = Modifier.fillMaxWidth()
            )
            Button(
                enabled = !busy && url.isNotBlank() && pin.length == 4,
                onClick = {
                    busy = true; err = null
                    scope.launch {
                        try {
                            val tok = api.pair(url, pin)
                            settings.save(url, tok)
                            onPaired()
                        } catch (e: Exception) {
                            err = e.message ?: e.toString()
                        } finally { busy = false }
                    }
                }
            ) { Text(if (busy) "Pairing..." else "Pair") }
            err?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(api: Api, settings: Settings, onUnpair: () -> Unit) {
    val app = (LocalContext.current.applicationContext as android.app.Application)
    val model: ChatModel = viewModel(factory = ChatModelFactory(api, app))
    val tab = remember { mutableStateOf(0) }
    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = tab.value == 0, onClick = { tab.value = 0 },
                    icon = { Icon(Icons.Default.Person, "Chat") }, label = { Text("Chat") })
                NavigationBarItem(
                    selected = tab.value == 1, onClick = { tab.value = 1 },
                    icon = { Icon(Icons.Default.Info, "Status") }, label = { Text("Status") })
                NavigationBarItem(
                    selected = tab.value == 2, onClick = { tab.value = 2 },
                    icon = { Icon(Icons.AutoMirrored.Filled.List, "Memory") }, label = { Text("Memory") })
            }
        }
    ) { pad ->
        when (tab.value) {
            0 -> ChatScreen(model, Modifier.padding(pad))
            1 -> StatusScreen(api, settings, onUnpair, Modifier.padding(pad))
            else -> MemoryScreen(api, Modifier.padding(pad))
        }
    }
}
