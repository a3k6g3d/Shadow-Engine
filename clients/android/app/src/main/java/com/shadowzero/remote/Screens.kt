package com.shadowzero.remote

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.launch
import kotlinx.serialization.json.*

@Composable
fun ChatScreen(model: ChatModel, mod: Modifier) {
    val scope = rememberCoroutineScope()
    val msgs by model.msgs.collectAsState()
    val stream by model.stream.collectAsState()
    val cards by model.cards.collectAsState()
    val busy by model.busy.collectAsState()
    val err by model.error.collectAsState()
    var input by remember { mutableStateOf(TextFieldValue("")) }

    Column(mod) {
        LazyColumn(
            Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(msgs) { m ->
                if (m.role == "assistant" || m.role == "user") {
                    Bubble(if (m.role == "user") "You" else "Shadow", m.content,
                        bg = if (m.role == "user") MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.surfaceVariant)
                } else if (m.role == "tool") {
                    Text(m.content, fontSize = androidx.compose.ui.unit.TextUnit.Unspecified,
                        modifier = Modifier.fillMaxWidth(), color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (stream.isNotEmpty()) {
                item { Bubble("Shadow", stream, bg = MaterialTheme.colorScheme.secondaryContainer) }
            }
            cards.forEach { c ->
                item {
                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(10.dp), Arrangement.spacedBy(4.dp)) {
                            Text("${c.state}  ${c.name}", style = MaterialTheme.typography.titleSmall)
                            Text(c.args, fontFamily = FontFamily.Monospace)
                            if (c.output.isNotEmpty()) {
                                Text(c.output, fontFamily = FontFamily.Monospace)
                            }
                            if (c.state == "waiting") {
                                Row(Modifier, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(onClick = { model.approve(c); model.send("") }) { Text("Approve & continue") }
                                }
                            }
                        }
                    }
                }
            }
            err?.let { item { Text(it, color = MaterialTheme.colorScheme.error) } }
        }
        Row(
            Modifier
                .fillMaxWidth()
                .padding(8.dp), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            OutlinedTextField(
                value = input, onValueChange = { input = it },
                modifier = Modifier.weight(1f), placeholder = { Text("Ask Shadow...") },
                maxLines = 4
            )
            if (busy) {
                OutlinedButton(onClick = { model.stop() }) { Text("Stop") }
            } else {
                Button(onClick = {
                    val t = input.text; input = TextFieldValue("")
                    model.send(t)
                }) { Text("Send") }
            }
            IconButton(onClick = { model.newChat() }) { Icon(Icons.Default.Add, "New chat") }
        }
    }
}

@Composable
fun Bubble(who: String, text: String, bg: androidx.compose.ui.graphics.Color) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(bg, RoundedCornerShape(12.dp))
            .padding(10.dp)
    ) {
        Text(who, style = MaterialTheme.typography.labelSmall)
        Text(text)
    }
}

@Composable
fun StatusScreen(api: Api, settings: Settings, onUnpair: () -> Unit, mod: Modifier) {
    val scope = rememberCoroutineScope()
    val ctx = LocalContext.current
    var st by remember { mutableStateOf<JsonObject?>(null) }
    var err by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        while (true) {
            try {
                st = api.status().jsonObject
                err = null
            } catch (e: Exception) {
                err = e.message
            }
            kotlinx.coroutines.delay(5000)
        }
    }
    Column(
        mod
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(16.dp), Arrangement.spacedBy(8.dp)
    ) {
        Text("PC status", style = MaterialTheme.typography.headlineSmall)
        err?.let { Text("not reachable: $it", color = MaterialTheme.colorScheme.error) }
        st?.let { o ->
            fun s(k: String) = o[k]?.jsonPrimitive?.contentOrNull ?: "-"
            fun d(k: String) = o[k]?.jsonPrimitive?.doubleOrNull
            Text("state: ${s("state")}   busy: ${s("busy")}   queued: ${s("queued")}")
            Text("model: ${s("model")}")
            Text("requests: ${s("requests")}  prompt tok: ${s("prompt_tokens")}  cached: ${s("cached_tokens")}  out tok: ${s("completion_tokens")}")
            d("decode_tokens_per_second")?.let { Text("decode: %.1f tok/s".format(it)) }
            d("uptime_s")?.let { Text("uptime: %.0f s".format(it)) }
            d("available_ram_gib")?.let { Text("free RAM: %.1f GiB".format(it)) }
            Text("n_ctx: ${s("n_ctx")}  session tok: ${s("session_tokens")}")
        }
        Row(Modifier, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = !busy, onClick = {
                busy = true; scope.launch {
                    try { api.enginePark() } catch (e: Exception) { err = e.message }
                    busy = false
                }
            }) { Text("Park engine") }
            Button(enabled = !busy, onClick = {
                busy = true; scope.launch {
                    try { api.engineResume() } catch (e: Exception) { err = e.message }
                    busy = false
                }
            }) { Text("Resume engine") }
        }
        OutlinedButton(onClick = {
            scope.launch { settings.clear(); onUnpair() }
        }) { Text("Unpair this phone") }
    }
}

@Composable
fun MemoryScreen(api: Api, mod: Modifier) {
    val scope = rememberCoroutineScope()
    var notes by remember { mutableStateOf<JsonArray>(buildJsonArray {}) }
    var err by remember { mutableStateOf<String?>(null) }
    var kind by remember { mutableStateOf("user") }
    var text by remember { mutableStateOf("") }
    suspend fun refresh() {
        try { notes = api.memoryList().jsonArray; err = null } catch (e: Exception) { err = e.message }
    }
    LaunchedEffect(Unit) { refresh() }
    Column(
        mod
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(16.dp), Arrangement.spacedBy(8.dp)
    ) {
        Text("Memory on the PC", style = MaterialTheme.typography.headlineSmall)
        err?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        notes.forEach { n ->
            val o = n.jsonObject
            val id = o["id"]?.jsonPrimitive?.longOrNull
            val k = o["kind"]?.jsonPrimitive?.contentOrNull ?: "user"
            val t = o["text"]?.jsonPrimitive?.contentOrNull ?: ""
            Card(Modifier.fillMaxWidth()) {
                Row(Modifier.padding(10.dp), Arrangement.spacedBy(8.dp), Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("[$k]", style = MaterialTheme.typography.labelSmall)
                        Text(t)
                    }
                    if (id != null) {
                        IconButton(onClick = {
                            scope.launch {
                                try { api.memoryDelete(id); refresh() } catch (e: Exception) { err = e.message }
                            }
                        }) { Icon(Icons.Default.Delete, "Forget") }
                    }
                }
            }
        }
        Text("Add a note", style = MaterialTheme.typography.titleSmall)
        Row(Modifier, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf("user", "project", "preference", "reference").forEach { k ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(selected = kind == k, onClick = { kind = k })
                    Text(k)
                }
            }
        }
        OutlinedTextField(value = text, onValueChange = { text = it },
            modifier = Modifier.fillMaxWidth(), label = { Text("What to remember") })
        Button(onClick = {
            scope.launch {
                try { api.memoryAdd(kind, text); text = ""; refresh() } catch (e: Exception) { err = e.message }
            }
        }) { Text("Remember") }
    }
}
