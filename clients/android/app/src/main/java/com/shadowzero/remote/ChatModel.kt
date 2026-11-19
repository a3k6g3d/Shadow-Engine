package com.shadowzero.remote

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.*

data class Msg(val role: String, val content: String, val tools: String = "")
data class ToolCard(val name: String, val args: String, var state: String = "waiting", var output: String = "")

/** Drives one client-side tool loop against shadowzero-serve, the way the embedded UI does. */
class ChatModel(private val api: Api, private val app: Application) : ViewModel() {

    private val _msgs = MutableStateFlow<List<Msg>>(emptyList())
    val msgs: StateFlow<List<Msg>> = _msgs
    private val _stream = MutableStateFlow("")
    val stream: StateFlow<String> = _stream
    private val _cards = MutableStateFlow<List<ToolCard>>(emptyList())
    val cards: StateFlow<List<ToolCard>> = _cards
    private val _busy = MutableStateFlow(false)
    val busy: StateFlow<Boolean> = _busy
    private val _awaiting = MutableStateFlow(false)
    val awaiting: StateFlow<Boolean> = _awaiting
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error
    private val _title = MutableStateFlow("New chat")
    val title: StateFlow<String> = _title

    private var sessionId: String? = null
    private var runJob: Job? = null
    private var pending: List<ToolCard> = emptyList()
    private var lastRaw = ""
    private var approved: MutableSet<String> = mutableSetOf()

    fun send(text: String) {
        if (_busy.value) return
        if (text.isNotBlank()) _msgs.value = _msgs.value + Msg("user", text)
        _error.value = null
        _awaiting.value = false
        startRun()
    }

    /** Run the tool cards the user approved/denied, then let the model continue. */
    fun runTools() {
        if (_busy.value) return
        val waiting = pending.filter { it.state == "waiting" }
        if (waiting.isEmpty()) return
        _error.value = null
        _awaiting.value = false
        _busy.value = true
        ChatRunService.start(app)
        runJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                _msgs.value = _msgs.value + Msg("assistant", lastRaw, sig(pending))
                val outs = waiting.map { exec(it) }
                _msgs.value = _msgs.value + outs
                loop()
                persist()
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                _error.value = e.message ?: e.toString()
            } finally {
                _busy.value = false; _stream.value = ""; ChatRunService.stop(app)
            }
        }
    }

    fun approve(card: ToolCard) {
        approved.add(card.name)
        _cards.value = _cards.value.map { if (it === card) it.copy(state = "approved") else it }
        pending = _cards.value
    }

    fun deny(card: ToolCard) {
        _cards.value = _cards.value.map { if (it === card) it.copy(state = "denied") else it }
        pending = _cards.value
    }

    fun approveAll() {
        pending.forEach { approved.add(it.name) }
        _cards.value = _cards.value.map { if (it.state == "waiting") it.copy(state = "approved") else it }
        pending = _cards.value
    }

    fun stop() {
        runJob?.cancel()
        _busy.value = false; _awaiting.value = false; _stream.value = ""
        ChatRunService.stop(app)
    }

    fun newChat() {
        stop()
        _msgs.value = emptyList(); _cards.value = emptyList(); pending = emptyList()
        sessionId = null; _title.value = "New chat"
    }

    fun loadHistory(history: JsonArray) {
        stop()
        _msgs.value = history.mapNotNull { el ->
            val o = el.jsonObject
            val role = o["role"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
            if (role == "system") return@mapNotNull null
            Msg(role, o["content"]?.jsonPrimitive?.contentOrNull ?: "")
        }
        _cards.value = emptyList(); pending = emptyList()
    }

    fun setSession(id: String?, title: String?) {
        sessionId = id
        if (!title.isNullOrBlank()) _title.value = title
    }

    private fun startRun() {
        _busy.value = true
        ChatRunService.start(app)
        runJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                loop()
                persist()
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                _error.value = e.message ?: e.toString()
            } finally {
                _busy.value = false; _stream.value = ""; ChatRunService.stop(app)
            }
        }
    }

    private suspend fun loop() {
        val tools = try { api.tools().jsonArray } catch (e: Exception) { buildJsonArray {} }
        var steps = 0
        while (steps < 12) {
            steps++
            _stream.value = ""
            _cards.value = emptyList()
            val step = api.chatStep(historyForModel(), tools) { c -> _stream.value = _stream.value + c }
            _stream.value = ""
            lastRaw = step.content
            val calls = step.toolCalls
            if (calls.isEmpty()) {
                if (step.content.isNotBlank()) _msgs.value = _msgs.value + Msg("assistant", step.content)
                return
            }
            pending = calls.map { ToolCard(it.name, it.arguments) }
            _cards.value = pending
            val auto = pending.filter { approved.contains(it.name) || it.name in AUTO_TOOLS }
            if (auto.size != pending.size) {
                _awaiting.value = true
                _error.value = "Shadow wants to run ${pending.size - auto.size} tool(s); approve or deny them, then press Run."
                _busy.value = false
                ChatRunService.stop(app)
                return
            }
            _msgs.value = _msgs.value + Msg("assistant", step.content, sig(pending))
            val outs = pending.map { exec(it) }
            _msgs.value = _msgs.value + outs
        }
        _error.value = "stopped after 12 tool steps"
    }

    private suspend fun exec(card: ToolCard): Msg {
        _cards.value = _cards.value.map { if (it === card) it.copy(state = "running") else it }
        pending = _cards.value
        val out = if (card.state == "denied") {
            "tool_response\nuser denied this tool; it did not run.\n$TOOL_CLOSE"
        } else {
            try {
                val r = api.executeTool(card.name, card.args)
                val txt = r["output"]?.jsonPrimitive?.contentOrNull ?: ""
                val isErr = r["is_error"]?.jsonPrimitive?.booleanOrNull == true
                card.state = if (isErr) "error" else "done"
                card.output = txt.take(4000)
                "tool_response\n$txt\n$TOOL_CLOSE"
            } catch (e: Exception) {
                card.state = "error"; card.output = "${e.message}"
                "tool_response\nerror: ${e.message}\n$TOOL_CLOSE"
            }
        }
        _cards.value = _cards.value.map { if (it === card) it.copy(state = card.state, output = card.output) else it }
        pending = _cards.value
        return Msg("tool", out)
    }

    private fun sig(cards: List<ToolCard>): String =
        cards.joinToString("\n") { "${it.state} ${it.name} ${it.args}" }

    private suspend fun historyForModel(): JsonArray = buildJsonArray {
        add(buildJsonObject { put("role", "system"); put("content", systemPrompt()) })
        _msgs.value.forEach { m ->
            add(buildJsonObject { put("role", m.role); put("content", m.content) })
        }
    }

    private suspend fun systemPrompt(): String = buildString {
        appendLine("You are Shadow, a direct, accurate AI assistant running locally on Windows. " +
            "Keep answers as concise as the question allows, state clearly when unsure, and adopt any task role the user requests.")
        appendLine("## Action & Tool Rules")
        appendLine("- Act, don't describe: when asked to inspect, test or modify files, execute the tools directly rather than explaining what you would do.")
        appendLine("- Local tools: read_file, glob, grep, search, edit_file, write_file, cmd (PowerShell) and any mcp__ tools given.")
        appendLine("- Web access: use web_search and web_fetch for anything current or uncertain (and cite the page).")
        append("When asked to look at, test or change code, use the tools instead of just describing them. Work in small steps and give a clear final answer.")
        try {
            val mem = api.memoryList().jsonArray
            if (mem.isNotEmpty()) {
                appendLine("\n\nWhat you remember about the user (use it naturally, ignore what does not apply):")
                mem.forEach { n ->
                    val o = n.jsonObject
                    val kind = o["kind"]?.jsonPrimitive?.contentOrNull ?: "user"
                    val text = o["text"]?.jsonPrimitive?.contentOrNull ?: ""
                    appendLine("- [$kind] $text")
                }
            }
        } catch (_: Exception) {
        }
    }

    private suspend fun persist() {
        val hist = buildJsonArray {
            _msgs.value.forEach { m ->
                add(buildJsonObject { put("role", m.role); put("content", m.content) })
            }
        }
        try {
            val body = buildJsonObject {
                put("history", hist)
                sessionId?.let { put("id", it) }
            }
            val resp = api.sessionSave(body).jsonObject
            sessionId = resp["id"]?.jsonPrimitive?.contentOrNull ?: sessionId
        } catch (_: Exception) {
        }
    }

    companion object {
        val AUTO_TOOLS = setOf(
            "web_search", "web_fetch", "read_file", "glob", "grep", "search", "recall",
            "memory_list", "memory_save", "memory_forget", "notes", "finish"
        )
        const val TOOL_CLOSE = "\n</tool_response>"
    }
}

class ChatModelFactory(private val api: Api, private val app: Application) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = ChatModel(api, app) as T
}
